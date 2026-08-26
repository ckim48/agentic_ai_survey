"""Minimal, dependency-free OpenAI Responses API client for the LLM audit.

The API key is read only from OPENAI_API_KEY. It is never written to an output
artifact, included in an exception, or accepted as a command-line argument.
"""

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request


class RequestBudgetExceeded(RuntimeError):
    pass


class OpenAIResponsesClient(object):
    def __init__(self, model="gpt-4o", timeout_s=90, max_retries=3,
                 max_total_requests=220, max_output_tokens=160):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._api_key = api_key
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.max_total_requests = int(max_total_requests)
        self.max_output_tokens = int(max_output_tokens)
        self._request_attempts = 0
        self._lock = threading.Lock()

    @property
    def request_attempts(self):
        with self._lock:
            return self._request_attempts

    def _claim_request(self):
        with self._lock:
            if self._request_attempts >= self.max_total_requests:
                raise RequestBudgetExceeded(
                    "maximum OpenAI request budget (%d) reached" %
                    self.max_total_requests)
            self._request_attempts += 1

    @staticmethod
    def _output_text(response):
        chunks = []
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    chunks.append(content.get("text", ""))
                elif content.get("type") == "refusal":
                    raise RuntimeError("model refused the structured planning request")
        if not chunks:
            raise RuntimeError("Responses API returned no output_text")
        return "".join(chunks)

    def call_json(self, instructions, input_object, schema_name, schema,
                  metadata=None):
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": json.dumps(input_object, separators=(",", ":"),
                                sort_keys=True),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "description": "A bounded orchestration decision.",
                    "strict": True,
                    "schema": schema,
                }
            },
            "max_output_tokens": self.max_output_tokens,
            "temperature": 0,
            "store": False,
        }
        if metadata:
            payload["metadata"] = {
                str(k)[:64]: str(v)[:512] for k, v in metadata.items()
            }
        body = json.dumps(payload).encode("utf-8")
        retryable = {408, 409, 429, 500, 502, 503, 504}
        last_error = None
        for attempt in range(self.max_retries + 1):
            self._claim_request()
            request = urllib.request.Request(
                "https://api.openai.com/v1/responses",
                data=body,
                method="POST",
                headers={
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": "application/json",
                    "User-Agent": "aai-cdos-llm-audit/1.0",
                },
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as r:
                    response = json.loads(r.read().decode("utf-8"))
                    request_id = r.headers.get("x-request-id", "")
                elapsed_s = time.perf_counter() - started
                result = json.loads(self._output_text(response))
                usage = response.get("usage") or {}
                return {
                    "result": result,
                    "response_id": response.get("id", ""),
                    "request_id": request_id,
                    "returned_model": response.get("model", ""),
                    "status": response.get("status", ""),
                    "service_tier": response.get("service_tier", ""),
                    "latency_s": elapsed_s,
                    "input_tokens": int(usage.get("input_tokens", 0) or 0),
                    "output_tokens": int(usage.get("output_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                    "retry_index": attempt,
                }
            except urllib.error.HTTPError as exc:
                elapsed_s = time.perf_counter() - started
                try:
                    detail = json.loads(exc.read().decode("utf-8"))
                    message = detail.get("error", {}).get("message", "HTTP error")
                except Exception:
                    message = "HTTP error"
                last_error = RuntimeError(
                    "OpenAI API HTTP %d after %.2fs: %s" %
                    (exc.code, elapsed_s, message[:300]))
                if exc.code not in retryable or attempt >= self.max_retries:
                    raise last_error
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = RuntimeError("OpenAI API transport error: %s" % exc)
                if attempt >= self.max_retries:
                    raise last_error
            time.sleep(min(8.0, (2.0 ** attempt) + random.random()))
        raise last_error


def object_schema(properties, required=None):
    """Build the strict JSON-schema subset used by GPT-4o."""
    return {
        "type": "object",
        "properties": properties,
        "required": required or list(properties),
        "additionalProperties": False,
    }
