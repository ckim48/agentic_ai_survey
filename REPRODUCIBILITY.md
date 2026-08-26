# Reproducibility and server migration

## 1. Clone and install

```bash
git clone https://github.com/ckim48/agentic_ai_survey.git
cd agentic_ai_survey
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Transfer the licensed trace separately

Do not commit the NPZ to Git. Copy it through an approved private channel:

```bash
mkdir -p data
scp USER@SOURCE_HOST:/authorized/path/seoul_v2x_trace_evening45.npz \
  data/seoul_v2x_trace_evening45.npz
sha256sum data/seoul_v2x_trace_evening45.npz
```

The expected digest is in `data/README.md`.

## 3. Reproduce policy-level DT results

```bash
python3 scripts/inspect_trace.py
python3 scripts/run_experiments.py --config config/simulation.yaml
python3 scripts/summarize_results.py --config config/simulation.yaml
python3 scripts/run_sensitivity.py --config config/simulation.yaml
python3 scripts/generate_dt_results.py
```

These 30-seed results use the deterministic planner surrogate. They provide
policy-level deadline, network latency, DT age, and position-error results.

## 4. Optional real-GPT-4o audit

Use a newly issued project key only through the process environment:

```bash
export OPENAI_API_KEY="..."
python3 scripts/run_llm_audit.py --smoke
python3 scripts/run_llm_audit.py
unset OPENAI_API_KEY
```

The default audit has a request-attempt cap. Raw `calls.jsonl`, connection-test
metadata, host manifests, and API correlation IDs are ignored by Git. Publish
only `conversation_trace_public.jsonl` after reviewing it.

## Evidence boundary

- `results/full/` and `results/dt_analysis/`: deterministic 30-seed
  policy-level DT outcomes.
- `results/llm_audit/`: bounded 30-event real-model process evidence and
  sanitized public interaction traces.

Do not add cloud API wall time to the reported simulated network E2E latency,
and do not present the bounded model audit as the 30-seed policy experiment.
