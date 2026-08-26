# Seoul V2X trace

The mobility trace is intentionally not redistributed in this public
repository. Obtain an authorized copy of the Seoul T-data V2X vehicle-status
trace derivative and place it at:

```text
data/seoul_v2x_trace_evening45.npz
```

Expected SHA-256:

```text
2885aa7e90874c4f1a82c2ff4690ae2fcbf45161ec0894b9d7870d1644f18cbf
```

Verify on Linux:

```bash
sha256sum data/seoul_v2x_trace_evening45.npz
python3 scripts/inspect_trace.py
```

The trace provides mobility only. Channel, network, queue, computing, task,
deadline, and agent states are simulated.
