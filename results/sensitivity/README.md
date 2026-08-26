# Sensitivity findings

This directory is generated from 30 independent seeds and 180 nonempty trace
snapshots per run.

## Independent-agent reconciliation delay

Deadline-satisfaction means at zero added reconciliation delay are 97.01%,
87.67%, and 75.90% for 10, 25, and 50 vehicles. AAI-CDOS reaches 99.17%, 97.28%,
and 89.22% under the same workloads, so its gains remain 2.16, 9.61, and 13.31
percentage points even when this baseline penalty is removed.

The default main experiment uses 80 ms; the size of the gain must therefore be
reported together with that assumption.

## Negotiation cap

Moving from one to two allowed rounds changes AAI-CDOS deadline satisfaction
from 99.17/97.23/88.82% to 99.17/97.28/89.22%. A third round yields no further
gain in this configuration. The reported `R=3` is a safety cap; it should not be
described as the source of the main performance improvement.
