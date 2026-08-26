# Parameter provenance

| Setting | Value used | Provenance and adaptation |
|---|---:|---|
| DT update size | Uniform 0.6--0.8 Mbit | Li et al., IEEE TVT 2025, Table II. Directly reused. |
| Workload | 300 cycles/bit | Li et al., IEEE TVT 2025, Table II. Directly reused. |
| Vehicle transmit power | Uniform 0.01--0.1 W | Li et al., IEEE TVT 2025, Table II. Directly reused for the default; a 24 dBm upper-bound sensitivity can be configured from Zhang et al., IEEE TVT 2023. |
| Ground edge CPU | 10 GHz per server | Li et al., IEEE TVT 2025, Table II. Reused per deployed edge server. |
| Ground NLoS path loss | 140.7 + 36.7 log10(d[km]) dB | Zhang et al., IEEE TVT 2023. Reused as the urban obstruction model. |
| Shadowing / noise figure | 10 dB / 9 dB | Zhang et al., IEEE TVT 2023. Directly reused. |
| LEO altitude | 780 km | Gao et al., IEEE JSAC 2024. Geometry only. |
| UAV count / altitude | 5 / 100 m | Gao et al., IEEE JSAC 2024. Geometry only. |
| Air carrier / bandwidth | 4 GHz / 400 MHz total | Gao et al., IEEE JSAC 2024. Shared among active UAV links. |
| Satellite bandwidth / CPU | 150 MHz / 200 Gcycles/s | Zhang et al., IEEE Access 2024. Shared satellite capacity. |
| Vehicle counts | 10, 25, 50 | User-supplied manuscript. |
| Negotiation limit | R = 3 | Design choice; sensitivity-ready in YAML. |
| Independent runs | N = 30 | Design choice; supports 95% CI reporting. |
| Replay window | 180 nonempty Gangnam snapshots | Design choice; no synthetic replay inside a run. |
| Independent-agent conflict | 80 ms reconciliation delay | Explicit baseline assumption; configurable and sensitivity-ready. |
| Hard deadlines | 0.15 / 0.35 / 1.0 s | Application-class design choice, not copied from a paper. Must be reported as an assumption. |

Primary links:

- Li et al., IEEE TVT 2025: https://doi.org/10.1109/TVT.2025.3548844
- Zhang et al., IEEE TVT 2023: https://doi.org/10.1109/TVT.2023.3270859
- Liu et al., IEEE TVT 2024: https://doi.org/10.1109/TVT.2023.3312676
- Gao et al., IEEE JSAC 2024: https://doi.org/10.1109/JSAC.2024.3459073
- Zhang et al., IEEE Access 2024: https://doi.org/10.1109/ACCESS.2024.3486564
- Seoul T-data V2X source description: http://t-data.seoul.go.kr/
