# Digital twin engineering

Uses the [VitalDB](https://physionet.org/content/vitaldb/1.0.0/) dataset, cached locally under
`data/vitaldb` (override with `VITALDB_CACHE_DIR`). Files are downloaded on first use.

```python
from digital_twin_engineering import vitaldb_cache as vdb

clinical = vdb.table("clinical_data")                     # pandas DataFrame
hr = vdb.load_case(1, ["Solar8000/HR"], interval=1)       # numpy array, 1 s grid
vf = vdb.load_vital(1)                                    # full vitaldb.VitalFile
```

Pre-download (all ~95 GB, or specific cases):

```bash
uv run vitaldb-cache download            # foreground
uv run vitaldb-cache download 1 2 3
./download_vitaldb_daemon.sh             # background systemd user service
tail -f data/vitaldb-download.log
```

## Pipeline

Terms are defined in [GLOSSARY.md](GLOSSARY.md); design decisions in [docs/adr/](docs/adr/); what the models achieve in [docs/results.md](docs/results.md).

```bash
uv run dte inventory                          # once: tracks per case + drugs actually given (~12 min)
uv run dte coverage                           # share of cases with each channel
uv run dte explore map bis hr propofol_dose --by department optype   # largest consistent groups
uv run dte cohort cohorts/tiva-artline.yaml   # resolve and freeze a cohort's case list
uv run dte verify configs/default.yaml cohorts/smoke.yaml   # experts vs pump models and vitals
uv run dte train configs/default.yaml cohorts/tiva-artline.yaml --run first
uv run dte train configs/smoke.yaml cohorts/smoke.yaml --run smoke   # ~30 s end-to-end check
uv run dte ablate configs/default.yaml cohorts/tiva-artline.yaml --run ablation   # what each input group adds
uv run dte transfer configs/default.yaml cohorts/tiva-artline.yaml --run transfer # train on one patient group, test on another
uv run dte effects --run sixth                # what-ifs vs what followed the clinicians' target changes
uv run dte replay 13 --run smoke              # follow one case through the twin
uv run dte publish 13 --run smoke             # the same, through MQTT to the twin service (see Live stack)
uv run pytest
```

Stages (each cached per case under `data/derived/`, keyed by its config):

| Module | Does |
|---|---|
| `channels` | tracks -> logical channels on a causal 1 s grid with staleness; training-time sensor faults |
| `inventory`, `cohort` | which cases have which channels; cohort YAMLs and the explorer |
| `experts` | PAS and AnestheSIM (ported) models stepped through each case, each cached on its own; pre-dose baselines; lookaheads and what-ifs; online calibration |
| `labels`, `targets` | hypotension events, interventions, forecast targets (add vitals in `TARGETS`) |
| `features`, `dataset` | decision-point rows: windowed features, covariates, labels |
| `forecasting`, `suggestion` | one LightGBM per (target, horizon); per-drug intervention suggesters |
| `verification` | expert fidelity: Ce vs the pumps' own models, BIS/MAP vs measured |
| `effects`, `transfer` | validation: what-ifs against observed target changes; forecasters across patient groups |
| `evaluation`, `pipeline` | patient-split training, baselines (also on hypotension onsets), metrics in `models/<run>/` |
| `ablation` | retrains the forecasters without each group of inputs |
| `live` | the twin (raw records in, decisions out); the replayer (in-process or to MQTT) and the twin service (MQTT in, InfluxDB out) |

## Live stack

Replay -> Mosquitto (MQTT) -> twin service -> InfluxDB -> Grafana, as in the project plan; message
formats in `live/messages.py`, what the service writes in `live/service.py`.

```bash
RUN=sixth docker compose up -d --build        # broker, database, Grafana, twin service with models/sixth
docker compose run --rm replayer 17           # replay case 17 at 30x (--speed, --until, --start)
# Grafana: http://localhost:3000 (admin/admin), dashboard "Anaesthesia twin"; pick the case at the top

# without Docker for the Python parts (a broker and InfluxDB still needed):
uv run dte serve --run sixth --host localhost --influx http://localhost:8086
uv run dte publish 17 --run sixth --speed 30
```

The dashboard shows each forecast where it comes due, drawn dashed over what was measured; the
hypotension risk and alerts; estimated drug concentrations against the pumps'; the suggestions with
their expected effects; unexplained deviations; and fidelity so far. Runs trained before the alert
threshold was saved (`fifth` and older) need `ALERT_THRESHOLD=0.575` (10% alert rate for `fifth`).
The Python side is tested through an MQTT broker; the InfluxDB queries and the dashboard have not yet
run against a live InfluxDB/Grafana.
After editing `grafana/dashboard.py`, run it to regenerate `grafana/dashboards/twin.json`.
