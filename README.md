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
