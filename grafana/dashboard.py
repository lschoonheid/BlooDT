"""Writes grafana/dashboards/twin.json, the twin's dashboard (run after editing):

    uv run python grafana/dashboard.py

Panels follow the demo script in the project plan: vitals with the twin's
forecasts drawn where they come due, the hypotension warning, estimated drug
concentrations against the pumps', suggestions, fidelity and deviations.
"""

import json
from pathlib import Path

DS = {"type": "influxdb", "uid": "influx-twin"}
OUT = Path(__file__).parent / "dashboards" / "twin.json"


def series(measurement: str, field: str, label: str, where: str = "", fn: str = "last") -> str:
    """A Flux query for one series of the selected case, named `label`."""
    extra = f" and {where}" if where else ""
    return (
        'from(bucket: "twin") |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n'
        f'  |> filter(fn: (r) => r._measurement == "{measurement}" and r.case == "${{case}}" '
        f'and r._field == "{field}"{extra})\n'
        f"  |> aggregateWindow(every: v.windowPeriod, fn: {fn}, createEmpty: false)\n"
        # Grafana names a Flux series after its value column, so the label goes there
        f'  |> keep(columns: ["_time", "_value"]) |> rename(columns: {{_value: "{label}"}})'
    )


def table(measurement: str, rows: list[str], where: str = "") -> str:
    """The latest value of every field, one row per combination of the `rows` tags."""
    extra = f" and {where}" if where else ""
    keys = ", ".join(f'"{r}"' for r in rows)
    return (
        'from(bucket: "twin") |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n'
        f'  |> filter(fn: (r) => r._measurement == "{measurement}" and r.case == "${{case}}"{extra})\n'
        f"  |> last() |> group() |> pivot(rowKey: [{keys}], columnKey: [\"_field\"], valueColumn: \"_value\")\n"
        f'  |> drop(columns: ["_start", "_stop", "_time", "_measurement", "case"])'
    )


def timeseries(title, queries, pos, unit=None, thresholds=None, decimals=None, overrides=()):
    defaults = {"custom": {"spanNulls": 60000, "showPoints": "never"}}
    if unit:
        defaults["unit"] = unit
    if decimals is not None:
        defaults["decimals"] = decimals
    if thresholds:
        defaults["thresholds"] = {
            "mode": "absolute",
            "steps": [{"color": "transparent", "value": None}]
            + [{"color": c, "value": v} for v, c in thresholds],
        }
        defaults["custom"]["thresholdsStyle"] = {"mode": "dashed"}
    return {
        "type": "timeseries",
        "title": title,
        "datasource": DS,
        "gridPos": pos,
        "targets": [{"refId": chr(65 + i), "query": q, "datasource": DS} for i, q in enumerate(queries)],
        "fieldConfig": {"defaults": defaults, "overrides": list(overrides)},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
    }


def dashed(name: str) -> dict:
    """Draw the series `name` dashed (forecasts)."""
    return {
        "matcher": {"id": "byName", "options": name},
        "properties": [{"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [10, 6]}}],
    }


def tablepanel(title, query, pos, decimals=2):
    return {
        "type": "table",
        "title": title,
        "datasource": DS,
        "gridPos": pos,
        "targets": [{"refId": "A", "query": query, "datasource": DS}],
        "fieldConfig": {"defaults": {"decimals": decimals}, "overrides": []},
        "options": {"showHeader": True},
    }


def pos(x, y, w=12, h=8):
    return {"x": x, "y": y, "w": w, "h": h}


FORECAST_5 = 'r.horizon == "300"'

panels = [
    timeseries(
        "Mean arterial pressure",
        [
            series("vitals", "map", "MAP measured"),
            series("forecast", "map_1min", "1-min MAP forecast 5 min ago", FORECAST_5),
            series("vitals", "nibp_map", "NIBP MAP"),
        ],
        pos(0, 0), "mmHg", thresholds=[(65, "red")],
        overrides=[dashed("1-min MAP forecast 5 min ago")],
    ),
    timeseries(
        "Hypotension risk (MAP < 65 for 1 min within the horizon)",
        [
            series("warning", "p@300", "within 5 min"),
            series("warning", "p@60", "within 1 min"),
            series("warning", "p@600", "within 10 min"),
        ],
        pos(12, 0), "percentunit", decimals=2,
    ),
    timeseries(
        "BIS",
        [
            series("vitals", "bis", "BIS measured"),
            series("forecast", "bis", "BIS forecast 5 min ago", FORECAST_5),
            series("expert", "bis", "AnestheSIM (Greco)", 'r.expert == "asim_greco"'),
            series("expert", "bis", "PAS (Eleveld)", 'r.expert == "pas_eleveld"'),
        ],
        pos(0, 8), thresholds=[(40, "green"), (60, "orange")],
        overrides=[dashed("BIS forecast 5 min ago")],
    ),
    timeseries(
        "Heart rate",
        [
            series("vitals", "hr", "HR measured"),
            series("forecast", "hr", "HR forecast 5 min ago", FORECAST_5),
        ],
        pos(12, 8), "bpm", overrides=[dashed("HR forecast 5 min ago")],
    ),
    timeseries(
        "Propofol (µg/mL)",
        [
            series("vitals", "propofol_target", "TCI target"),
            series("vitals", "propofol_ce_pump", "Ce, pump"),
            series("expert", "propofol_ce", "Ce, PAS Marsh", 'r.expert == "pas_marsh"'),
            series("expert", "propofol_ce", "Ce, AnestheSIM", 'r.expert == "asim_greco"'),
        ],
        pos(0, 16),
    ),
    timeseries(
        "Remifentanil (ng/mL)",
        [
            series("vitals", "remifentanil_target", "TCI target"),
            series("vitals", "remifentanil_ce_pump", "Ce, pump"),
            series("expert", "remifentanil_ce", "Ce, PAS Minto", 'r.expert == "pas_marsh"'),
        ],
        pos(12, 16),
    ),
    timeseries(
        "Probability the anaesthetist changes a target within 1 min",
        [
            series("suggestion", "p_decrease", "propofol lower", 'r.drug == "propofol"'),
            series("suggestion", "p_increase", "propofol raise", 'r.drug == "propofol"'),
            series("suggestion", "p_decrease", "remifentanil lower", 'r.drug == "remifentanil"'),
            series("suggestion", "p_increase", "remifentanil raise", 'r.drug == "remifentanil"'),
        ],
        pos(0, 24), "percentunit",
    ),
    tablepanel(
        "Suggested targets and their expected effect in 5 min (AnestheSIM)",
        'from(bucket: "twin") |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n'
        '  |> filter(fn: (r) => r._measurement == "suggestion" and r.case == "${case}" and '
        '(r._field == "setting_decrease" or r._field == "setting_increase" or r._field =~ /^effect_.*\\.(bis|map)@300$/))\n'
        '  |> last() |> group() |> pivot(rowKey: ["drug"], columnKey: ["_field"], valueColumn: "_value")\n'
        '  |> drop(columns: ["_start", "_stop", "_time", "_measurement", "case"])',
        pos(12, 24),
    ),
    timeseries(
        "Deviation: measured minus the forecast made 1 min earlier (flags: orange marks)",
        [
            series("deviation", "residual", "BIS", 'r.signal == "bis"'),
            series("deviation", "residual", "HR", 'r.signal == "hr"'),
            series("deviation", "residual", "1-min MAP", 'r.signal == "map_1min"'),
        ],
        pos(0, 32),
    ),
    tablepanel(
        "Forecast fidelity so far: RMSE, twin vs persistence",
        table("forecast_fidelity", ["signal", "horizon"]),
        pos(12, 32),
    ),
    tablepanel(
        "Expert fidelity over the case (Ce vs pump; BIS, MAP vs measured)",
        table("fidelity", ["expert", "signal"], 'r.window == "case"'),
        pos(0, 40, 24, 9),
    ),
]

dashboard = {
    "uid": "twin",
    "title": "Anaesthesia twin",
    "timezone": "browser",
    "refresh": "5s",
    "time": {"from": "now-15m", "to": "now+3h"},
    "schemaVersion": 39,
    "templating": {
        "list": [
            {
                "name": "case",
                "label": "Case",
                "type": "query",
                "datasource": DS,
                "query": 'import "influxdata/influxdb/schema"\nschema.tagValues(bucket: "twin", tag: "case")',
                "refresh": 2,
            }
        ]
    },
    "annotations": {
        "list": [
            {
                "name": "Unexplained deviation",
                "datasource": DS,
                "enable": True,
                "iconColor": "orange",
                "query": 'from(bucket: "twin") |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n'
                '  |> filter(fn: (r) => r._measurement == "deviation" and r.case == "${case}" and r._field == "flagged" and r._value == true)\n'
                '  |> map(fn: (r) => ({_time: r._time, _value: "deviation: " + r.signal}))',
            },
            {
                "name": "Hypotension alert",
                "datasource": DS,
                "enable": True,
                "iconColor": "red",
                "query": 'from(bucket: "twin") |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n'
                '  |> filter(fn: (r) => r._measurement == "warning" and r.case == "${case}" and r._field == "alert" and r._value == true)\n'
                '  |> map(fn: (r) => ({_time: r._time, _value: "hypotension alert"}))',
            },
        ]
    },
    "panels": panels,
}

if __name__ == "__main__":
    OUT.write_text(json.dumps(dashboard, indent=2) + "\n")
    print(f"-> {OUT}")
