"""`dte`: the pipeline's command line. Run `uv run dte --help`."""

import argparse
import math
import os
import sys
import time

import pandas as pd

from . import cohort as cohorts
from . import ablation, dataset, inventory, pipeline, verification
from .channels import CHANNELS
from .config import Config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dte", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inventory", help="scan every case's tracks and drug usage (once, ~15 min)")
    p.add_argument("-j", "--jobs", type=int, default=16)

    p = sub.add_parser("coverage", help="share of cases that have each channel")
    p.add_argument("--cohort", help="restrict to this cohort YAML")

    p = sub.add_parser("explore", help="largest case groups that have all given channels")
    p.add_argument("channels", nargs="+", choices=sorted(CHANNELS))
    p.add_argument("--by", nargs="+", default=["department", "optype", "agents"])
    p.add_argument("--cohort", help="start from this cohort YAML's criteria")
    p.add_argument("--top", type=int, default=25)

    p = sub.add_parser("cohort", help="resolve a cohort YAML and freeze its case list")
    p.add_argument("cohort")

    p = sub.add_parser("dataset", help="build (and cache) decision-point rows for a cohort")
    p.add_argument("config")
    p.add_argument("cohort")
    p.add_argument("--perturbed", action="store_true")
    p.add_argument("-j", "--jobs", type=int, default=16)

    p = sub.add_parser("verify", help="expert fidelity vs the pumps' models and measured vitals")
    p.add_argument("config")
    p.add_argument("cohort")
    p.add_argument("-j", "--jobs", type=int, default=16)

    p = sub.add_parser("train", help="train and evaluate forecasters and suggesters")
    p.add_argument("config")
    p.add_argument("cohort")
    p.add_argument("--run", required=True, help="name of the output folder under models/")
    p.add_argument("-j", "--jobs", type=int, default=16)

    p = sub.add_parser("ablate", help="retrain forecasters without each group of inputs")
    p.add_argument("config")
    p.add_argument("cohort")
    p.add_argument("--run", required=True, help="name of the output folder under models/")
    p.add_argument("--only", nargs="+", help="variants to run (default: all)")
    p.add_argument("-j", "--jobs", type=int, default=16)

    p = sub.add_parser("replay", help="replay a case through the twin with a trained run's models")
    p.add_argument("caseid", type=int)
    p.add_argument("--run", required=True, help="the training run under models/")
    p.add_argument("--out", help="write the decisions to this Parquet/CSV file")

    p = sub.add_parser("publish", help="replay a case to an MQTT broker for the twin service")
    p.add_argument("caseid", type=int)
    p.add_argument("--run", required=True, help="the training run whose config sets the channels")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--speed", type=float, default=30.0, help="times real time (inf: as fast as possible)")
    p.add_argument("--start", type=float, help="epoch s of case time 0 (default now)")
    p.add_argument("--until", type=float, help="stop at this case time (s)")

    p = sub.add_parser("serve", help="the twin service: MQTT in, InfluxDB out")
    p.add_argument("--run", required=True, help="the training run under models/")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--influx", default=os.environ.get("INFLUX_URL", "http://localhost:8086"))
    p.add_argument("--influx-token", default=os.environ.get("INFLUX_TOKEN", "twin-token"))
    p.add_argument("--influx-org", default=os.environ.get("INFLUX_ORG", "twin"))
    p.add_argument("--influx-bucket", default=os.environ.get("INFLUX_BUCKET", "twin"))
    p.add_argument(
        "--alert-threshold", type=float, default=float(os.environ["ALERT_THRESHOLD"]) if os.environ.get("ALERT_THRESHOLD") else None,
        help="5-min hypotension probability that alerts (default: the run's 10%%-alert-rate threshold)",
    )
    p.add_argument("--deviation-k", type=float, default=3.0, help="flag residuals over k test RMSEs")
    p.add_argument("--once", action="store_true", help="exit after the first case ends")

    p = sub.add_parser("transfer", help="train on one patient group, test on another")
    p.add_argument("config")
    p.add_argument("cohort")
    p.add_argument("--run", required=True, help="output directory under models/")
    p.add_argument("--only", nargs="+", help="splits to run (default: all)")
    p.add_argument("-j", "--jobs", type=int, default=16)

    p = sub.add_parser("effects", help="validate the what-ifs against what followed target changes")
    p.add_argument("--run", required=True, help="the training run under models/ (its test cases)")
    p.add_argument("-j", "--jobs", type=int, default=16)

    args = parser.parse_args(argv)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    base = cohorts.Cohort.from_yaml(args.cohort) if getattr(args, "cohort", None) else None

    match args.command:
        case "inventory":
            df = inventory.build(args.jobs)
            print(f"{df.caseid.nunique()} cases, {df.track.nunique()} distinct tracks -> {inventory.PATH}")
            used = inventory.build_usage(args.jobs)
            print(f"drug usage of {len(used)} cases -> {inventory.USAGE_PATH}")
        case "coverage":
            print(cohorts.channel_coverage(base).to_string(float_format="{:.1%}".format))
        case "explore":
            print(cohorts.explore(args.channels, args.by, base).head(args.top).to_string())
        case "cohort":
            path = base.save_manifest()
            df = base.cases()
            print(f"{len(df)} cases, {df.subjectid.nunique()} patients -> {path}")
        case "dataset":
            rows = dataset.build(base.caseids(), Config.from_yaml(args.config), args.perturbed, args.jobs)
            print(f"{len(rows)} rows from {rows.caseid.nunique()} cases, {rows.shape[1]} columns")
        case "verify":
            per_case = verification.fidelity(base.caseids(), Config.from_yaml(args.config), args.jobs)
            print(verification.summary(per_case).to_string(float_format="{:.2f}".format))
        case "train":
            out = pipeline.train(Config.from_yaml(args.config), base, args.run, args.jobs)
            print(f"models and metrics -> {out}")
        case "replay":
            from .live import replayer

            config, models = replayer.load_run(args.run)
            decisions = replayer.replay(args.caseid, config, models)
            if args.out:
                write = decisions.to_csv if args.out.endswith(".csv") else decisions.to_parquet
                write(args.out)
            print(decisions.iloc[:: max(1, len(decisions) // 30)].to_string(float_format="{:.2f}".format))
        case "publish":
            from .live import replayer

            config, _ = replayer.load_run(args.run)
            start = math.floor(time.time()) if args.start is None else args.start
            print(f"case {args.caseid}: case time 0 is epoch {start:.0f} "
                  f"(Grafana: ?from={start * 1000:.0f}&to=now%2B6h&var-case={args.caseid})", flush=True)
            replayer.replay_mqtt(args.caseid, config, args.host, args.port, args.speed, start, args.until)
        case "serve":
            from .live import influx, service

            sink = influx.InfluxSink(args.influx, args.influx_token, args.influx_org, args.influx_bucket)
            svc = service.from_run(args.run, sink, args.alert_threshold, deviation_k=args.deviation_k)
            print(f"twin service on {args.host}:{args.port} -> {args.influx}", flush=True)
            service.serve(svc, args.host, args.port, args.once)
        case "transfer":
            from . import transfer

            results = transfer.run(Config.from_yaml(args.config), base, args.run, args.only, args.jobs)
            print(transfer.table(results).to_string(index=False, float_format="{:.3f}".format))
        case "effects":
            from . import effects

            result = effects.run(args.run, args.jobs)
            print(effects.summary(result).to_string(index=False, float_format="{:.2f}".format))
        case "ablate":
            results = ablation.run(Config.from_yaml(args.config), base, args.run, args.only, args.jobs)
            for test in results["all"]:
                if test != "features":
                    print(f"\n{test}:\n" + ablation.table(results, test).to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    sys.exit(main())
