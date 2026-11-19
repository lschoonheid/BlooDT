"""`dte`: the pipeline's command line. Run `uv run dte --help`."""

import argparse
import sys

import pandas as pd

from . import cohort as cohorts
from . import dataset, inventory, pipeline, verification
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


if __name__ == "__main__":
    sys.exit(main())
