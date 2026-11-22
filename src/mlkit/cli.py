"""Command-line interface for mlkit.

Commands::

    mlkit run      CONFIG [--data PATH] [--out DIR] [--html-report FILE]
    mlkit predict  BUNDLE DATA [--out FILE] [--threshold T]
    mlkit monitor  BUNDLE DATA [--out FILE] [--fail-on warn|error]
    mlkit evaluate BUNDLE LABELLED_DATA [--by COL] [--out FILE] [--threshold T]
    mlkit serve    BUNDLE [--host 127.0.0.1] [--port 8000]
    mlkit eda      DATA [--target COL] [--task-type T] [--out report.html] [--profile-json F]
    mlkit benchmark [--case NAME ...] [--no-eda] [--out FILE]   generality suite (exit 1 on failure)
    mlkit steps    list registered pipeline steps

Global flags: ``-v/--verbose``, ``-q/--quiet``, ``--log-json``.
Exit codes: 0 success, 1 user/data error or failed gate, 2 drift detected (monitor) or
metric_thresholds violated on labelled data (evaluate).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mlkit.core.config import TaskConfig
from mlkit.core.errors import MLKitError
from mlkit.core.log import configure_logging, get_logger, run_context
from mlkit.runner import (  # noqa: F401  (re-exported for backwards compatibility)
    build_default_pipeline,
    build_pipeline_from_config,
    evaluate_bundle,
    monitor_bundle,
    predict_bundle,
    read_table,
    train,
    write_table,
)

logger = get_logger("mlkit.cli")


def cmd_run(
    config_path: str,
    data_path: str | None = None,
    output_dir: str = "./bundle",
    html_report: str | None = None,
) -> int:
    """Execute full pipeline training from a config file."""
    cfg = TaskConfig.from_yaml(config_path)
    resolved = data_path or cfg.data_path
    if not resolved or not Path(resolved).exists():
        logger.error("Data file not found: %s", resolved)
        return 1

    with run_context() as run_id:
        logger.info("Run %s: config=%s data=%s", run_id, config_path, resolved)
        try:
            result = train(
                cfg, read_table(resolved), output_dir=output_dir, html_report=html_report
            )
        except MLKitError as exc:
            logger.error("%s: %s", type(exc).__name__, exc)
            return 1

        arena_rows = result.metrics.get("arena")
        if arena_rows:
            import pandas as pd

            print(pd.DataFrame(arena_rows).to_string(index=False))
        print(
            json.dumps(
                {"champion": result.metrics.get("champion"), "holdout": result.metrics["holdout"]},
                indent=2,
            )
        )
        print(result.drift_report.summary())

        if cfg.tracking.get("mlflow"):
            from mlkit.tracking import log_to_mlflow

            log_to_mlflow(result, run_name=run_id)

        if not result.metrics.get("thresholds_passed", True):
            logger.error(
                "Release gate failed: holdout metrics did not meet metric_thresholds (%s). "
                "The bundle was still written; metrics.json has thresholds_passed=false.",
                "; ".join(result.metrics.get("threshold_failures", [])),
            )
            return 1
    return 0


def cmd_predict(
    bundle_path: str, data_path: str, out_path: str | None = None, threshold: float | None = None
) -> int:
    """Score a batch file with a saved bundle."""
    try:
        preds = predict_bundle(bundle_path, read_table(data_path), threshold=threshold)
    except (MLKitError, FileNotFoundError, ValueError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    if out_path:
        write_table(preds, out_path)
        logger.info("Wrote %d predictions to %s", len(preds), out_path)
    else:
        preds.to_csv(sys.stdout, index=False)
    return 0


def cmd_monitor(
    bundle_path: str,
    data_path: str,
    out_path: str | None = None,
    fail_on: str = "error",
) -> int:
    """Evaluate drift of a new batch against the baseline stored in the bundle."""
    try:
        report = monitor_bundle(bundle_path, read_table(data_path))
    except (MLKitError, FileNotFoundError, ValueError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    print(report.summary())
    if out_path:
        if out_path.lower().endswith(".html"):
            report.to_html(out_path)
        else:
            Path(out_path).write_text(report.to_json(), encoding="utf-8")
        logger.info("Drift report written to %s", out_path)
    drifted = report.has_errors or (fail_on == "warn" and report.has_warnings)
    return 2 if drifted else 0


def cmd_evaluate(
    bundle_path: str,
    data_path: str,
    by: str | None = None,
    out_path: str | None = None,
    threshold: float | None = None,
) -> int:
    """Measure model performance on labelled data, overall and per group (backtest)."""
    try:
        res = evaluate_bundle(bundle_path, read_table(data_path), by=by, threshold=threshold)
    except (MLKitError, FileNotFoundError, ValueError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    import pandas as pd

    keys = [
        "n",
        "positive_rate",
        "mean_score",
        "roc_auc",
        "brier",
        "ece",
        "accuracy",
        "rmse",
        "mae",
        "r2",
    ]
    overall = {k: res.overall[k] for k in keys if k in res.overall}
    ref = {k: res.reference[k] for k in keys if k in res.reference}
    print(
        pd.DataFrame([overall, ref], index=["evaluation", "training holdout"]).round(4).to_string()
    )
    if res.by_group is not None:
        cols = [by] + [k for k in keys if k in res.by_group.columns]
        print(res.by_group[cols].round(4).to_string(index=False))
    for f in res.failures:
        logger.warning("Threshold not met: %s", f)
    if out_path:
        if out_path.lower().endswith(".json"):
            Path(out_path).write_text(
                json.dumps(res.to_dict(), indent=2, default=str), encoding="utf-8"
            )
        elif res.by_group is not None:
            write_table(res.by_group, out_path)
        else:
            write_table(pd.DataFrame([res.overall]), out_path)
        logger.info("Evaluation written to %s", out_path)
    return 0 if res.passed else 2


def cmd_serve(bundle_path: str, host: str = "127.0.0.1", port: int = 8000) -> int:
    try:
        import uvicorn
    except ImportError:
        logger.error("Serving requires extras: pip install 'mlkit[serve]'")
        return 1
    from mlkit.serving import create_app

    uvicorn.run(create_app(bundle_path), host=host, port=port)
    return 0


def cmd_eda(
    data_path: str,
    target: str | None = None,
    task_type: str | None = None,
    out_path: str | None = None,
    profile_json: str | None = None,
    sample_rows: int | None = 200_000,
) -> int:
    """Profile + QC + EDA of a data file; writes a standalone HTML report."""
    from mlkit.qc_eda import run_eda

    try:
        df = read_table(data_path)
        report = run_eda(df, target=target, task_type=task_type, sample_rows=sample_rows)
    except (MLKitError, FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    print(report.summary())
    if out_path:
        report.to_html(out_path, title=f"EDA - {Path(data_path).name}")
        logger.info("EDA report written to %s", out_path)
    if profile_json:
        payload = {
            "profile": report.profile.to_dict(),
            "target_association": report.target_association.to_dict("records"),
        }
        if report.qc is not None:
            payload["qc"] = report.qc.report_.to_dict()
            payload["issues"] = report.qc.issues_
        Path(profile_json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("Profile written to %s", profile_json)
    return 0


def cmd_benchmark(
    cases: list[str] | None = None, with_eda: bool = True, out_path: str | None = None
) -> int:
    """Run the multi-dataset generality suite; exit 1 if any expectation fails."""
    import pandas as pd

    from mlkit.benchmarks import run_suite

    try:
        res = run_suite(cases, with_eda=with_eda)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    with pd.option_context("display.width", 200, "display.max_colwidth", 100):
        print(res[["case", "passed", "seconds", "detail"]].to_string(index=False))
    if out_path:
        write_table(res, out_path)
    failed = res.loc[~res["passed"], "case"].tolist()
    print(f"{len(res) - len(failed)}/{len(res)} passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


def cmd_steps() -> int:
    from mlkit.core.registry import list_steps

    for name in list_steps():
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlkit", description="MLKit CLI: 6-stage tabular ML pipeline runner"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only warnings and errors")
    parser.add_argument(
        "--log-json", action="store_true", help="Emit structured JSON logs (ADR-002)"
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Train the pipeline from a config file")
    run_p.add_argument("config", help="Path to config.yaml")
    run_p.add_argument("--data", help="Override path to dataset (csv/parquet/xlsx/json)")
    run_p.add_argument("--out", default="./bundle", help="Output directory for the bundle")
    run_p.add_argument("--html-report", help="Write an HTML diagnostic report to this path")

    pred_p = sub.add_parser("predict", help="Score a batch with a saved bundle")
    pred_p.add_argument("bundle")
    pred_p.add_argument("data")
    pred_p.add_argument("--out", help="Output file (csv/parquet/json). Defaults to stdout.")
    pred_p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (classification). Default: the threshold stored in the bundle",
    )

    mon_p = sub.add_parser("monitor", help="Check a batch for drift against a saved bundle")
    mon_p.add_argument("bundle")
    mon_p.add_argument("data")
    mon_p.add_argument("--out", help="Write report (.json or .html)")
    mon_p.add_argument(
        "--fail-on", choices=["warn", "error"], default="error", help="Exit 2 at this severity"
    )

    srv_p = sub.add_parser("serve", help="Serve a bundle over HTTP (requires mlkit[serve])")
    srv_p.add_argument("bundle")
    srv_p.add_argument("--host", default="127.0.0.1")
    srv_p.add_argument("--port", type=int, default=8000)

    ev_p = sub.add_parser("evaluate", help="Measure performance on labelled data (backtest)")
    ev_p.add_argument("bundle")
    ev_p.add_argument("data", help="Labelled data containing the target column")
    ev_p.add_argument("--by", help="Report metrics per value of this column (e.g. cutoff, week)")
    ev_p.add_argument(
        "--out", help="Write per-group metrics (.csv/.parquet) or full result (.json)"
    )
    ev_p.add_argument(
        "--threshold", type=float, default=None, help="Override the bundle decision threshold"
    )

    eda_p = sub.add_parser(
        "eda", help="Profile, quality-check and explore a data file (HTML report)"
    )
    eda_p.add_argument("data")
    eda_p.add_argument("--target", help="Target column (optional)")
    eda_p.add_argument(
        "--task-type", choices=["classification", "regression"], help="Override target type"
    )
    eda_p.add_argument("--out", help="HTML report path")
    eda_p.add_argument(
        "--profile-json", help="Write the profile, QC findings and associations as JSON"
    )
    eda_p.add_argument(
        "--sample-rows", type=int, default=200_000, help="Analyse at most this many rows (0 = all)"
    )

    bm_p = sub.add_parser("benchmark", help="Run the multi-dataset generality suite")
    bm_p.add_argument("--case", action="append", help="Run only this case (repeatable)")
    bm_p.add_argument("--no-eda", action="store_true", help="Skip the EDA/report part of each case")
    bm_p.add_argument("--out", help="Write results (.csv/.parquet/.json)")

    sub.add_parser("steps", help="List registered pipeline steps")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for mlkit CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    configure_logging(level=level, json_format=args.log_json)

    if args.command == "run":
        code = cmd_run(
            args.config, data_path=args.data, output_dir=args.out, html_report=args.html_report
        )
    elif args.command == "predict":
        code = cmd_predict(args.bundle, args.data, out_path=args.out, threshold=args.threshold)
    elif args.command == "monitor":
        code = cmd_monitor(args.bundle, args.data, out_path=args.out, fail_on=args.fail_on)
    elif args.command == "evaluate":
        code = cmd_evaluate(
            args.bundle, args.data, by=args.by, out_path=args.out, threshold=args.threshold
        )
    elif args.command == "serve":
        code = cmd_serve(args.bundle, host=args.host, port=args.port)
    elif args.command == "eda":
        code = cmd_eda(
            args.data,
            target=args.target,
            task_type=args.task_type,
            out_path=args.out,
            profile_json=args.profile_json,
            sample_rows=args.sample_rows or None,
        )
    elif args.command == "benchmark":
        code = cmd_benchmark(args.case, with_eda=not args.no_eda, out_path=args.out)
    elif args.command == "steps":
        code = cmd_steps()
    else:
        parser.print_help()
        code = 0
    sys.exit(code)


if __name__ == "__main__":
    main()
