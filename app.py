import math
import os
import time
from typing import Any

import wandb
from flask import Flask, jsonify, render_template, request, session
from wandb.errors import CommError

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "replace-this-for-shared-use")
app.config["JSON_SORT_KEYS"] = False

MAX_RUNS_SCAN = max(100, int(os.getenv("MAX_RUNS_SCAN", "5000")))
MAX_METRICS = int(os.getenv("MAX_METRICS", "20"))
MAX_METRIC_KEYS = int(os.getenv("MAX_METRIC_KEYS", "200"))
MAX_METRICS_PER_REQUEST = int(os.getenv("MAX_METRICS_PER_REQUEST", "12"))
MAX_HISTORY_ROWS = int(os.getenv("MAX_HISTORY_ROWS", "8000"))
MAX_POINTS_PER_METRIC = int(os.getenv("MAX_POINTS_PER_METRIC", "350"))
MAX_METRIC_DISCOVERY_ROWS = int(os.getenv("MAX_METRIC_DISCOVERY_ROWS", "800"))
DEFAULT_PREVIEW_TIME_BUDGET_S = float(os.getenv("DEFAULT_PREVIEW_TIME_BUDGET_S", "8"))
DEFAULT_FULL_TIME_BUDGET_S = float(os.getenv("DEFAULT_FULL_TIME_BUDGET_S", "22"))
DEFAULT_KEYS_TIME_BUDGET_S = float(os.getenv("DEFAULT_KEYS_TIME_BUDGET_S", "8"))

DEFAULT_BASE_URL = os.getenv("WANDB_BASE_URL", "https://api.wandb.ai")
PREFERRED_METRICS = [
    "train/loss",
    "val/loss",
    "loss",
    "train/accuracy",
    "val/accuracy",
    "accuracy",
    "f1",
    "precision",
    "recall",
]


def is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def downsample_points(points: list[dict[str, float]], max_points: int) -> list[dict[str, float]]:
    if len(points) <= max_points:
        return points
    stride = math.ceil(len(points) / max_points)
    return [points[i] for i in range(0, len(points), stride)][:max_points]


def clamp_int(raw: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(value, maximum))


def clamp_float(raw: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(value, maximum))


def current_config() -> dict[str, str]:
    return {
        "api_key": (session.get("api_key") or "").strip(),
        "entity": (session.get("entity") or "").strip(),
        "project": (session.get("project") or "").strip(),
        "base_url": (session.get("base_url") or DEFAULT_BASE_URL).strip(),
    }


def build_api(config: dict[str, str]) -> wandb.Api:
    kwargs: dict[str, Any] = {"overrides": {"base_url": config["base_url"]}}
    if config["api_key"]:
        kwargs["api_key"] = config["api_key"]
    return wandb.Api(**kwargs)


def serialize_run(run: Any) -> dict[str, Any]:
    summary = run.summary or {}
    numeric_metric_count = sum(
        1 for key, value in summary.items() if not str(key).startswith("_") and is_number(value)
    )
    return {
        "id": run.id,
        "name": run.name,
        "group": run.group,
        "state": run.state,
        "created_at": run.created_at,
        "url": run.url,
        "numeric_metric_count": numeric_metric_count,
    }


def pick_metric_keys(run: Any, max_metrics: int, deadline: float | None = None) -> tuple[list[str], bool]:
    summary = run.summary or {}
    candidates = [
        key
        for key, value in summary.items()
        if not str(key).startswith("_") and is_number(value)
    ]

    ordered: list[str] = []
    for key in PREFERRED_METRICS:
        if key in candidates and key not in ordered:
            ordered.append(key)
    for key in sorted(candidates):
        if key not in ordered:
            ordered.append(key)
    if ordered:
        return ordered[:max_metrics], False

    discovered: list[str] = []
    row_count = 0
    timed_out = False
    for row in run.scan_history(page_size=200):
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            break
        row_count += 1
        if row_count > MAX_METRIC_DISCOVERY_ROWS:
            break
        for key, value in row.items():
            if str(key).startswith("_"):
                continue
            if is_number(value) and key not in discovered:
                discovered.append(key)
                if len(discovered) >= max_metrics:
                    return discovered, timed_out
    return discovered, timed_out


def parse_requested_metric_keys(max_keys: int) -> tuple[list[str], bool]:
    requested: list[str] = []
    for raw in request.args.getlist("key"):
        for token in str(raw).split(","):
            metric = token.strip()
            if metric and metric not in requested:
                requested.append(metric)

    raw_keys = request.args.get("keys")
    if raw_keys:
        for token in str(raw_keys).split(","):
            metric = token.strip()
            if metric and metric not in requested:
                requested.append(metric)

    if len(requested) <= max_keys:
        return requested, False
    return requested[:max_keys], True


def collect_single_metric_series(
    run: Any,
    metric_key: str,
    max_history_rows: int,
    max_points_per_metric: int,
    deadline: float | None = None,
) -> tuple[list[dict[str, float]], int, bool, bool]:
    points: list[dict[str, float]] = []
    row_count = 0
    timed_out = False
    row_limit_hit = False

    sample_count = min(max(max_points_per_metric * 8, 600), 6000)
    iterator: Any = None
    try:
        iterator = run.history(keys=["_step", metric_key], pandas=False, samples=sample_count)
    except Exception:
        iterator = None

    if iterator is None:
        iterator = run.scan_history(keys=["_step", metric_key], page_size=1000)

    for row in iterator:
        if deadline is not None and row_count % 64 == 0 and time.monotonic() >= deadline:
            timed_out = True
            break
        row_count += 1
        if row_count > max_history_rows:
            row_limit_hit = True
            break

        step = row.get("_step")
        value = row.get(metric_key)
        if is_number(step) and is_number(value):
            points.append({"x": float(step), "y": float(value)})

    return downsample_points(points, max_points_per_metric), row_count, timed_out, row_limit_hit


def collect_metric_series_by_key(
    run: Any,
    metric_keys: list[str],
    max_history_rows: int,
    max_points_per_metric: int,
    deadline: float | None = None,
) -> tuple[dict[str, list[dict[str, float]]], int, bool, bool]:
    series: dict[str, list[dict[str, float]]] = {key: [] for key in metric_keys}
    if not metric_keys:
        return series, 0, False, False

    total_rows_scanned = 0
    timed_out = False
    row_limit_hit = False

    for metric_key in metric_keys:
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            break

        (
            points,
            rows_scanned,
            metric_timed_out,
            metric_row_limit_hit,
        ) = collect_single_metric_series(
            run,
            metric_key,
            max_history_rows=max_history_rows,
            max_points_per_metric=max_points_per_metric,
            deadline=deadline,
        )
        series[metric_key] = points
        total_rows_scanned += rows_scanned
        timed_out = timed_out or metric_timed_out
        row_limit_hit = row_limit_hit or metric_row_limit_hit

    return series, total_rows_scanned, timed_out, row_limit_hit


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/config")
def get_config() -> Any:
    config = current_config()
    return jsonify(
        {
            "entity": config["entity"],
            "project": config["project"],
            "base_url": config["base_url"],
            "has_api_key": bool(config["api_key"]),
        }
    )


@app.post("/api/config")
def save_config() -> Any:
    payload = request.get_json(silent=True) or {}
    api_key = (payload.get("api_key") or "").strip()
    if api_key:
        session["api_key"] = api_key
    session["entity"] = (payload.get("entity") or "").strip()
    session["project"] = (payload.get("project") or "").strip()
    session["base_url"] = (payload.get("base_url") or DEFAULT_BASE_URL).strip()

    config = current_config()
    return jsonify(
        {
            "ok": True,
            "entity": config["entity"],
            "project": config["project"],
            "base_url": config["base_url"],
            "has_api_key": bool(config["api_key"]),
        }
    )


@app.get("/api/runs")
def list_runs() -> Any:
    config = current_config()
    if not config["entity"] or not config["project"]:
        return jsonify({"error": "Missing entity/project. Set config first."}), 400

    try:
        api = build_api(config)
        project_path = f"{config['entity']}/{config['project']}"
        limit = clamp_int(request.args.get("limit"), MAX_RUNS_SCAN, 100, MAX_RUNS_SCAN)
        runs_iter = api.runs(project_path, order="-created_at", per_page=100)

        runs: list[dict[str, Any]] = []
        for idx, run in enumerate(runs_iter):
            if idx >= limit:
                break
            runs.append(serialize_run(run))
        return jsonify(
            {
                "project_path": project_path,
                "count": len(runs),
                "runs": runs,
                "meta": {
                    "limit": limit,
                    "max_runs_scan": MAX_RUNS_SCAN,
                },
            }
        )
    except CommError as exc:
        return jsonify({"error": str(exc)}), 502


@app.get("/api/runs/<run_id>/metric_keys")
def run_metric_keys(run_id: str) -> Any:
    config = current_config()
    if not config["entity"] or not config["project"]:
        return jsonify({"error": "Missing entity/project. Set config first."}), 400

    run_path = f"{config['entity']}/{config['project']}/{run_id}"
    try:
        started_at = time.monotonic()
        api = build_api(config)
        run = api.run(run_path)

        limit = clamp_int(request.args.get("limit"), MAX_METRIC_KEYS, 1, MAX_METRIC_KEYS)
        time_budget_s = clamp_float(
            request.args.get("time_budget_s"),
            DEFAULT_KEYS_TIME_BUDGET_S,
            2.0,
            60.0,
        )
        deadline = time.monotonic() + time_budget_s

        metrics, timed_out = pick_metric_keys(run, limit, deadline=deadline)
        elapsed_ms = int((time.monotonic() - started_at) * 1000)

        return jsonify(
            {
                "run": serialize_run(run),
                "run_path": run_path,
                "metrics": metrics,
                "count": len(metrics),
                "partial": timed_out,
                "timed_out": timed_out,
                "meta": {
                    "elapsed_ms": elapsed_ms,
                    "time_budget_s": time_budget_s,
                },
                "limits": {
                    "limit": limit,
                    "max_metric_keys": MAX_METRIC_KEYS,
                },
            }
        )
    except CommError as exc:
        return jsonify({"error": str(exc)}), 502


@app.get("/api/runs/<run_id>/metrics")
def run_metrics(run_id: str) -> Any:
    config = current_config()
    if not config["entity"] or not config["project"]:
        return jsonify({"error": "Missing entity/project. Set config first."}), 400

    run_path = f"{config['entity']}/{config['project']}/{run_id}"
    try:
        started_at = time.monotonic()
        api = build_api(config)
        run = api.run(run_path)
        metrics_limit = clamp_int(request.args.get("metrics_limit"), MAX_METRICS, 1, MAX_METRICS)
        history_rows = clamp_int(request.args.get("history_rows"), MAX_HISTORY_ROWS, 500, MAX_HISTORY_ROWS)
        points_limit = clamp_int(
            request.args.get("points_limit"), MAX_POINTS_PER_METRIC, 80, MAX_POINTS_PER_METRIC
        )
        requested_metric_keys, metric_keys_truncated = parse_requested_metric_keys(MAX_METRICS_PER_REQUEST)
        is_preview_request = (
            metrics_limit < MAX_METRICS
            or history_rows < MAX_HISTORY_ROWS
            or points_limit < MAX_POINTS_PER_METRIC
            or bool(requested_metric_keys)
        )
        default_budget = DEFAULT_PREVIEW_TIME_BUDGET_S if is_preview_request else DEFAULT_FULL_TIME_BUDGET_S
        time_budget_s = clamp_float(request.args.get("time_budget_s"), default_budget, 2.0, 60.0)
        deadline = time.monotonic() + time_budget_s

        if requested_metric_keys:
            metric_keys = requested_metric_keys
            discovery_timed_out = False
            series, rows_scanned, scan_timed_out, row_limit_hit = collect_metric_series_by_key(
                run,
                metric_keys,
                history_rows,
                points_limit,
                deadline=deadline,
            )
        else:
            metric_keys, discovery_timed_out = pick_metric_keys(run, metrics_limit, deadline=deadline)
            series, rows_scanned, scan_timed_out, row_limit_hit = collect_metric_series_by_key(
                run,
                metric_keys,
                history_rows,
                points_limit,
                deadline=deadline,
            )

        charts = [
            {"metric": key, "points": points}
            for key, points in series.items()
            if points
        ]
        partial = (
            metrics_limit < MAX_METRICS
            or history_rows < MAX_HISTORY_ROWS
            or points_limit < MAX_POINTS_PER_METRIC
            or metric_keys_truncated
            or discovery_timed_out
            or scan_timed_out
            or row_limit_hit
        )
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        return jsonify(
            {
                "run": serialize_run(run),
                "run_path": run_path,
                "charts": charts,
                "partial": partial,
                "timed_out": discovery_timed_out or scan_timed_out,
                "meta": {
                    "elapsed_ms": elapsed_ms,
                    "rows_scanned": rows_scanned,
                    "row_limit_hit": row_limit_hit,
                    "discovery_timed_out": discovery_timed_out,
                    "scan_timed_out": scan_timed_out,
                    "time_budget_s": time_budget_s,
                    "metric_keys_truncated": metric_keys_truncated,
                },
                "limits": {
                    "metrics_limit": metrics_limit,
                    "history_rows": history_rows,
                    "points_limit": points_limit,
                    "requested_metrics": len(requested_metric_keys),
                    "max_metrics_per_request": MAX_METRICS_PER_REQUEST,
                },
            }
        )
    except CommError as exc:
        return jsonify({"error": str(exc)}), 502


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    app.run(host="0.0.0.0", port=port, debug=debug)
