#!/usr/bin/env bash
set -Eeuo pipefail

image="${1:?Pass the image tag to smoke test}"
artifacts="${2:-docker-smoke-results}"
container="fp-smoke-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
volume="${container}-data"
container_owned=false
volume_owned=false
mkdir -p "$artifacts"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if "$container_owned" && docker container inspect "$container" >/dev/null 2>&1; then
    docker logs "$container" >"$artifacts/final.log" 2>&1 || status=1
    docker stop --time 20 "$container" >/dev/null || status=1
    docker rm "$container" >/dev/null || status=1
  fi
  if "$volume_owned"; then
    docker volume rm "$volume" >/dev/null || status=1
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if docker container inspect "$container" >/dev/null 2>&1 \
    || docker volume inspect "$volume" >/dev/null 2>&1; then
  echo "Refusing to reuse existing smoke resources" >&2
  exit 1
fi
docker volume create "$volume" >/dev/null
volume_owned=true

start_container() {
  container_owned=true
  docker run -d --name "$container" --cpus=2 --memory=2g \
    -e AUTH_ENABLED=false -e API_DEV_MODE=true \
    -p 127.0.0.1::8000 \
    --mount "type=volume,source=$volume,target=/app/forecasting-product/data" \
    "$image" >/dev/null
}

wait_ready() {
  label="$1"
  selected_run="${2:-}"
  deadline=$((SECONDS + 240))
  port="$(docker port "$container" 8000/tcp)"
  base="http://${port}"
  while (( SECONDS < deadline )); do
    if [[ "$(docker inspect --format '{{.State.Running}}' "$container")" != "true" ]]; then
      echo "Container exited before $label readiness" >&2
      return 1
    fi
    if docker exec "$container" test -f /app/forecasting-product/data/.seeded \
        && curl --fail --silent --show-error --max-time 5 "$base/health" >"$artifacts/$label-health.json"; then
      forecast_url="$base/forecast/walmart_m5_weekly"
      if [[ -n "$selected_run" ]]; then
        forecast_url="$forecast_url?run_id=$selected_run"
        curl --fail --silent --show-error --max-time 10 \
          "$base/pipeline/runs/walmart_m5_weekly/$selected_run/config" >"$artifacts/$label-config.json"
      fi
      curl --fail --silent --show-error --max-time 10 "$forecast_url" >"$artifacts/$label-forecast.json"
      curl --fail --silent --show-error --max-time 10 "$base/metrics/leaderboard/walmart_m5_weekly" >"$artifacts/$label-leaderboard.json"
      curl --fail --silent --show-error --max-time 10 "$base/pipeline/runs?lob=walmart_m5_weekly" >"$artifacts/$label-runs.json"
      python3 - "$artifacts" "$label" "$selected_run" <<'PY'
import csv
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

root, label = Path(sys.argv[1]), sys.argv[2]
selected_run = sys.argv[3] if len(sys.argv) > 3 else ""
health = json.loads((root / f"{label}-health.json").read_text())
assert health["status"] == "ok" and all(health["checks"].values()), health
forecast = json.loads((root / f"{label}-forecast.json").read_text())
assert forecast["series_count"] == 10 and len(forecast["points"]) == 120, forecast
assert all(math.isfinite(point["forecast"]) for point in forecast["points"])
assert len({point["series_id"] for point in forecast["points"]}) == 10
assert len({(point["series_id"], point["week"]) for point in forecast["points"]}) == 120
for point in forecast["points"]:
    quantiles = [point[f"forecast_p{level}"] for level in (10, 50, 90)]
    assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in quantiles), point
    assert quantiles[0] <= quantiles[1] <= quantiles[2], point
    if point["model"] == "naive_seasonal":
        assert quantiles[1] == point["forecast"], point
leaderboard = json.loads((root / f"{label}-leaderboard.json").read_text())
assert leaderboard["entries"], leaderboard
assert forecast["artifact_source"] == leaderboard["artifact_source"] == "committed"
runs = json.loads((root / f"{label}-runs.json").read_text())["runs"]
assert len(runs) == (3 if selected_run else 2)
assert {run["run_type"] for run in runs} == {"forecast", "backtest"}
if selected_run:
    assert forecast["run_id"] == selected_run
    assert sum(run["run_type"] == "forecast" for run in runs) == 2
    assert any(run["run_type"] == "forecast" and run["run_id"] == selected_run for run in runs)
    assert all(point["model"] == "naive_seasonal" for point in forecast["points"])
    configuration = json.loads((root / f"{label}-config.json").read_text())
    assert configuration["run_id"] == selected_run
    assert configuration["config_hash"] == forecast["config_hash"]
    settings = configuration["configuration"]["settings"]["forecast"]
    assert settings["frequency"] == "W" and settings["horizon_weeks"] == 12
    panel = {}
    with (root / "explicit-naive-input.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            panel.setdefault(row["series_id"], []).append(
                (date.fromisoformat(row["date"]), float(row["quantity"]))
            )
    assert len(panel) == 10
    expected = {}
    for series_id, history in panel.items():
        history.sort()
        assert len(history) >= 52
        for lead in range(12):
            expected[(series_id, (history[-1][0] + timedelta(weeks=lead + 1)).isoformat())] = (
                history[-52 + lead][1]
            )
    actual = {(point["series_id"], point["week"]): point["forecast"] for point in forecast["points"]}
    assert actual == expected, "Explicit Naive points changed from the weekly seasonal oracle"
else:
    assert forecast["run_id"] == next(run["run_id"] for run in runs if run["run_type"] == "forecast")
assert leaderboard["run_id"] == next(run["run_id"] for run in runs if run["run_type"] == "backtest")
print(f"{label}: healthy, 10 series / 120 finite ordered forecasts, persisted leaderboard retrieved")
PY
      return 0
    fi
    sleep 2
  done
  echo "Seeded $label readiness exceeded 240 seconds" >&2
  return 1
}

snapshot() {
  docker exec -i "$container" python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("/app/forecasting-product/data")
files = [root / ".seeded"]
for directory in ("demo", "forecasts", "metrics", "runs"):
    files.extend(path for path in (root / directory).rglob("*") if path.is_file())
assert any(path.name == "forecast.parquet" and "runs" in path.parts for path in files)
assert any(path.name == "metrics.parquet" and "runs" in path.parts for path in files)
print(json.dumps({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(files)}, indent=2, sort_keys=True))
PY
}

start_container
wait_ready cold
snapshot >"$artifacts/cold-artifacts.json"
docker logs "$container" >"$artifacts/cold.log" 2>&1
docker stop --time 20 "$container" >/dev/null
docker rm "$container" >/dev/null
container_owned=false

# A fresh container must retrieve exactly the same data from the same volume.
start_container
wait_ready restart
snapshot >"$artifacts/restart-artifacts.json"
docker logs "$container" >"$artifacts/restart.log" 2>&1
grep -Fq "Sentinel found" "$artifacts/restart.log"
diff -u "$artifacts/cold-artifacts.json" "$artifacts/restart-artifacts.json"
diff -u "$artifacts/cold-runs.json" "$artifacts/restart-runs.json"
echo "Cold start and fresh-container restart preserved identical seeded artifacts"

# Keep the selected default champion above; this is a separate, explicit attempt.
docker cp "$container:/app/forecasting-product/data/demo/m5_weekly.csv" "$artifacts/explicit-naive-input.csv"
docker cp "$container:/app/forecasting-product/configs/m5_weekly_config.yaml" "$artifacts/explicit-naive-config.yaml"
curl --fail --silent --show-error --max-time 180 \
  -F "file=@$artifacts/explicit-naive-input.csv;type=text/csv" \
  -F "config_file=@$artifacts/explicit-naive-config.yaml;type=application/x-yaml" \
  "$base/pipeline/forecast?lob=walmart_m5_weekly&model_id=naive_seasonal&horizon=12" \
  >"$artifacts/explicit-naive-created.json"
naive_run_id="$(python3 - "$artifacts" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
prior_artifacts = json.loads((root / "restart-artifacts.json").read_text())
assert hashlib.sha256((root / "explicit-naive-input.csv").read_bytes()).hexdigest() == (
    prior_artifacts["demo/m5_weekly.csv"]
), "Explicit Naive must use the unchanged seeded weekly input"
result = json.loads((root / "explicit-naive-created.json").read_text())
assert result["status"] == "completed" and result["forecast_rows"] == 120
assert result["model_id"] == "naive_seasonal" and result["horizon"] == 12
assert re.fullmatch(r"[0-9a-f]{32}", result["run_id"])
prior = json.loads((root / "restart-runs.json").read_text())["runs"]
assert result["run_id"] not in {run["run_id"] for run in prior}
print(result["run_id"])
PY
)"
prior_run_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$artifacts/restart-forecast.json")"
wait_ready naive "$naive_run_id"
snapshot >"$artifacts/naive-artifacts.json"
curl --fail --silent --show-error --max-time 10 \
  "$base/forecast/walmart_m5_weekly?run_id=$prior_run_id" >"$artifacts/naive-prior-forecast.json"
diff -u "$artifacts/restart-forecast.json" "$artifacts/naive-prior-forecast.json"
diff -u "$artifacts/restart-leaderboard.json" "$artifacts/naive-leaderboard.json"
python3 - "$artifacts" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
prior = json.loads((root / "restart-artifacts.json").read_text())
after = json.loads((root / "naive-artifacts.json").read_text())
assert prior.keys() < after.keys(), "The explicit attempt must add an independent bundle"
assert all(after[key] == value for key, value in prior.items()), "A prior artifact changed"
PY
docker logs "$container" >"$artifacts/naive.log" 2>&1
docker stop --time 20 "$container" >/dev/null
docker rm "$container" >/dev/null
container_owned=false

start_container
wait_ready naive-restart "$naive_run_id"
snapshot >"$artifacts/naive-restart-artifacts.json"
curl --fail --silent --show-error --max-time 10 \
  "$base/forecast/walmart_m5_weekly?run_id=$prior_run_id" >"$artifacts/naive-restart-prior-forecast.json"
docker logs "$container" >"$artifacts/naive-restart.log" 2>&1
grep -Fq "Sentinel found" "$artifacts/naive-restart.log"
diff -u "$artifacts/naive-artifacts.json" "$artifacts/naive-restart-artifacts.json"
diff -u "$artifacts/naive-forecast.json" "$artifacts/naive-restart-forecast.json"
diff -u "$artifacts/naive-config.json" "$artifacts/naive-restart-config.json"
diff -u "$artifacts/naive-runs.json" "$artifacts/naive-restart-runs.json"
diff -u "$artifacts/restart-forecast.json" "$artifacts/naive-restart-prior-forecast.json"
diff -u "$artifacts/restart-leaderboard.json" "$artifacts/naive-restart-leaderboard.json"
echo "Explicit Naive: 120 ordered unchanged points/P50, exact restart, prior runs unchanged"
