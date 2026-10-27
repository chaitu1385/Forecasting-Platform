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
      curl --fail --silent --show-error --max-time 10 "$base/forecast/walmart_m5_weekly" >"$artifacts/$label-forecast.json"
      curl --fail --silent --show-error --max-time 10 "$base/metrics/leaderboard/walmart_m5_weekly" >"$artifacts/$label-leaderboard.json"
      curl --fail --silent --show-error --max-time 10 "$base/pipeline/runs?lob=walmart_m5_weekly" >"$artifacts/$label-runs.json"
      python3 - "$artifacts" "$label" <<'PY'
import json
import math
import sys
from pathlib import Path

root, label = Path(sys.argv[1]), sys.argv[2]
health = json.loads((root / f"{label}-health.json").read_text())
assert health["status"] == "ok" and all(health["checks"].values()), health
forecast = json.loads((root / f"{label}-forecast.json").read_text())
assert forecast["series_count"] == 10 and len(forecast["points"]) == 120, forecast
assert all(math.isfinite(point["forecast"]) for point in forecast["points"])
leaderboard = json.loads((root / f"{label}-leaderboard.json").read_text())
assert leaderboard["entries"], leaderboard
assert forecast["artifact_source"] == leaderboard["artifact_source"] == "committed"
runs = json.loads((root / f"{label}-runs.json").read_text())["runs"]
assert len(runs) == 2 and {run["run_type"] for run in runs} == {"forecast", "backtest"}
assert forecast["run_id"] == next(run["run_id"] for run in runs if run["run_type"] == "forecast")
assert leaderboard["run_id"] == next(run["run_id"] for run in runs if run["run_type"] == "backtest")
print(f"{label}: healthy, 10 series / 120 forecasts, persisted leaderboard retrieved")
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
