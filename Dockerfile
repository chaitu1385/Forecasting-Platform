FROM python:3.12.16-slim-bookworm@sha256:6dd8b6dab6b3adc7812bab69eca5dbdc1a5cc690ad28306d234ffb4ef8ac9753 AS base

WORKDIR /app

RUN python -c "import platform; assert platform.machine().lower() in ('x86_64', 'amd64'), 'The locked API image supports linux/amd64 only; native ARM is not qualified'"

# System deps for LightGBM, XGBoost
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY forecasting-product/requirements/ /tmp/requirements/
RUN python -m pip install --no-cache-dir --require-hashes --only-binary=:all: \
    -r /tmp/requirements/py312/api.txt

COPY forecasting-product/ forecasting-product/

# ---------------------------------------------------------------------------
#  FastAPI target (production by default; seeding requires explicit demo mode)
# ---------------------------------------------------------------------------
FROM base AS api
EXPOSE 8000

# Pre-copy the M5 daily fixture so docker_seed.py can aggregate it on first boot
COPY forecasting-product/tests/integration/fixtures/m5_daily_sample.csv \
     forecasting-product/tests/integration/fixtures/m5_daily_sample.csv

ENV API_DATA_DIR=forecasting-product/data/ \
    AUTH_ENABLED=true \
    API_DEV_MODE=false \
    DEMO_LOB=walmart_m5_weekly \
    POLARS_MAX_THREADS=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1

CMD ["python", "forecasting-product/scripts/docker_seed.py"]
