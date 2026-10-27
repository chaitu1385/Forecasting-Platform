FROM python:3.10-slim AS base

WORKDIR /app

# System deps for LightGBM, XGBoost
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY forecasting-product/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY forecasting-product/ forecasting-product/

# ---------------------------------------------------------------------------
#  FastAPI target (with first-boot demo seeding)
# ---------------------------------------------------------------------------
FROM base AS api
EXPOSE 8000

# Pre-copy the M5 daily fixture so docker_seed.py can aggregate it on first boot
COPY forecasting-product/tests/integration/fixtures/m5_daily_sample.csv \
     forecasting-product/tests/integration/fixtures/m5_daily_sample.csv

ENV API_DATA_DIR=forecasting-product/data/ \
    DEMO_LOB=walmart_m5_weekly \
    POLARS_MAX_THREADS=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1

CMD ["python", "forecasting-product/scripts/docker_seed.py"]
