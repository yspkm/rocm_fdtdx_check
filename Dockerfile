# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=rocm/jax:rocm7.2.4-jax0.8.2-py3.12
FROM ${BASE_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache

WORKDIR /app
COPY . /app
RUN command -v git >/dev/null \
    && python3 -m pip install --constraint requirements/rocm.txt --editable . \
    && python3 -m pip check

ENTRYPOINT ["python3", "-m", "fdtdx_check"]
