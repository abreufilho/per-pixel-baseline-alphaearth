FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/work/code \
    ALPHAEARTH_MG_DATA=/work/data \
    HOME=/tmp/home \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        fontconfig \
        fonts-dejavu-core \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /work/data /tmp/home/.config /tmp/.cache /tmp/matplotlib \
    && chmod -R 777 /work/data /tmp/home /tmp/.cache /tmp/matplotlib

WORKDIR /work

COPY code/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt

COPY . /work

CMD ["python", "code/check_artifacts.py"]
