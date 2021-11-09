FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    python3-setuptools \
    git \
    git-crypt \
    unzip \
    chromium-driver \
    gcc \
    make

RUN python -m pip install cassandra-driver

RUN apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

RUN chmod +x entrypoint.sh

RUN python3 -m venv /opt/venv && /opt/venv/bin/python -m pip install -r requirements.txt

RUN /opt/venv/bin/python -m pypyr  /app/pipelines/ai-model-download

RUN /opt/venv/bin/python -m pypyr  /app/pipelines/decrypt

CMD ["entrypoint.sh"]