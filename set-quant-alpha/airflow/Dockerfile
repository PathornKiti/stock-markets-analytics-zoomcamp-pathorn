# Airflow image with set-quant-alpha installed in its OWN virtualenv (/opt/setalpha-venv),
# so the project's pandas / scikit-learn / xgboost versions never clash with Airflow's pins.
FROM apache/airflow:2.10.5-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/setalpha && chown -R airflow: /opt/setalpha
USER airflow

COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/airflow/setalpha-venv \
 && /opt/airflow/setalpha-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

ENV SETALPHA_PYTHON=/opt/airflow/setalpha-venv/bin/python \
    SETALPHA_HOME=/opt/setalpha \
    PYTHONPATH=/opt/setalpha/src
