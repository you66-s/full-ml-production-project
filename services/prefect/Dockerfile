FROM prefecthq/prefect:3-latest

WORKDIR /opt/prefect/flows

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash", "-c", "prefect server start --host 0.0.0.0 --port 4200 & echo 'Waiting for Prefect server...'; until (echo > /dev/tcp/127.0.0.1/4200) 2>/dev/null; do sleep 1; done; echo 'Server is up, starting worker'; prefect worker start --pool default-agent-pool --work-queue default"]