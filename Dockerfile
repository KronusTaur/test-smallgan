FROM python:3.6
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
COPY run_experiments.sh /run_experiments.sh
RUN chmod +x /run_experiments.sh
ENTRYPOINT "/run_experiments.sh"