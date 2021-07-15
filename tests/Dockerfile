FROM vad:latest

COPY requirements.txt .

RUN pip install -r requirements.txt 

WORKDIR /voice_activity_detection
COPY . .

CMD ["bash", "-c", "isort --check ./ && black --check ./ && flake8 ./"]
# CMD ["bash", "-c", "isort --check ./ && black --check ./ && flake8 ./ && pep257 ./"]
