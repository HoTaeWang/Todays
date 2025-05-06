# Rebecca Web-Bot Docker Image Example



1. Dockerfile

```
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Soucrce code 복사.
COPY . .

# App 실행 Command
CMD ["python", "app.py"]

```

