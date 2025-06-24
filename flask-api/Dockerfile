FROM python:3.10-slim

WORKDIR /app

# upgrading system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# installing python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=600 -r requirements.txt

# code
COPY . .

EXPOSE 5000
CMD ["python", "app.py"]