# Backend Dockerfile per FastAPI
FROM python:3.10-slim

WORKDIR /app

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia codice applicazione
COPY . .

# Esponi porta 8000
EXPOSE 8000

# Comando di avvio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
