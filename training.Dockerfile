# ========================================
# Training Dockerfile
# ========================================
# Container temporaneo per addestrare il modello ML
# Genera models/unified_model.pkl senza installare nulla localmente

FROM python:3.11-slim

WORKDIR /app

# Installa dipendenze per il training
RUN pip install --no-cache-dir pandas scikit-learn numpy matplotlib seaborn joblib deep_translator tqdm

# Copia gli script di training
COPY src/prepare_data.py /app/
COPY src/train_unified_model.py /app/
COPY src/unified_model.py /app/

# Monta come volumi: data/ (input) e models/ (output)
VOLUME ["/app/data", "/app/models"]

# Comando di default: prepara dati e addestra modello
CMD python prepare_data.py && python train_unified_model.py
