# ========================================
# Comparison Dockerfile
# ========================================
# Container one-shot per generare i grafici di confronto
# tra modello su dati sintetici e dati reali
# Genera i grafici in img/png/ senza installare nulla localmente

FROM python:3.11-slim

WORKDIR /app

# Installa dipendenze per il confronto modelli
RUN pip install --no-cache-dir pandas scikit-learn numpy matplotlib seaborn tqdm

# Copia gli script necessari
COPY src/generate_synthetic_data.py /app/
COPY src/compare_models.py /app/
COPY src/unified_model.py /app/

# Monta come volumi: data/ (input/output), img/ (output grafici), models/ (per unified_model.py)
VOLUME ["/app/data", "/app/img", "/app/models"]

# Comando di default: genera dati sintetici e confronta i modelli
CMD python generate_synthetic_data.py && python compare_models.py
