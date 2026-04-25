"""
========================================
Health Check Routes
========================================

Endpoint per verificare lo stato di salute dell'API:
- Verifica che l'API sia responsiva
- Controlla che il modello ML sia caricato correttamente
- Restituisce informazioni sulla versione e percorso modello

Utile per monitoring, health checks da orchestratori (Kubernetes, Docker).
"""

from fastapi import APIRouter
from app.models.unified_model import ModelManager
from app.core.config import settings

# ========================================
# ROUTER CONFIGURATION
# ========================================
router = APIRouter()


# ========================================
# HEALTH CHECK ENDPOINT
# ========================================
@router.get("/", status_code=200)
async def health_check() -> dict:
    """
    Health check endpoint.

    Verifica che l'API sia funzionante e il modello ML caricato.

    Returns:
        Dict con:
        - status: "healthy" o "unhealthy"
        - app: Nome dell'applicazione
        - version: Versione corrente
        - model_loaded: True se modello caricato
        - model_path: Percorso del file modello

    Note:
        Se il modello non è caricato, restituisce status "unhealthy"
        con dettagli dell'errore. Questo permette ai sistemi di
        monitoraggio di rilevare problemi di configurazione.
    """
    try:
        # Ottieni istanza singleton del ModelManager
        model_manager = ModelManager()

        # Tentativo di caricamento modello (lazy load)
        model = model_manager.get_model()

        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "model_loaded": model is not None,
            "model_path": settings.MODEL_PATH
        }
    except Exception as e:
        # In caso di errore, restituisci status unhealthy
        return {
            "status": "unhealthy",
            "error": str(e)
        }
