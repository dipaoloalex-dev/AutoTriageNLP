"""
========================================
Configurazione Centrale FastAPI
========================================

Gestisce tutte le impostazioni dell'applicazione:
- Variabili d'ambiente
- Configurazioni API
- Percorsi file
- Impostazioni CORS

Utilizza Pydantic Settings per validazione e type-safety.
"""

from pydantic_settings import BaseSettings
from typing import List
import os

# ========================================
# CLASSE SETTINGS
# ========================================
class Settings(BaseSettings):
    """
    Impostazioni dell'applicazione caricate da variabili d'ambiente.

    Tutte le impostazioni possono essere sovrascritte con:
    - File .env nella cartella backend/
    - Variabili d'ambiente del sistema
    """

    # ----------------------------------------
    # INFO APPLICAZIONE
    # ----------------------------------------
    APP_NAME: str = "AutoTriage NLP API"      # Nome dell'API
    APP_VERSION: str = "2.0.0"                   # Versione semantica
    API_PREFIX: str = "/api/v1"                   # Prefisso URL per tutti gli endpoint

    # ----------------------------------------
    # CONFIGURAZIONE MODELLO
    # ----------------------------------------
    # Percorso relativo al file .pkl del modello ML
    # Viene convertito in assoluto dopo il caricamento
    MODEL_PATH: str = "../models/unified_model.pkl"

    # ----------------------------------------
    # CONFIGURAZIONE CORS
    # ----------------------------------------
    # Origini autorizzate per le chiamate API dal frontend
    # Necessario per sicurezza in ambiente di produzione
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",      # Frontend locale (sviluppo)
        "http://127.0.0.1:3000",     # Frontend locale alternativo
        "http://localhost:3001",      # Frontend porta alternativa
    ]

    # ----------------------------------------
    # LIMITI API
    # ----------------------------------------
    MAX_BATCH_SIZE: int = 1000       # Massimo ticket per batch
    MAX_TEXT_LENGTH: int = 10000     # Massimo caratteri per testo

    # ----------------------------------------
    # LOGGING
    # ----------------------------------------
    LOG_LEVEL: str = "info"           # Livello log (debug, info, warning, error)

    class Config:
        env_file = ".env"             # File .env nella cartella backend/
        case_sensitive = True         # Rispetta maiuscolo delle variabili

# ========================================
# ISTANZA GLOBALE SETTINGS
# ========================================
settings = Settings()

# ========================================
# NORMALIZAZIONE PATH MODELLO
# ========================================
# Converte il path relativo del modello in assoluto
if not os.path.isabs(settings.MODEL_PATH):
    settings.MODEL_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", settings.MODEL_PATH)
    )
