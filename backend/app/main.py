"""
========================================
Main Application Entry Point
========================================

Entry Point dell'applicazione FastAPI.
Configura l'app, middleware, CORS e include i routers.

Responsabilità:
- Inizializza l'app FastAPI con configurazioni
- Configura CORS per chiamate dal frontend
- Pre-carica il modello ML allo startup
- Include tutti i router API
- Gestisce eccezioni globali
- Gestisce shutdown graceful

Versione: 2.0.0
Framework: FastAPI + scikit-learn + LIME
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.routes import ticket, metrics, health

# ----------------------------------------
# CONFIGURAZIONE LOGGING
# ----------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================
# LIFESPAN MANAGEMENT
# ========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestisce il lifecycle dell'applicazione.

    Startup:
    - Log informazioni applicazione
    - Pre-carica il modello ML (evita cold start prima richiesta)

    Shutdown:
    - Log di chiusura
    - Cleanup risorse (automatico Python)

    Args:
        app: Istanza FastAPI

    Yields:
        None (controllo all'applicazione)
    """
    # ----------------------------------------
    # STARTUP
    # ----------------------------------------
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} starting...")

    # Pre-carica il modello ML allo startup
    from app.models.unified_model import ModelManager
    model_manager = ModelManager()
    try:
        model = model_manager.get_model()
        logger.info("✅ ML Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load ML model: {e}")

    # ----------------------------------------
    # YIELD CONTROLLO ALL'APP
    # ----------------------------------------
    yield

    # ----------------------------------------
    # SHUTDOWN
    # ----------------------------------------
    logger.info("👋 Shutting down...")


# ========================================
# INIZIALIZZAZIONE FASTAPI
# ========================================
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API per classificazione intelligente di ticket di assistenza",
    lifespan=lifespan,
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc"
)


# ========================================
# CORS MIDDLEWARE
# ========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,      # Origini consentite
    allow_credentials=True,                       # Cookies e auth
    allow_methods=["*"],                          # Tutti i metodi HTTP
    allow_headers=["*"],                          # Tutti gli header
)


# ========================================
# INCLUDE ROUTERS
# ========================================
# Health Check Router
app.include_router(
    health.router,
    prefix=f"{settings.API_PREFIX}/health",
    tags=["Health"]
)

# Ticket Classification Router
app.include_router(
    ticket.router,
    prefix=f"{settings.API_PREFIX}/ticket",
    tags=["Ticket"]
)

# Metrics Router
app.include_router(
    metrics.router,
    prefix=f"{settings.API_PREFIX}/metrics",
    tags=["Metrics"]
)


# ========================================
# ROOT ENDPOINT
# ========================================
@app.get("/", tags=["Default"])
async def root():
    """
    Endpoint root con informazioni sull'API.

    **Returns: Dict con:**

        - app: nome applicazione
        - version: versione corrente
        - docs: link documentazione Swagger
        - status: "running"
    """
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": f"{settings.API_PREFIX}/docs",
        "status": "running"
    }


# ========================================
# GLOBAL EXCEPTION HANDLER
# ========================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Gestore globale delle eccezioni non catturate.

    Cattura tutte le eccezioni non gestite e restituisce
    una risposta JSON consistente invece di un errore 500 grezzo.

    Args:
        request: Richiesta FastAPI
        exc: Eccezione sollevata

    Returns:
        JSONResponse con:
        - status_code: 500
        - error: "Internal server error"
        - detail: Messaggio dettagliato (solo in debug mode)

    Note:
        In produzione, il dettaglio è oscurato per sicurezza.
        In debug mode, viene mostrato il messaggio completo.
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.LOG_LEVEL == "debug" else "An error occurred"
        }
    )
