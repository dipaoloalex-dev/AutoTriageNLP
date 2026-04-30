"""
========================================
Metrics Routes - Performance Modello
========================================

Endpoint API per recuperare le metriche di performance del modello:
- /summary: Metriche numeriche (accuracy, precision, recall, f1)
- /images: Path delle immagini dei grafici generati

Le metriche sono generate durante il training e salvate in file JSON.
Le immagini sono salvate come PNG nella cartella img/png/.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import os
import json
import logging

from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


# ========================================
# METRICS SUMMARY ENDPOINT
# ========================================
@router.get("/summary", response_model=Dict[str, Any])
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Restituisce il riepilogo delle metriche del modello.

    **Returns: Dict con metriche per Categoria e Priorità:**

        - accuracy: percentuale di classificazioni corrette
        - precision: affidabilità delle predizioni positive
        - recall: capacità di rilevare tutti i positivi
        - f1: media armonica di precision e recall

    Se il file **metrics_summary.json** non viene trovato, restituisce valori di default con flag **note**. Questo permette al frontend di funzionare anche in assenza del file **metrics_summary.json**.
    """
    try:
        # ----------------------------------------
        # RICERCA FILE METRICS
        # ----------------------------------------
        # Prima prova: nella stessa cartella del modello
        metrics_path = os.path.join(
            os.path.dirname(settings.MODEL_PATH),
            "metrics_summary.json"
        )

        if not os.path.exists(metrics_path):
            # Fallback: cerca nella root del progetto
            metrics_path = "../metrics_summary.json"
            if not os.path.exists(metrics_path):
                # Secondo fallback: restituisci valori di default
                return {
                    "category": {"accuracy": 0.85, "precision": 0.84, "recall": 0.83, "f1": 0.83},
                    "priority": {"accuracy": 0.75, "precision": 0.74, "recall": 0.73, "f1": 0.73},
                    "note": "Default metrics - file not found"
                }

        # ----------------------------------------
        # CARICAMENTO E RESTITUZIONE
        # ----------------------------------------
        with open(metrics_path) as f:
            metrics = json.load(f)

        return metrics

    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# METRICS IMAGES ENDPOINT
# ========================================
@router.get("/images", response_model=Dict[str, Any])
async def get_metrics_images() -> Dict[str, Any]:
    """
    Restituisce i path delle immagini delle metriche.

    **Returns: Dict con URLs per le varie immagini metriche:**

        - confusion_matrix_category/priority: matrici di confusione
        - accuracy_category/priority: grafici accuracy
        - precision_category/priority: grafici precision
        - recall_category/priority: grafici recall
        - f1score_category/priority: grafici F1-score

    Filtra automaticamente le immagini che non esistono per evitare errori 404 nel frontend. Il path **"../img"** viene tradotto in **"/assets"** per il routing.
    """
    try:
        # Base path per le immagini metriche
        base_path = "../img/png"

        # Mappa completa di tutte le immagini possibili
        images = {
            "confusion_matrix_category": f"{base_path}/confusion_matrix_category.png",
            "confusion_matrix_priority": f"{base_path}/confusion_matrix_priority.png",
            "accuracy_category": f"{base_path}/accuracy_category.png",
            "accuracy_priority": f"{base_path}/accuracy_priority.png",
            "precision_category": f"{base_path}/precision_category.png",
            "precision_priority": f"{base_path}/precision_priority.png",
            "recall_category": f"{base_path}/recall_category.png",
            "recall_priority": f"{base_path}/recall_priority.png",
            "f1score_category": f"{base_path}/f1score_category.png",
            "f1score_priority": f"{base_path}/f1score_priority.png"
        }

        # ----------------------------------------
        # FILTRA IMMAGINI ESISTENTI
        # ----------------------------------------
        # Rimuovi le entry per i file che non esistono
        existing_images = {
            k: v for k, v in images.items()
            if os.path.exists(v.replace("../img", "img"))
        }

        return existing_images

    except Exception as e:
        logger.error(f"Error getting metrics images: {e}")
        return {}
