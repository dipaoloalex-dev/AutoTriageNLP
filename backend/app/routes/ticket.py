"""
========================================
Ticket Routes - Classificazione API
========================================

Endpoint API per la classificazione dei ticket:
- /classify: Classificazione singola con spiegazioni
- /batch: Classificazione multipla con statistiche
- /upload-csv: Upload file CSV per batch processing

Ogni endpoint restituisce:
- Categoria assegnata
- Priorità assegnata (con logica ibrida)
- Confidenza della predizione
- Spiegazioni LIME (per singola)
- Statistiche aggregate (per batch)
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import logging

from app.models.unified_model import ModelManager
from app.services.ticket_service import TicketService
from app.services.lime_service import LimeService
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


# ========================================
# PYDANTIC REQUEST MODELS
# ========================================
class TicketRequest(BaseModel):
    """
    Request model per classificazione singola.

    Attributes:
        text: Testo del ticket da classificare
    """
    text: str = Field(..., min_length=1, max_length=settings.MAX_TEXT_LENGTH)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Il sistema è bloccato e non riusciamo a emettere fatture. Aiuto!"
            }
        }


class BatchClassifyRequest(BaseModel):
    """
    Request model per classificazione batch.

    Attributes:
        texts: Lista di testi da classificare
    """
    texts: List[str] = Field(..., min_length=1, max_length=settings.MAX_BATCH_SIZE)

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Il sistema è bloccato",
                    "Vorrei informazioni sulle nuove funzionalità"
                ]
            }
        }


# ========================================
# SINGLE CLASSIFICATION ENDPOINT
# ========================================
@router.post("/classify", response_model=Dict[str, Any])
async def classify_ticket(request: TicketRequest) -> Dict[str, Any]:
    """
    Classifica un singolo ticket.

    **Pipeline:**

        - Recupera il modello ML (singleton cached)
        - Applica TicketService per classificazione
        - Genera spiegazioni LIME
        - Unisce risultati e formatta per frontend

    **Returns: Dict con:**

        - category: categoria assegnata
        - priority: priorità assegnata
        - confidence: confidenza (0-1)
        - probabilities: dict probabilità per classe
        - lime_explanation: lista parole chiave LIME + triggers
        - text_preview: anteprima testo (max 200 char)

    **Raises:**

        - HTTPException 500: Errore durante classificazione
    """
    try:
        # ----------------------------------------
        # RECUPERA MODELLO
        # ----------------------------------------
        model_manager = ModelManager()
        model = model_manager.get_model()

        # ----------------------------------------
        # CLASSIFICA TICKET
        # ----------------------------------------
        result = TicketService.classify_text(model, request.text)

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error"))

        # ----------------------------------------
        # AGGIUNGI SPIEGAZIONE LIME
        # ----------------------------------------
        lime_results = LimeService.explain_prediction(model, request.text)
        lime_formatted = LimeService.format_for_frontend(
            lime_results,
            result.get("triggers", [])
        )

        result["lime_explanation"] = lime_formatted
        result["text_preview"] = (
            request.text[:200] + "..." if len(request.text) > 200 else request.text
        )

        return result

    except Exception as e:
        logger.error(f"Error in classify_ticket: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# BATCH CLASSIFICATION ENDPOINT
# ========================================
@router.post("/batch", response_model=Dict[str, Any])
async def classify_batch(request: BatchClassifyRequest) -> Dict[str, Any]:
    """
    Classifica una lista di ticket. Elabora ogni ticket singolarmente e calcola statistiche aggregate.

    **Returns: Dict con:**

        - results: lista di risultati per ogni ticket
        - summary: statistiche aggregate
            - total: numero totale ticket
            - high_priority: conteggio priorità Alta
            - medium_priority: conteggio priorità Media
            - low_priority: conteggio priorità Bassa
            - high_priority_percentage: % priorità Alta
        - success: `true` se operazione completata

    **Raises:**

        - HTTPException 500: Errore durante classificazione
    """
    try:
        model_manager = ModelManager()
        model = model_manager.get_model()

        # ----------------------------------------
        # CLASSIFICA TUTTI I TICKET
        # ----------------------------------------
        results = TicketService.classify_batch(model, request.texts)

        # ----------------------------------------
        # CALCOLA STATISTICHE
        # ----------------------------------------
        total = len(results)
        high_priority = sum(1 for r in results if r.get("priority") == "Alta")
        medium_priority = sum(1 for r in results if r.get("priority") == "Media")
        low_priority = sum(1 for r in results if r.get("priority") == "Bassa")

        summary = {
            "total": total,
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "low_priority": low_priority,
            "high_priority_percentage": round(high_priority / total * 100, 1) if total > 0 else 0
        }

        return {
            "results": results,
            "summary": summary,
            "success": True
        }

    except Exception as e:
        logger.error(f"Error in classify_batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# CSV UPLOAD ENDPOINT
# ========================================
@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Carica un file CSV e classifica tutti i ticket. Il CSV deve avere una colonna con nome: **text**, **body**, **testo** o **descrizione**. Le righe vengono classificate e restituite con le colonne aggiuntive.

    **Args:**

        - file: file CSV caricato via multipart/form-data

    **Returns: Dict con:**

        - results: lista risultati (max 100 per display)
        - summary: statistiche aggregate
        - total_rows: numero totale righe nel CSV

    **Raises:**

        - HTTPException 400: CSV vuoto, colonna mancante, troppe righe
        - HTTPException 500: Errore durante elaborazione

    I risultati sono limitati a 100 per evitare payload troppo grandi. Tutte le righe vengono comunque classificate.
    """
    try:
        # ----------------------------------------
        # LEGGI FILE CSV
        # ----------------------------------------
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # ----------------------------------------
        # TROVA COLONNA TESTO
        # ----------------------------------------
        text_col = None
        for col in ['text', 'body', 'testo', 'descrizione']:
            if col in df.columns:
                text_col = col
                break

        if text_col is None:
            raise HTTPException(
                status_code=400,
                detail="Colonna testo non trovata. Usa: text, body, testo o descrizione"
            )

        # ----------------------------------------
        # VALIDA DIMENSIONE
        # ----------------------------------------
        if len(df) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Too many rows. Max: {settings.MAX_BATCH_SIZE}"
            )

        # ----------------------------------------
        # ESTRAI TESTI E CLASSIFICA
        # ----------------------------------------
        texts = df[text_col].fillna('').tolist()

        model_manager = ModelManager()
        model = model_manager.get_model()
        results = TicketService.classify_batch(model, texts)

        # ----------------------------------------
        # AGGIUNGI COLONNE AL DATAFRAME
        # ----------------------------------------
        df['Categoria'] = [r.get('category') for r in results]
        df['Priorità'] = [r.get('priority') for r in results]
        df['Confidenza'] = [r.get('confidence') for r in results]

        # ----------------------------------------
        # CALCOLA STATISTICHE
        # ----------------------------------------
        high_priority = sum(1 for r in results if r.get("priority") == "Alta")

        summary = {
            "total_processed": len(results),
            "high_priority": high_priority,
            "success": True
        }

        # ----------------------------------------
        # PREPARA RISULTATI PER FRONTEND
        # ----------------------------------------
        display_results = []
        for i, row in df.iterrows():
            if i >= 100:  # Limita a 100 risultati
                break
            display_results.append({
                "description": row[text_col][:200] + "..." if len(row[text_col]) > 200 else row[text_col],
                "category": row['Categoria'],
                "priority": row['Priorità'],
                "confidence": row['Confidenza']
            })

        return {
            "results": display_results,
            "summary": summary,
            "total_rows": len(df)
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="File CSV vuoto")
    except Exception as e:
        logger.error(f"Error in upload_csv: {e}")
        raise HTTPException(status_code=500, detail=str(e))
