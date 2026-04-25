"""
========================================
Ticket Service - Logica Classificazione
========================================

Servizio per la classificazione dei ticket.
Gestisce la logica ibrida ML + keyword per il calcolo della priorità.

La logica ibrida migliora la classificazione della priorità combinando:
1. Predizione del modello ML (Logistic Regression)
2. Keyword critiche che forzano priorità Alta
3. Keyword attenuanti che riducono la priorità
4. Keyword di feedback positivo che forzano priorità Bassa
"""

from typing import Tuple, List, Dict, Any
import logging

from app.models.unified_model import UnifiedModel

logger = logging.getLogger(__name__)


# ========================================
# TICKET SERVICE CLASS
# ========================================
class TicketService:
    """
    Servizio che incapsula la logica di classificazione dei ticket.

    Fornisce metodi statici per:
    - Classificazione singola con logica ibrida
    - Classificazione batch di più ticket
    - Calcolo priorità con keyword triggers
    """

    # ----------------------------------------
    # KEYWORD CONFIGURATIONS
    # ----------------------------------------
    # Parole che innescano priorità ALTA indipendentemente dal modello
    CRITICAL_KEYWORDS = [
        "virus", "hacker", "attacco", "violazione", "perso dati", "cancellato",
        "fermo", "blocco", "bloccato", "scadenza", "entro domani", "urgent",
        "subito", "velocemente", "critico", "panico", "terribile"
    ]

    # Parole che riducono la priorità anche se il modello dice Alta
    DAMPENING_KEYWORDS = [
        "con calma", "non è urgente", "non urgente", "nessuna fretta",
        "quando potete", "appena possibile", "senza urgenza",
        "normale", "nessun problema", "funziona", "tutto ok", "informazione"
    ]

    # Parole che indicano soddisfazione (priorità Bassa)
    HAPPY_KEYWORDS = [
        "perfetto", "ottimo", "risolto", "funziona tutto", "eccellente",
        "bravi", "complimenti"
    ]

    # ========================================
    # PRIORITY CALCULATION (LOGICA IBRIDA)
    # ========================================
    @staticmethod
    def calculate_priority(
        model: UnifiedModel,
        text: str
    ) -> Tuple[str, float, Dict[str, float], List[str]]:
        """
        Calcola la priorità usando logica ibrida ML + keyword.

        Pipeline decisionale:
        1. Ottieni probabilità dal modello ML
        2. Check per keyword critiche (forza Alta se no attenuanti)
        3. Check per score ML alto > 60% (forza Alta se no attenuanti)
        4. Check per feedback positivo (forza Bassa se no critiche)
        5. Fallback: usa predizione del modello

        Args:
            model: Istanza di UnifiedModel
            text: Testo del ticket da classificare

        Returns:
            Tupla con:
            - priority_label: "Alta", "Media" o "Bassa"
            - confidence: Probabilità della classe scelta
            - probabilities_dict: Dict con tutte le probabilità
            - trigger_keywords: Lista keyword che hanno innescato override
        """
        # ----------------------------------------
        # PREDIZIONE MODELLO ML
        # ----------------------------------------
        # Ottieni probabilità per Priorità (index 1)
        probs = model.predict_proba([text])[1][0]
        classes = model.pipeline.named_steps['clf'].estimators_[1].classes_

        # Mappa classi → probabilità
        prob_map = {c: p for c, p in zip(classes, probs)}

        # Score specifico per priorità Alta
        score_alta = prob_map.get('Alta', 0.0)
        debug_probs = prob_map.copy()

        # Normalizza testo per keyword search
        text_lower = text.lower()

        # ----------------------------------------
        # CHECK KEYWORDS
        # ----------------------------------------
        is_critical = any(kw in text_lower for kw in TicketService.CRITICAL_KEYWORDS)
        is_dampener = any(kw in text_lower for kw in TicketService.DAMPENING_KEYWORDS)
        is_happy = any(kw in text_lower for kw in TicketService.HAPPY_KEYWORDS)

        # ----------------------------------------
        # REGOLE DI OVERRIDE
        # ----------------------------------------

        # Regola 1: Keywords critiche senza attenuanti → ALTA
        if is_critical and not is_dampener:
            # Trova la keyword specifica che ha innescato
            trigger_word = next(
                (kw for kw in TicketService.CRITICAL_KEYWORDS if kw in text_lower),
                "keyword"
            )
            debug_probs['Alta'] = 0.99
            debug_probs['Media'] = 0.01
            debug_probs['Bassa'] = 0.00
            return "Alta", 0.99, debug_probs, [trigger_word]

        # Regola 2: Score alto per Alta priorità (>60%) → ALTA
        if score_alta > 0.60 and not is_dampener:
            return "Alta", score_alta, debug_probs, ["Rischio Statistico Alta Priorità"]

        # Regola 3: Feedback positivo senza critiche → BASSA
        if is_happy and not is_critical:
            debug_probs['Alta'] = 0.00
            debug_probs['Media'] = 0.01
            debug_probs['Bassa'] = 0.99
            return "Bassa", 0.99, debug_probs, ["Feedback Positivo"]

        # ----------------------------------------
        # FALLBACK: PREDIZIONE MODELLO
        # ----------------------------------------
        # Nessuna regola applicata: usa predizione del modello
        pred_class = model.predict([text])[0][1]
        score = prob_map.get(pred_class, 0.0)
        return pred_class, score, debug_probs, []

    # ========================================
    # SINGLE TEXT CLASSIFICATION
    # ========================================
    @staticmethod
    def classify_text(model: UnifiedModel, text: str) -> Dict[str, Any]:
        """
        Esegue la classificazione completa di un ticket.

        Separa la logica:
        - Categoria: Predizione pura del modello ML
        - Priorità: Logica ibrida ML + keyword

        Args:
            model: Istanza di UnifiedModel
            text: Testo del ticket

        Returns:
            Dict con:
            - category: Categoria assegnata
            - priority: Priorità assegnata
            - confidence: Confidenza della priorità
            - probabilities: Dict con tutte le probabilità
            - triggers: Lista keyword che hanno innescato override
            - success: True/False
            - error: Messaggio errore (se success=False)
        """
        try:
            # Predizione Categoria (solo modello ML)
            category = model.predict([text])[0][0]

            # Calcolo Priorità (logica ibrida)
            priority, confidence, probs, triggers = TicketService.calculate_priority(
                model, text
            )

            return {
                "category": category,
                "priority": priority,
                "confidence": float(confidence),
                "probabilities": {k: float(v) for k, v in probs.items()},
                "triggers": triggers,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    # ========================================
    # BATCH CLASSIFICATION
    # ========================================
    @staticmethod
    def classify_batch(model: UnifiedModel, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classifica una lista di testi.

        Applica classify_text a ogni elemento della lista.
        Mantiene l'ordine originale dei testi.

        Args:
            model: Istanza di UnifiedModel
            texts: Lista di testi da classificare

        Returns:
            Lista di dict con i risultati per ogni testo
        """
        results = []
        for text in texts:
            result = TicketService.classify_text(model, text)
            # Aggiungi anteprima testo
            result["text"] = text[:200] + "..." if len(text) > 200 else text
            results.append(result)
        return results
