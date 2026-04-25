"""
========================================
LIME Service - Spiegazioni Predizioni
========================================

Servizio per l'analisi LIME (Local Interpretable Model-agnostic Explanations).
Fornisce spiegazioni sulle predizioni del modello identificando le parole
chiave che hanno influenzato la classificazione.

LIME è una tecnica di Explainable AI (XAI) che:
1. Crea perturbazioni locali del testo
2. Valuta come cambia la predizione
3. Identifica le parole più importanti per la decisione

Questo permette agli utenti di capire PERCHÉ un ticket ha ricevuto
una certa classificazione, aumentando la trasparenza e la fiducia.
"""

from typing import List, Tuple, Optional, Any, Dict
import logging

# ----------------------------------------
# IMPORT LIME (OPTIONAL)
# ----------------------------------------
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Install with: pip install lime")

from app.models.unified_model import UnifiedModel

logger = logging.getLogger(__name__)


# ========================================
# LIME SERVICE CLASS
# ========================================
class LimeService:
    """
    Servizio per generare spiegazioni LIME delle predizioni.

    Funzionalità:
    - Genera spiegazioni LIME per un testo
    - Formatta i risultati per il frontend
    - Unisce spiegazioni LIME con keyword triggers
    """

    # ----------------------------------------
    # VISUAL STOPWORDS
    # ----------------------------------------
    # Parole comuni da nascondere nelle spiegazioni
    # per focalizzarsi solo sui termini significativi
    VISUAL_STOPWORDS = {
        'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'in', 'di', 'a', 'è',
        'ho', 'sono', 'che', 'da', 'non', 'si', 'più', 'per', 'le',
        'alla', 'allo', 'delle', 'degli', 'della', 'del', 'al', 'su'
    }

    # ========================================
    # GENERA SPIEGAZIONE LIME
    # ========================================
    @staticmethod
    def explain_prediction(
        model: UnifiedModel,
        text: str,
        target_idx: int = 1,
        num_features: int = 6
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Genera una spiegazione LIME per la predizione.

        LIME lavora in due step:
        1. Crea N versioni perturbate del testo (rimuovendo parole casuali)
        2. Addestra un modello lineare interpretabile su queste perturbazioni

        Args:
            model: Il modello ML da spiegare
            text: Il testo da spiegare
            target_idx: 0 per Categoria, 1 per Priorità (default)
            num_features: Numero di parole chiave da estrarre

        Returns:
            Lista di tuple (parola, peso) ordinata per importanza.
            Peso positivo → parola spinge verso quella classe.
            None se LIME non è disponibile o errore.

        Note:
            Chiediamo il doppio delle feature (num_features * 2)
            perché molte verranno filtrate come stopwords.
        """
        if not LIME_AVAILABLE:
            logger.warning("LIME not available")
            return None

        try:
            # ----------------------------------------
            # SETUP EXPLAINER
            # ----------------------------------------
            # Ottieni l'estimatore specifico (Categoria o Priorità)
            estimator = model.pipeline.named_steps['clf'].estimators_[target_idx]
            explainer = LimeTextExplainer(class_names=estimator.classes_)

            # ----------------------------------------
            # WRAPPER PREDICT_PROBA
            # ----------------------------------------
            # LIME si aspetta una funzione che restituisca probabilità
            # Il nostro modello restituisce lista di 2 array [cat, priority]
            def predict_proba_wrapper(texts: List[str]) -> Any:
                return model.predict_proba(texts)[target_idx]

            # ----------------------------------------
            # GENERA SPIEGAZIONE
            # ----------------------------------------
            # explain_instance restituisce un oggetto Explanation
            exp = explainer.explain_instance(
                text,                          # Testo da spiegare
                predict_proba_wrapper,         # Funzione predizione
                num_features=num_features * 2  # Chiedi più feature
            )

            # ----------------------------------------
            # ESTRAI E FILTRA KEYWORD
            # ----------------------------------------
            keywords = []
            for word, weight in exp.as_list():
                word_clean = word.lower().strip()

                # Filtra stopwords e parole molto corte
                if (word_clean not in LimeService.VISUAL_STOPWORDS
                    and len(word_clean) > 2
                    and len(keywords) < num_features):
                    keywords.append((word, abs(weight)))

            return keywords if keywords else None

        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return None

    # ========================================
    # FORMATTAZIONE FRONTEND
    # ========================================
    @staticmethod
    def format_for_frontend(
        lime_results: Optional[List[Tuple[str, float]]],
        triggers: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Formatta i risultati LIME per il frontend.

        Unisce due fonti di spiegazione:
        1. Keyword triggers dal sistema ibrido (type="trigger")
        2. Parole LIME dal modello interpretabile (type="lime")

        Args:
            lime_results: Risultati grezzi da LIME [(parola, peso), ...]
            triggers: Keyword triggers dalla logica ibrida

        Returns:
            Lista di dict con:
            - word: La parola chiave
            - weight: Importanza (0-1)
            - type: "trigger" o "lime"

        Note:
            I triggers vengono sempre messi prima delle parole LIME.
            La lista viene ordinata per peso decrescente.
        """
        formatted = []

        # ----------------------------------------
        # AGGIUNGI TRIGGERS PRIMA
        # ----------------------------------------
        if triggers:
            for trigger in triggers:
                formatted.append({
                    "word": trigger,
                    "weight": 1.0,      # Triggers hanno massimo peso
                    "type": "trigger"
                })

        # ----------------------------------------
        # AGGIUNGI RISULTATI LIME
        # ----------------------------------------
        if lime_results:
            for word, weight in lime_results:
                formatted.append({
                    "word": word,
                    "weight": weight,
                    "type": "lime"
                })

        # ----------------------------------------
        # ORDINA E LIMITA
        # ----------------------------------------
        formatted.sort(key=lambda x: x["weight"], reverse=True)
        return formatted[:6]  # Max 6 parole chiave totali
