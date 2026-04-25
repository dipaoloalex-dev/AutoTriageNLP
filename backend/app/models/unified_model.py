"""
========================================
Modello Unificato - Wrapper per FastAPI
========================================

Questa classe gestisce l'intera logica predittiva del progetto.
È una versione adattata del modello originale per l'uso con FastAPI.

Differisce dalla versione src/ per:
- Gestione errori migliorata con logging
- Pattern Singleton per il caricamento modello
- Pipeline inizializzata lazy (solo al caricamento)
"""

import joblib
import pandas as pd
import numpy as np
from typing import List, Union, Tuple, Dict, Any, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# ========================================
# CLASSE UNIFIED MODEL
# ========================================
class UnifiedModel(BaseEstimator, ClassifierMixin):
    """
    Wrapper per la pipeline scikit-learn.
    Esegue vettorizzazione testo e doppia classificazione.
    """

    def __init__(self) -> None:
        # ----------------------------------------
        # STOPWORDS ITALIANE CUSTOM
        # ----------------------------------------
        # Include convenevoli email comuni che non portano informazione
        self.stopwords: List[str] = [
            'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
            'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',
            'è', 'sono', 'ho', 'hai', 'ha', 'abbiamo', 'avete', 'hanno',
            'che', 'chi', 'dove', 'quando', 'come', 'perché', 'salve',
            'attendo', 'riscontro', 'cordiali', 'saluti', 'buongiorno', 'buonasera',
            'grazie', 'prego'
        ]

        # ----------------------------------------
        # PIPELINE (inizializzata lazy)
        # ----------------------------------------
        # La pipeline viene creata solo durante il training o il caricamento
        self.pipeline: Optional[Pipeline] = None

    def fit(self, X: pd.Series, y: pd.DataFrame) -> 'UnifiedModel':
        """
        Addestra la pipeline sui dati testuali e sui due target.

        Args:
            X: Series con il testo dei ticket
            y: DataFrame con colonne ['category', 'priority']

        Returns:
            Self per method chaining
        """
        self.pipeline = Pipeline([
            # Vettorizzazione TF-IDF
            ('tfidf', TfidfVectorizer(
                stop_words=self.stopwords,
                max_features=5000,         # Top 5000 parole più frequenti
                ngram_range=(1, 2)         # Unigrammi + bigrammi
            )),
            # Classificatore multi-output
            ('clf', MultiOutputClassifier(
                LogisticRegression(
                    solver='lbfgs',
                    max_iter=1000,             # Iterazioni massime
                    random_state=42,            # Riproducibilità
                    class_weight='balanced'   # Gestisce classi sbilanciate
                )
            ))
        ])
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: Union[pd.Series, List[str]]) -> np.ndarray:
        """
        Restituisce un array con due colonne: [Categoria, Priorità]

        Args:
            X: Testo/i da classificare

        Returns:
            numpy array con shape (n_samples, 2)
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load() first.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: Union[pd.Series, List[str]]) -> List[np.ndarray]:
        """
        Restituisce le probabilità separate per Categoria e Priorità.

        Args:
            X: Testo/i da classificare

        Returns:
            Lista di 2 array: [proba_categoria, proba_priorità]
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load() first.")
        return self.pipeline.predict_proba(X)

    def get_priority_score(self, X: Union[pd.Series, List[str]], priority_label: str = 'Alta') -> Union[np.ndarray, float]:
        """
        Estrae solo la probabilità di una specifica classe di priorità.

        Args:
            X: Testo/i da classificare
            priority_label: Classe di cui voglio la probabilità

        Returns:
            Array di probabilità o singolo valore float
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load() first.")

        # Ottengo probabilità priorità (index 1)
        probs_priority = self.predict_proba(X)[1]
        estimator_priority = self.pipeline.named_steps['clf'].estimators_[1]
        classes = estimator_priority.classes_

        if priority_label in classes:
            idx = list(classes).index(priority_label)
            return probs_priority[:, idx]
        else:
            return 0.0

    def save(self, path: str) -> None:
        """
        Salva il modello addestrato in formato binario (.pkl).

        Args:
            path: Percorso dove salvare il file
        """
        if self.pipeline is None:
            raise ValueError("No model to save. Train the model first.")
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> 'UnifiedModel':
        """
        Carica un modello precedentemente salvato dal disco.

        Args:
            path: Percorso del file .pkl da caricare

        Returns:
            Istanza di UnifiedModel con la pipeline caricata
        """
        instance = cls()
        try:
            instance.pipeline = joblib.load(path)
            logger.info(f"Model loaded successfully from {path}")
        except FileNotFoundError:
            logger.error(f"Model file not found at {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        return instance

# ========================================
# MODEL MANAGER (SINGLETON)
# ========================================
class ModelManager:
    """
    Singleton manager per il modello ML.

    Garantisce che il modello sia caricato una sola volta in memoria
    e riutilizzato per tutte le richieste successive.
    """

    _instance: Optional['ModelManager'] = None  # Istanza singleton
    _model: Optional[UnifiedModel] = None         # Cache modello caricato

    def __new__(cls):
        """Implementa pattern Singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self) -> UnifiedModel:
        """
        Restituisce il modello ML, caricandolo se necessario.

        Returns:
            Istanza di UnifiedModel pronta all'uso

        Note:
            Prima chiamata: carica il modello dal disco
            Chiamate successive: ritorna l'istanza cachata
        """
        if self._model is None:
            logger.info(f"Loading model from {settings.MODEL_PATH}")
            self._model = UnifiedModel.load(settings.MODEL_PATH)
        return self._model

    def reload_model(self) -> UnifiedModel:
        """
        Ricarica il modello dal disco (resetta la cache).

        Returns:
            Istanza di UnifiedModel ricaricata

        Note:
            Utile per aggiornare il modello senza riavviare l'API
        """
        self._model = None
        return self.get_model()
