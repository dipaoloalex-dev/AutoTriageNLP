"""
Modello Unificato
=================

Questa classe gestisce l'intera logica predittiva del progetto.
Invece di avere due modelli separati, ho usato un approccio Multi-Task:
una singola pipeline con TF-IDF che classifica simultaneamente 
sia la Categoria che la Priorità del ticket.
"""

import joblib
import pandas as pd
import numpy as np
from typing import List, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

class UnifiedModel(BaseEstimator, ClassifierMixin):
    """
    Wrapper per la pipeline scikit-learn.
    Esegue vettorizzazione testo e doppia classificazione.
    """

    def __init__(self) -> None:
        # Stopwords italiane custom, ho aggiunto i soliti convenevoli delle email
        self.stopwords: List[str] = [
            'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 
            'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 
            'è', 'sono', 'ho', 'hai', 'ha', 'abbiamo', 'avete', 'hanno',
            'che', 'chi', 'dove', 'quando', 'come', 'perché', 'salve', 
            'attendo', 'riscontro', 'cordiali', 'saluti', 'buongiorno', 'buonasera',
            'grazie', 'prego'
        ]

        # Costruisco la pipeline
        self.pipeline: Pipeline = Pipeline([

            # 1. Vettorizzazione
            # Uso max_features=5000 per non appesantire il modello e n_gram=(1,2) 
            # per catturare concetti come "non funziona"
            ('tfidf', TfidfVectorizer(
                stop_words=self.stopwords,
                max_features=5000,
                ngram_range=(1, 2) 
            )),

            # 2. Classificazione Multipla
            # Il class_weight='balanced' mi aiuta a gestire le classi minoritarie
            ('clf', MultiOutputClassifier(
                LogisticRegression(
                    solver='lbfgs', 
                    max_iter=1000, 
                    random_state=42, 
                    class_weight='balanced'
                )
            ))
        ])

    def fit(self, X: pd.Series, y: pd.DataFrame) -> 'UnifiedModel':
        """Addestra la pipeline sui dati testuali e sui due target."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: Union[pd.Series, List[str]]) -> np.ndarray:
        """Restituisce un array con due colonne: [Categoria, Priorità]"""
        return self.pipeline.predict(X)

    def predict_proba(self, X: Union[pd.Series, List[str]]) -> List[np.ndarray]:
        """Restituisce le probabilità separate per Categoria (index 0) e Priorità (index 1)"""
        return self.pipeline.predict_proba(X)

    def get_priority_score(self, X: Union[pd.Series, List[str]], priority_label: str = 'Alta') -> Union[np.ndarray, float]:
        """
        Estrae solo la probabilità di una specifica classe di priorità (di default 'Alta').
        Mi serve per la logica ibrida nell'interfaccia web (app.py) per forzare
        i ticket a rischio.
        """
        
        # Prendo le probabilità relative solo al secondo classificatore (Priorità)
        probs_priority = self.predict_proba(X)[1] 

        # Recupero i nomi delle classi per capire in che colonna si trova 'Alta'
        estimator_priority = self.pipeline.named_steps['clf'].estimators_[1]
        classes = estimator_priority.classes_

        if priority_label in classes:
            idx = list(classes).index(priority_label)
            return probs_priority[:, idx]
        else:
            return 0.0

    def save(self, path: str) -> None:
        """Salva il modello addestrato in formato binario."""
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> 'UnifiedModel':
        """Carica un modello precedentemente salvato."""
        instance = cls()
        instance.pipeline = joblib.load(path)
        return instance
