"""
Modello Unificato
=================

Questa classe gestisce l'intera logica predittiva del progetto.
Invece di avere due modelli separati, ho usato un approccio Multi-Task:
una singola pipeline con TF-IDF che classifica simultaneamente
sia la Categoria che la Priorità del ticket.
"""

# ========================================
# IMPORT LIBRARY
# ========================================
# joblib: serializzazione del modello (save/load)
import joblib
# pandas: gestione dei dati in formato tabellare
import pandas as pd
# numpy: operazioni su array numerici
import numpy as np
# typing: type hints per better code documentation
from typing import List, Union
# sklearn.base: classi base per creare estimator personalizzati
from sklearn.base import BaseEstimator, ClassifierMixin
# sklearn.pipeline: per concatenare più trasformazioni e classificatore
from sklearn.pipeline import Pipeline
# TfidfVectorizer: trasformazione testo -> vettori numerici (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
# MultiOutputClassifier: wrapper per classificare multipli target simultaneamente
from sklearn.multioutput import MultiOutputClassifier
# LogisticRegression: algoritmo di classificazione lineare
from sklearn.linear_model import LogisticRegression

# ========================================
# CLASSE PRINCIPALE
# ========================================
class UnifiedModel(BaseEstimator, ClassifierMixin):
    """
    Wrapper per la pipeline scikit-learn.
    Esegue vettorizzazione testo e doppia classificazione.

    Eredita da BaseEstimator e ClassifierMixin per integrarsi perfettamente
    con l'ecosistema scikit-learn (cross-validation, grid search, ecc.)
    """

    def __init__(self) -> None:
        # ----------------------------------------
        # CONFIGURAZIONE STOPWORDS ITALIANE
        # ----------------------------------------
        # Lista di parole comuni da rimuovere durante la vettorizzazione.
        # Ho aggiunto i soliti convenevoli delle email perché non portano
        # informazione predittiva e appesantirebbero il modello.
        self.stopwords: List[str] = [
            'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
            'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',
            'è', 'sono', 'ho', 'hai', 'ha', 'abbiamo', 'avete', 'hanno',
            'che', 'chi', 'dove', 'quando', 'come', 'perché', 'salve',
            'attendo', 'riscontro', 'cordiali', 'saluti', 'buongiorno', 'buonasera',
            'grazie', 'prego'
        ]

        # ----------------------------------------
        # COSTRUZIONE PIPELINE
        # ----------------------------------------
        # La pipeline è una sequenza di operazioni eseguite in ordine:
        # Input testo -> [TF-IDF] -> [Classificatore] -> Output [Categoria, Priorità]
        self.pipeline: Pipeline = Pipeline([

            # 1. VETTORIZZAZIONE TESTO -> NUMERI
            # ----------------------------------
            # TF-IDF (Term Frequency - Inverse Document Frequency):
            # - Converte il testo in vettori numerici
            # - max_features=5000: tiene solo le 5000 parole più frequenti
            # - ngram_range=(1,2): considera singole parole + bigrammi (es: "non funziona")
            ('tfidf', TfidfVectorizer(
                stop_words=self.stopwords,
                max_features=5000,
                ngram_range=(1, 2)
            )),

            # 2. CLASSIFICATORE MULTI-TARGET
            # ------------------------------
            # MultiOutputClassifier addestra DUE LogisticRegression:
            # - Una per la Categoria (Tecnico/Amministrativo/Commerciale)
            # - Una per la Priorità (Alta/Media/Bassa)
            # - class_weight='balanced': compensa classi sbilanciate nel dataset
            ('clf', MultiOutputClassifier(
                LogisticRegression(
                    solver='lbfgs',       # Algoritmo di ottimizzazione
                    max_iter=1000,         # Iterazioni massime per convergenza
                    random_state=42,       # Seed per riproducibilità
                    class_weight='balanced'  # Pesi automatici per classi sbilanciate
                )
            ))
        ])

    # ========================================
    # METODI DI ADDESTRAMENTO
    # ========================================
    def fit(self, X: pd.Series, y: pd.DataFrame) -> 'UnifiedModel':
        """
        Addestra la pipeline sui dati testuali e sui due target.

        Args:
            X: Series contenente il testo dei ticket
            y: DataFrame con due colonne: ['category', 'priority']

        Returns:
            Self (per method chaining)
        """
        self.pipeline.fit(X, y)
        return self

    # ========================================
    # METODI DI PREDIZIONE
    # ========================================
    def predict(self, X: Union[pd.Series, List[str]]) -> np.ndarray:
        """
        Restituisce un array con due colonne: [Categoria, Priorità]

        Args:
            X: Testo/i del ticket da classificare

        Returns:
            numpy array con shape (n_samples, 2)
            - Colonna 0: Categoria predetta
            - Colonna 1: Priorità predetta
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X: Union[pd.Series, List[str]]) -> List[np.ndarray]:
        """
        Restituisce le probabilità separate per Categoria e Priorità.

        Returns:
            Lista di due array numpy:
            - [0]: Probabilità per le classi Categoria (Tecnico/Admin/Commerciale)
            - [1]: Probabilità per le classi Priorità (Alta/Media/Bassa)
        """
        return self.pipeline.predict_proba(X)

    def get_priority_score(self, X: Union[pd.Series, List[str]], priority_label: str = 'Alta') -> Union[np.ndarray, float]:
        """
        Estrae solo la probabilità di una specifica classe di priorità.

        MI SERVE PER: logica ibrida nell'interfaccia web (app.py)
        Se un ticket ha alta probabilità di essere "Alta" ma non viene classificato
        tale, possiamo forzarlo manualmente per sicurezza.

        Args:
            X: Testo/i del ticket
            priority_label: Classe di cui vogliamo la probabilità (default: 'Alta')

        Returns:
            Array di probabilità o singolo valore float
        """

        # Prendo le probabilità relative solo al secondo classificatore (Priorità)
        # predict_proba ritorna [proba_categoria, proba_priorità]
        probs_priority = self.predict_proba(X)[1]

        # Recupero i nomi delle classi per capire in che colonna si trova 'Alta'
        estimator_priority = self.pipeline.named_steps['clf'].estimators_[1]
        classes = estimator_priority.classes_

        if priority_label in classes:
            idx = list(classes).index(priority_label)
            return probs_priority[:, idx]
        else:
            return 0.0

    # ========================================
    # METODI DI PERSISTENZA
    # ========================================
    def save(self, path: str) -> None:
        """
        Salva il modello addestrato in formato binario (.pkl).

        Args:
            path: Percorso dove salvare il file (es: 'models/unified_model.pkl')
        """
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> 'UnifiedModel':
        """
        Carica un modello precedentemente salvato.

        Args:
            path: Percorso del file .pkl da caricare

        Returns:
            Istanza di UnifiedModel con la pipeline caricata
        """
        instance = cls()
        instance.pipeline = joblib.load(path)
        return instance
