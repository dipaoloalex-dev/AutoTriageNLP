"""
Modulo Modello Unificato
========================

Questo modulo definisce la classe `UnifiedModel`, che incapsula l'intera logica di Machine Learning
dell'applicazione AutoTriage. Implementa un approccio Multi-Task Learning per classificare
simultaneamente la categoria del ticket e la sua priorità.

Classi:
    UnifiedModel: Uno stimatore compatibile con scikit-learn composto da un vettorizzatore TF-IDF
                  e un classificatore MultiOutput basato su Regressione Logistica.
"""

import joblib
import pandas as pd
import numpy as np
from typing import List, Union, Optional, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

class UnifiedModel(BaseEstimator, ClassifierMixin):
    """
    Modello Unificato per la Classificazione Multi-Task dei Ticket.

    Questa classe avvolge una Pipeline scikit-learn che esegue la vettorizzazione del testo (TF-IDF)
    e la classificazione simultanea di due variabili target (Categoria e Priorità)
    utilizzando una strategia MultiOutputClassifier.

    Attributes:
        stopwords (List[str]): Lista di stop words italiane da ignorare durante la vettorizzazione.
        pipeline (Pipeline): L'oggetto pipeline scikit-learn sottostante.
    """

    def __init__(self) -> None:
        """
        Inizializza l'istanza di UnifiedModel.
        
        Configura la Pipeline con:
        1. TfidfVectorizer: Per convertire il testo grezzo in vettori di feature numeriche.
        2. MultiOutputClassifier: Per gestire indipendentemente target multipli (Categoria, Priorità).
        """

        # Definizione di stopwords italiane specifiche, inclusi formalismi comuni delle email
        self.stopwords: List[str] = [
            'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 
            'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 
            'è', 'sono', 'ho', 'hai', 'ha', 'abbiamo', 'avete', 'hanno',
            'che', 'chi', 'dove', 'quando', 'come', 'perché', 'salve', 
            'attendo', 'riscontro', 'cordiali', 'saluti', 'buongiorno', 'buonasera',
            'grazie', 'prego'
        ]

        # Inizializzazione della pipeline di elaborazione
        self.pipeline: Pipeline = Pipeline([

            # Step 1: Estrazione Feature
            # Limitiamo le feature a 5000 per mantenere il modello leggero e ridurre l'overfitting.
            # N-grams (1, 2) catturano il contesto (es. "non funziona" vs "funziona").
            ('tfidf', TfidfVectorizer(
                stop_words=self.stopwords,
                max_features=5000,
                ngram_range=(1, 2) 
            )),

            # Step 2: Classificazione
            # MultiOutputClassifier fitta un regressore per ogni variabile target.
            # class_weight='balanced' gestisce automaticamente lo sbilanciamento delle classi nei dati.
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
        """
        Addestra il modello sui dati forniti.

        Args:
            X (pd.Series): Le descrizioni testuali di input. Shape: (n_samples,).
            y (pd.DataFrame): Le etichette target. Deve includere le colonne ['category', 'priority']. 
                              Shape: (n_samples, 2).

        Returns:
            UnifiedModel: L'istanza del modello addestrato (self).
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: Union[pd.Series, List[str]]) -> np.ndarray:
        """
        Predice le classi per il testo di input fornito.

        Args:
            X (Union[pd.Series, List[str]]): Dati testuali di input da classificare. Shape: (n_samples,).

        Returns:
            np.ndarray: Etichette predette. Shape: (n_samples, 2).
                        Colonna 0: Categoria, Colonna 1: Priorità.
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X: Union[pd.Series, List[str]]) -> List[np.ndarray]:
        """
        Predice le probabilità di classe per ogni target.

        Args:
            X (Union[pd.Series, List[str]]): Dati testuali di input. Shape: (n_samples,).

        Returns:
            List[np.ndarray]: Una lista contenente due array (uno per stimatore).
                              - Indice 0: Probabilità per 'Categoria'. Shape: (n_samples, n_classes_cat).
                              - Indice 1: Probabilità per 'Priorità'. Shape: (n_samples, n_classes_pri).
        """
        return self.pipeline.predict_proba(X)

    def get_priority_score(self, X: Union[pd.Series, List[str]], priority_label: str = 'Alta') -> Union[np.ndarray, float]:
        """
        Estrae il punteggio di probabilità per una specifica classe di priorità (es. 'Alta').

        Questo helper è cruciale per la logica ibrida, permettendo di rilevare ticket ad alto rischio
        anche se la classificazione argmax assegnerebbe una priorità inferiore.

        Args:
            X (Union[pd.Series, List[str]]): Dati testuali di input.
            priority_label (str): L'etichetta della classe di cui estrarre la probabilità. Default 'Alta'.

        Returns:
            Union[np.ndarray, float]: Punteggi di probabilità per l'etichetta specificata.
                                      Ritorna 0.0 se l'etichetta non viene trovata nelle classi.
        """

        # Ottiene le probabilità per il task Priorità (indice 1)
        probs_priority = self.predict_proba(X)[1] 

        # Accede allo stimatore interno per la Priorità per mappare i nomi delle classi agli indici
        estimator_priority = self.pipeline.named_steps['clf'].estimators_[1]
        classes = estimator_priority.classes_

        if priority_label in classes:
            idx = list(classes).index(priority_label)
            # Ritorna la colonna corrispondente alla classe richiesta
            return probs_priority[:, idx]
        else:
            return 0.0

    def save(self, path: str) -> None:
        """
        Serializza l'intera pipeline del modello su disco.

        Args:
            path (str): Il percorso file dove salvare il modello (es. 'model.pkl').
        """
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> 'UnifiedModel':
        """
        Deserializza una pipeline del modello dal disco.

        Args:
            path (str): Il percorso file da cui caricare il modello.

        Returns:
            UnifiedModel: Un'istanza di UnifiedModel con la pipeline caricata.
        """
        instance = cls()
        instance.pipeline = joblib.load(path)
        return instance
