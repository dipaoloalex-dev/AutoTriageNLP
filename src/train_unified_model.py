"""
Pipeline di Addestramento Modello Unificato
===========================================

Questo script gestisce l'intero ciclo di vita del training per il modello UnifiedModel:
1. Caricamento del dataset aumentato e preprocessato.
2. Splitting del dataset in Training e Test set.
3. Addestramento del modello Multi-Task (Categoria + Priorità).
4. Valutazione dettagliata delle performance con metriche multiple.
5. Generazione di grafici analitici (Matrice di Confusione, Accuracy, Precision/Recall).
6. Salvataggio dell'artefatto del modello (.pkl) e dei report.
"""

import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
import json
from typing import List, Dict, Any, Tuple

# Aggiunge la directory corrente al path per importare i moduli locali
sys.path.append(os.path.dirname(__file__))
from unified_model import UnifiedModel

# ==============================================================================
# 1. CONFIGURAZIONE E PATH
# ==============================================================================

DATA_PATH: str = os.path.join('data', 'tickets_it_augmented.csv') 
MODEL_PATH: str = os.path.join('models', 'unified_model.pkl')     
IMG_DIR: str = os.path.join('assets', 'img', 'png')              
REPORT_FILE: str = "metrics_report_unified.txt"               

def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, classes: List[str], title: str, filename: str) -> None:
    """
    Genera e salva la matrice di confusione come immagine PNG.

    Args:
        y_true (pd.Series): Etichette reali (Ground Truth).
        y_pred (pd.Series): Etichette predette dal modello.
        classes (List[str]): Lista ordinata delle classi univoche.
        title (str): Titolo del grafico.
        filename (str): Nome del file di output.
    """
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Valore Reale (Ground Truth)')
    plt.xlabel('Valore Predetto dal Modello')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, filename))
    plt.close() 

def save_accuracy_chart(accuracy: float, prefix: str) -> None:
    """
    Salva un grafico a barre semplice che visualizza l'accuratezza globale.

    Args:
        accuracy (float): Valore di accuratezza (0.0 - 1.0).
        prefix (str): Prefisso per il nome file e colore ('category' o 'priority').
    """
    plt.figure(figsize=(6, 5))
    color = 'Purples' if prefix == 'category' else 'RdPu' 
    
    df_acc = pd.DataFrame({'Metric': ['Accuracy'], 'Value': [accuracy]})
    sns.barplot(x='Metric', y='Value', data=df_acc, palette=color)
    
    plt.title(f'Global Accuracy - {prefix.capitalize()}')
    plt.ylim(0, 1.1) 
    plt.ylabel('Score')
    
    # Annotazione del valore sopra la barra
    plt.text(0, accuracy + 0.02, f"{accuracy:.2f}", ha='center', fontweight='bold')
    
    plt.savefig(os.path.join(IMG_DIR, f'accuracy_{prefix}.png'))
    plt.close()

def save_metric_charts(report_dict: Dict[str, Any], prefix: str) -> None:
    """
    Salva grafici a barre per Precision, Recall e F1-Score per ogni classe.

    Args:
        report_dict (Dict[str, Any]): Dizionario di output da classification_report.
        prefix (str): Identificatore del task ('category' o 'priority').
    """

    # Filtra chiavi speciali (medie aggregate) per ottenere solo le classi
    classes = [k for k in report_dict.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
    metrics = ['precision', 'recall', 'f1-score']
    palettes = {'precision': 'Blues', 'recall': 'Greens', 'f1-score': 'Oranges'}
    
    for metric in metrics:
        values = [report_dict[c][metric] for c in classes]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=classes, y=values, palette=palettes[metric])
        plt.title(f'{metric.capitalize()} - Dettaglio per {prefix.capitalize()}')
        plt.ylabel('Score (0-1)')
        plt.ylim(0, 1.1)
        
        # Annotazione valori
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
            
        filename = f"{metric.replace('-', '')}_{prefix}.png" 
        plt.savefig(os.path.join(IMG_DIR, filename))

def main() -> None:
    """
    Main function della pipeline di training.
    """
    print("=== AUTO-TRIAGE TRAINING PIPELINE (APPROCCIO SCIENTIFICO) ===")
    
    # Verifica esistenza dati
    if not os.path.exists(DATA_PATH):
        print(f"[ERRORE] Dataset {DATA_PATH} mancante.")
        print("Esegui prima lo script: python src/prepare_data.py")
        return

    print(f"[INFO] Caricamento dataset: {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Records caricati: {len(df)}")

    # Preparazione Feature (X) e Target (y)
    X = df['text']
    y = df[['category', 'priority']]
    
    # Split training/test set (80/20)
    print("[INFO] Splitting dataset (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inizializzazione e Training
    print("[INFO] Addestramento UnifiedModel (Multi-Task Learning)...")
    model = UnifiedModel()
    model.fit(X_train, y_train)
    
    # Valutazione
    print("[INFO] Valutazione sul Test Set...")
    predictions = model.predict(X_test)
    
    # Conversione predizioni in DataFrame per facilitare l'analisi
    y_pred_df = pd.DataFrame(predictions, columns=['category', 'priority'], index=X_test.index)
    
    # --- REPORTING TASK 1: CATEGORIA ---
    print("\n--- PERFORMANCE TASK: CATEGORIA ---")
    cat_classes = sorted(y['category'].unique())
    
    report_cat_str = classification_report(y_test['category'], y_pred_df['category'])
    report_cat_dict = classification_report(y_test['category'], y_pred_df['category'], output_dict=True)
    
    print(report_cat_str)
    
    save_confusion_matrix(y_test['category'], y_pred_df['category'], cat_classes, 
                          "Matrice di Confusione - Categoria", "confusion_matrix_category.png")
    save_metric_charts(report_cat_dict, 'category')
    save_accuracy_chart(report_cat_dict['accuracy'], 'category')
    
    # --- REPORTING TASK 2: PRIORITÀ ---
    print("\n--- PERFORMANCE TASK: PRIORITÀ ---")
    # Usa unione set per garantire che tutte le classi appaiano anche se non predette
    unique_pri = sorted(list(set(y_test['priority']) | set(y_pred_df['priority'])))
    
    report_pri_str = classification_report(y_test['priority'], y_pred_df['priority'])
    report_pri_dict = classification_report(y_test['priority'], y_pred_df['priority'], output_dict=True)
    
    print(report_pri_str)
    
    save_confusion_matrix(y_test['priority'], y_pred_df['priority'], unique_pri, 
                          "Matrice di Confusione - Priorità", "confusion_matrix_priority.png")
    save_metric_charts(report_pri_dict, 'priority')
    save_accuracy_chart(report_pri_dict['accuracy'], 'priority')

    # Salvataggio Modello
    print(f"[INFO] Salvataggio artefatto modello in {MODEL_PATH}...")
    model.save(MODEL_PATH)
    
    # Salvataggio Report Testuale
    with open(REPORT_FILE, "w") as f:
        f.write("=== REPORT VALIDAZIONE UNIFIED MODEL ===\n\n")
        f.write("--- CATEGORIA ---\n")
        f.write(report_cat_str)
        f.write("\n\n--- PRIORITÀ ---\n")
        f.write(report_pri_str)
    
    # Salvataggio Metriche JSON (per la dashboard Streamlit)
    metrics_summary = {
        "category": {
            "accuracy": report_cat_dict['accuracy'],
            "precision": report_cat_dict['weighted avg']['precision'],
            "recall": report_cat_dict['weighted avg']['recall'],
            "f1": report_cat_dict['weighted avg']['f1-score']
        },
        "priority": {
            "accuracy": report_pri_dict['accuracy'],
            "precision": report_pri_dict['weighted avg']['precision'],
            "recall": report_pri_dict['weighted avg']['recall'],
            "f1": report_pri_dict['weighted avg']['f1-score']
        }
    }
    with open("metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f)
        
    print("-" * 50)
    print("[SUCCESS] Pipeline completata.")
    print(f"Modello salvato: {MODEL_PATH}")
    print(f"Grafici generati in: {IMG_DIR}")

if __name__ == "__main__":
    main()
