"""
Script di Addestramento Modello
===============================

Questo script si occupa del training vero e proprio del modello unificato.
Legge i dati preparati dallo script precedente, fa lo split (80/20),
addestra l'algoritmo sia per la Categoria che per la Priorità, 
e infine salva le metriche (grafici e JSON) e il file .pkl del modello.
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
from typing import List, Dict, Any

# Aggiungo la root al path per poter importare la classe UnifiedModel
sys.path.append(os.path.dirname(__file__))
from unified_model import UnifiedModel

# --- CONFIGURAZIONE FILE ---
DATA_PATH: str = os.path.join('data', 'tickets_it_augmented.csv') 
MODEL_PATH: str = os.path.join('models', 'unified_model.pkl')     
IMG_DIR: str = os.path.join('assets', 'img', 'png')              
REPORT_FILE: str = "metrics_report_unified.txt"               

def save_confusion_matrix(y_true, y_pred, classes: List[str], title: str, filename: str) -> None:
    """Utility per generare e salvare le matrici di confusione."""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Valore Reale')
    plt.xlabel('Valore Predetto')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, filename))
    plt.close() 

def save_accuracy_chart(accuracy: float, prefix: str) -> None:
    """Salva un semplice grafico a barre per l'accuratezza globale."""
    plt.figure(figsize=(6, 5))
    color = 'Purples' if prefix == 'category' else 'RdPu' 
    
    df_acc = pd.DataFrame({'Metric': ['Accuracy'], 'Value': [accuracy]})
    sns.barplot(x='Metric', y='Value', data=df_acc, palette=color)
    
    plt.title(f'Global Accuracy - Dettaglio per {prefix.capitalize()}')
    plt.ylim(0, 1.1) 
    plt.ylabel('Score')
    
    # Scrivo il valore esatto sopra la barra
    plt.text(0, accuracy + 0.02, f"{accuracy:.2f}", ha='center', fontweight='bold')
    
    plt.savefig(os.path.join(IMG_DIR, f'accuracy_{prefix}.png'))
    plt.close()

def save_metric_charts(report_dict: Dict[str, Any], prefix: str) -> None:
    """Estrae Precision, Recall e F1-Score dal report di sklearn e ne fa dei grafici."""
    
    # Escludo le medie per avere solo le classi singole
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
        
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
            
        filename = f"{metric.replace('-', '')}_{prefix}.png" 
        plt.savefig(os.path.join(IMG_DIR, filename))

def main() -> None:
    print("Avvio addestramento del modello unificato...")
    
    if not os.path.exists(DATA_PATH):
        print(f"Errore: Dataset '{DATA_PATH}' non trovato.")
        print("Devi prima eseguire prepare_data.py per creare il file tradotto.")
        return

    print("Caricamento dataset in corso...")
    df = pd.read_csv(DATA_PATH)
    print(f"Ticket caricati: {len(df)}")

    X = df['text']
    y = df[['category', 'priority']]
    
    print("Suddivisione dei dati (80% training, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Inizio addestramento (potrebbe volerci qualche istante)...")
    model = UnifiedModel()
    model.fit(X_train, y_train)
    
    print("Addestramento completato. Inizio valutazione sul Test Set...")
    predictions = model.predict(X_test)
    y_pred_df = pd.DataFrame(predictions, columns=['category', 'priority'], index=X_test.index)
    
    # --- ANALISI CATEGORIA ---
    print("\n--- RISULTATI: CATEGORIA ---")
    cat_classes = sorted(y['category'].unique())
    
    report_cat_str = classification_report(y_test['category'], y_pred_df['category'])
    report_cat_dict = classification_report(y_test['category'], y_pred_df['category'], output_dict=True)
    
    print(report_cat_str)
    
    save_confusion_matrix(y_test['category'], y_pred_df['category'], cat_classes, 
                          "Matrice di Confusione - Categoria", "confusion_matrix_category.png")
    save_metric_charts(report_cat_dict, 'category')
    save_accuracy_chart(report_cat_dict['accuracy'], 'category')
    
    # --- ANALISI PRIORITÀ ---
    print("\n--- RISULTATI: PRIORITÀ ---")
    unique_pri = sorted(list(set(y_test['priority']) | set(y_pred_df['priority'])))
    
    report_pri_str = classification_report(y_test['priority'], y_pred_df['priority'])
    report_pri_dict = classification_report(y_test['priority'], y_pred_df['priority'], output_dict=True)
    
    print(report_pri_str)
    
    save_confusion_matrix(y_test['priority'], y_pred_df['priority'], unique_pri, 
                          "Matrice di Confusione - Priorità", "confusion_matrix_priority.png")
    save_metric_charts(report_pri_dict, 'priority')
    save_accuracy_chart(report_pri_dict['accuracy'], 'priority')

    print("Salvataggio modello...")
    model.save(MODEL_PATH)
    
    # Salvo un file di testo con i report di sklearn per comodità
    with open(REPORT_FILE, "w") as f:
        f.write("=== REPORT VALIDAZIONE MODELLO ===\n\n")
        f.write("--- CATEGORIA ---\n")
        f.write(report_cat_str)
        f.write("\n\n--- PRIORITÀ ---\n")
        f.write(report_pri_str)
    
    # Salvo questo JSON che serve all'interfaccia Streamlit per leggere rapidamente i KPI
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
        
    print("-" * 30)
    print("Processo terminato con successo.")
    print(f"File del modello salvato in: {MODEL_PATH}")
    print(f"Grafici aggiornati in: {IMG_DIR}")

if __name__ == "__main__":
    main()
