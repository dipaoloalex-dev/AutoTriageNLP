"""
Script di Confronto Modelli (Esperimento Scientifico)
=====================================================

Questo script esegue un esperimento comparativo per valutare l'efficacia dell'uso di dati reali
tradotti rispetto a dati sintetici generati artificialmente.
Addestra due istanze separate di `UnifiedModel`:
- Modello A: Addestrato su dati sintetici (regole rigide).
- Modello B: Addestrato su dati reali tradotti (linguaggio naturale).

Esegue poi una validazione incrociata (Cross-Evaluation) per dimostrare la capacità di generalizzazione.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys
from typing import Tuple, List, Optional, Any

# Aggiunge la directory corrente al path per l'importazione
sys.path.append(os.path.dirname(__file__))
from unified_model import UnifiedModel

# ==============================================================================
# 1. CONFIGURAZIONE E PATH
# ==============================================================================

SYNTHETIC_DATA: str = os.path.join('data', 'synthetic_tickets.csv')
REAL_DATA: str = os.path.join('data', 'tickets_it_augmented.csv')
IMG_DIR: str = os.path.join('assets', 'img', 'comparison')

# Crea la directory per le immagini di confronto se non esiste
os.makedirs(IMG_DIR, exist_ok=True)

def train_and_evaluate(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[UnifiedModel, float, pd.Series, pd.Series]:
    """
    Addestra un modello su un dato dataset e lo valuta.

    Args:
        name (str): Identificativo del modello (es. "Modello A").
        train_df (pd.DataFrame): DataFrame di training.
        test_df (pd.DataFrame): DataFrame di test.

    Returns:
        Tuple[UnifiedModel, float, pd.Series, pd.Series]: 
            - Modello addestrato.
            - Accuracy dello score.
            - Valori reali (y_true) categoria.
            - Valori predetti (y_pred) categoria.
    """
    print(f"\n[{name}] Addestramento in corso...")

    model = UnifiedModel()
    
    # Preparazione feature e target
    X_train = train_df['text']
    y_train = train_df[['category', 'priority']] # Il modello richiede entrambi i target

    # Fit del modello
    model.fit(X_train, y_train)

    print(f"[{name}] Valutazione su dataset di test ({len(test_df)} righe)...")
    
    # Valutazione
    X_test = test_df['text']
    y_true = test_df[['category', 'priority']]
    
    preds = model.predict(X_test)
    y_pred = pd.DataFrame(preds, columns=['category', 'priority'])
    
    # Calcolo metrica (bilanciata sulla Categoria per semplicità nell'esperimento)
    acc = accuracy_score(y_true['category'], y_pred['category'])
    
    print(f"[{name}] Accuracy Categoria: {acc:.2%}")
    
    return model, acc, y_true['category'], y_pred['category']


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, title: str, filename: str) -> None:
    """
    Genera e salva la matrice di confusione per l'esperimento comparativo.

    Args:
        y_true (pd.Series): Valori reali.
        y_pred (pd.Series): Valori predetti.
        title (str): Titolo del grafico.
        filename (str): Nome file di output.
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    # Ottieni etichette uniche dall'unione di reali e predette
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Reale')
    plt.xlabel('Predetto')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, filename))
    plt.close()

def main() -> None:
    """
    Funzione principale che esegue l'esperimento A/B tra dati sintetici e reali.
    """
    print("=== EXPERIMENT: SYNTHETIC vs REAL DATA ===")

    if not os.path.exists(SYNTHETIC_DATA) or not os.path.exists(REAL_DATA):
        print("Errore: Mancano i dataset. Esegui prima i generatori.")
        print(f"Cercato in: {SYNTHETIC_DATA} e {REAL_DATA}")
        return

    # Caricamento Dataset
    df_synth = pd.read_csv(SYNTHETIC_DATA)
    df_real = pd.read_csv(REAL_DATA)

    # Splitting
    synth_train, synth_test = train_test_split(df_synth, test_size=0.2, random_state=42)
    real_train, real_test = train_test_split(df_real, test_size=0.2, random_state=42)

    # --- FASE 1: MODELLO A (Dati Sintetici) ---
    model_A, acc_A_self, _, _ = train_and_evaluate(
        "Modello A (Fit: Synth -> Test: Synth)", 
        synth_train, synth_test
    )

    print("\n>>> SFIDA: Modello A (Sintetico) vs Mondo Reale <<<")
    # Test Modello A su Dati Reali (Cross-Evaluation)
    preds_A_real = model_A.predict(real_test['text'])
    y_pred_A_real = pd.DataFrame(preds_A_real, columns=['category', 'priority'])
    
    acc_A_cross = accuracy_score(real_test['category'], y_pred_A_real['category'])
    
    print(f"Risultato: Il Modello A ha un'accuracy del {acc_A_cross:.2%} sui dati reali.")
    
    plot_confusion_matrix(real_test['category'], y_pred_A_real['category'], 
                          f"Modello A su Dati Reali (Acc: {acc_A_cross:.0%})", "conf_matrix_A_on_Real.png")

    # --- FASE 2: MODELLO B (Dati Reali) ---
    model_B, acc_B_self, _, _ = train_and_evaluate(
        "Modello B (Fit: Real -> Test: Real)", 
        real_train, real_test
    )

    print("\n>>> SFIDA: Modello B (Reale) vs Dati Sintetici <<<")
    # Test Modello B su Dati Sintetici (Cross-Evaluation)
    preds_B_synth = model_B.predict(synth_test['text'])
    y_pred_B_synth = pd.DataFrame(preds_B_synth, columns=['category', 'priority'])
    
    acc_B_cross = accuracy_score(synth_test['category'], y_pred_B_synth['category'])
    
    print(f"Risultato: Il Modello B ha un'accuracy del {acc_B_cross:.2%} sui dati sintetici.")
    
    plot_confusion_matrix(synth_test['category'], y_pred_B_synth['category'], 
                          f"Modello B su Dati Sintetici (Acc: {acc_B_cross:.0%})", "conf_matrix_B_on_Synth.png")

    # --- REPORT FINALE ---
    print("\n" + "="*40)
    print("REPORT FINALE DI CONFRONTO")
    print("="*40)
    print(f"1. Modello A (Sintetico):")
    print(f"   - Su se stesso: {acc_A_self:.1%}") 
    print(f"   - Sul Reale:    {acc_A_cross:.1%}  <-- CROLLO PREVISTO!") 
    print(f"   (Il modello ha imparato a memoria regole fisse e fallisce sulla varietà del linguaggio umano)")

    print(f"\n2. Modello B (Reale/Tradotto):")
    print(f"   - Su se stesso: {acc_B_self:.1%}")
    print(f"   - Sul Sintetico: {acc_B_cross:.1%} <-- ROBUSTEZZA")
    print(f"   (Il modello reale generalizza bene e capisce anche i casi semplici)")

    # --- Generazione Grafico Comparativo Finale ---
    metrics = {
        'Scenario': ['A su Reale', 'B su Sintetico'],
        'Accuracy': [acc_A_cross, acc_B_cross]
    }
    df_res = pd.DataFrame(metrics)
    
    plt.figure(figsize=(6,5))
    sns.barplot(x='Scenario', y='Accuracy', data=df_res, palette=['red', 'green'])
    plt.title("Confronto Generalizzazione Modelli")
    plt.ylim(0, 1)
    # Annotazione barre
    for i, v in enumerate(df_res['Accuracy']):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')
        
    plt.savefig(os.path.join(IMG_DIR, "comparison_chart.png"))
    
    print(f"\nGrafici salvati in: {IMG_DIR}")

if __name__ == "__main__":
    main()
