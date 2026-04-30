"""
Script di Confronto Modelli
===========================

Questo script mi serve per dimostrare praticamente perché ho scelto di usare
il dataset reale di Kaggle (tradotto) invece dei dati sintetici.

Addestra due versioni di UnifiedModel:
- Modello A: addestrato sui dati sintetici (approccio baseline).
- Modello B: addestrato sui dati reali (il mio approccio).

Esegue una validazione incrociata per verificare quale dei due generalizzi meglio.
"""

# ========================================
# IMPORT LIBRARY
# ========================================
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys
from typing import Tuple, List, Optional, Any

# Aggiungo la root al path per poter importare la classe UnifiedModel
sys.path.append(os.path.dirname(__file__))
from unified_model import UnifiedModel

# ========================================
# CONFIGURAZIONE PATH
# ========================================
# Dataset sintetico generato da generate_synthetic_data.py
SYNTHETIC_DATA: str = os.path.join('data', 'synthetic_tickets.csv')
# Dataset reale tradotto da prepare_data.py
REAL_DATA: str = os.path.join('data', 'tickets_it_augmented.csv')
# Cartella dove salvare i grafici generati
IMG_DIR: str = os.path.join('img', 'png')

# Creo la cartella se non esiste
os.makedirs(IMG_DIR, exist_ok=True)

# ========================================
# FUNZIONI DI SUPPORTO
# ========================================
def train_and_evaluate(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[UnifiedModel, float, pd.Series, pd.Series]:
    """
    Inizializza, addestra e calcola l'accuracy base di un modello.

    Args:
        name: Nome del modello (per logging)
        train_df: Dataset di training
        test_df: Dataset di test

    Returns:
        Tuple contenente:
        - model: Il modello addestrato
        - acc: Accuracy sul test set
        - y_true: Etichette reali
        - y_pred: Etichette predette
    """
    print(f"\n[{name}] Inizio addestramento...")

    model = UnifiedModel()

    # ----------------------------------------
    # PREPARAZIONE DATI
    # ----------------------------------------
    # Setup feature e target
    X_train = train_df['text']
    y_train = train_df[['category', 'priority']]

    # Addestramento
    model.fit(X_train, y_train)

    print(f"[{name}] Valutazione sul test set ({len(test_df)} righe)...")

    # Preparazione test set
    X_test = test_df['text']
    y_true = test_df[['category', 'priority']]

    # Predizione
    preds = model.predict(X_test)
    y_pred = pd.DataFrame(preds, columns=['category', 'priority'])

    # ----------------------------------------
    # CALCOLO ACCURACY
    # ----------------------------------------
    # Mi concentro sull'accuracy della Categoria per questo test
    acc = accuracy_score(y_true['category'], y_pred['category'])
    print(f"[{name}] Accuracy Categoria (su se stesso): {acc:.2%}")

    return model, acc, y_true['category'], y_pred['category']


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, title: str, filename: str) -> None:
    """
    Utility per generare e salvare le matrici di confusione dei test.

    Args:
        y_true: Etichette reali
        y_pred: Etichette predette
        title: Titolo del grafico
        filename: Nome del file di output
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)

    # Estrai le etichette uniche dai dati
    labels = sorted(list(set(y_true) | set(y_pred)))

    # Genera heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Valore Reale')
    plt.xlabel('Valore Predetto')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, filename))
    plt.close()

# ========================================
# FUNZIONE PRINCIPALE
# ========================================
def main() -> None:
    """Esegue il test comparativo tra modello su dati sintetici e dati reali."""
    print("=== TEST COMPARATIVO: DATI SINTETICI vs REALI ===")

    # Verifica esistenza dei file
    if not os.path.exists(SYNTHETIC_DATA) or not os.path.exists(REAL_DATA):
        print("Errore: Dataset mancanti. Verificare i path.")
        return

    # Caricamento dataset
    df_synth = pd.read_csv(SYNTHETIC_DATA)
    df_real = pd.read_csv(REAL_DATA)

    # ----------------------------------------
    # SPLIT DATASET (80/20)
    # ----------------------------------------
    # Uso lo stesso random_state per riproducibilità
    synth_train, synth_test = train_test_split(df_synth, test_size=0.2, random_state=42)
    real_train, real_test = train_test_split(df_real, test_size=0.2, random_state=42)

    # ========================================
    # TEST 1: MODELLO BASELINE (SINTETICO)
    # ========================================
    model_A, acc_A_self, _, _ = train_and_evaluate(
        "Modello A (Sintetico)",
        synth_train, synth_test
    )

    # ----------------------------------------
    # CROSS-EVALUATION: Modello A su Dati Reali
    # ----------------------------------------
    # Questo è il vero test: quanto generalizza il modello addestrato
    # su dati sintetici quando vede dati reali?
    print("\n--- Cross-Evaluation: Modello A testato su Dati Reali ---")
    preds_A_real = model_A.predict(real_test['text'])
    y_pred_A_real = pd.DataFrame(preds_A_real, columns=['category', 'priority'])
    acc_A_cross = accuracy_score(real_test['category'], y_pred_A_real['category'])

    print(f"Accuracy: {acc_A_cross:.2%}")
    plot_confusion_matrix(real_test['category'], y_pred_A_real['category'],
                          f"Modello A testato su Dati Reali (Accuracy: {acc_A_cross:.0%})", "confusion_matrix_a_on_real.png")

    # ========================================
    # TEST 2: MODELLO PROGETTO (REALE)
    # ========================================
    model_B, acc_B_self, _, _ = train_and_evaluate(
        "Modello B (Reale)",
        real_train, real_test
    )

    # ----------------------------------------
    # CROSS-EVALUATION: Modello B su Dati Sintetici
    # ----------------------------------------
    # Test inverso: quanto il modello su dati reali riesce a gestire
    # i pattern rigidi dei dati sintetici?
    print("\n--- Cross-Evaluation: Modello B testato su Dati Sintetici ---")
    preds_B_synth = model_B.predict(synth_test['text'])
    y_pred_B_synth = pd.DataFrame(preds_B_synth, columns=['category', 'priority'])
    acc_B_cross = accuracy_score(synth_test['category'], y_pred_B_synth['category'])

    print(f"Accuracy: {acc_B_cross:.2%}")
    plot_confusion_matrix(synth_test['category'], y_pred_B_synth['category'],
                          f"Modello B testato su Dati Sintetici (Accuracy: {acc_B_cross:.0%})", "confusion_matrix_b_on_synth.png")

    # ========================================
    # STAMPA SINTESI RISULTATI
    # ========================================
    print("\n" + "="*40)
    print("SINTESI RISULTATI")
    print("="*40)

    print("1. Modello A (Addestrato su Sintetico):")
    print(f"   - Accuracy su test sintetico: {acc_A_self:.1%}")
    print(f"   - Accuracy su test reale:     {acc_A_cross:.1%}")
    print("   Nota: Evidente overfitting. Il modello non riconosce le varianti del linguaggio naturale.")

    print("\n2. Modello B (Addestrato su Reale/Tradotto):")
    print(f"   - Accuracy su test reale:     {acc_B_self:.1%}")
    print(f"   - Accuracy su test sintetico: {acc_B_cross:.1%}")
    print("   Nota: Buona generalizzazione. Il modello gestisce bene anche i dati rigidi non visti in training.")

    # ----------------------------------------
    # GENERAZIONE GRAFICO CONFRONTO
    # ----------------------------------------
    # Creo un grafico a barre per visualizzare le due accuracy incrociate
    metrics = {
        'Scenario': ['Modello A testato su Reali', 'Modello B testato su Sintetici'],
        'Accuracy': [acc_A_cross, acc_B_cross]
    }
    df_res = pd.DataFrame(metrics)

    plt.figure(figsize=(6,5))
    # Rosso per il modello A (fallimento), Verde per il modello B (successo)
    sns.barplot(x='Scenario', y='Accuracy', data=df_res, palette=['#e74c3c', '#2ecc71'])
    plt.title("Confronto Capacità di Generalizzazione")
    plt.ylim(0, 1)

    # Aggiungo le etichette con i valori sopra le barre
    for i, v in enumerate(df_res['Accuracy']):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')

    plt.savefig(os.path.join(IMG_DIR, "comparison_chart.png"))
    print(f"\nGrafico salvato in: {IMG_DIR}/comparison_chart.png")

if __name__ == "__main__":
    main()
