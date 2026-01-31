"""
Pipeline di Preparazione Dati
=============================

Questo modulo gestisce l'ingestione, la pulizia, la normalizzazione e la traduzione del dataset
raw di Kaggle. Prepara il file finale `tickets_it_augmented.csv` pronto per l'addestramento
del modello di Machine Learning.

Supporta:
- Mapping delle categorie e priorità a standard definiti.
- Traduzione multithread (o fallback sequenziale).
- Gestione intelligente del dataset pre-tradotto per efficienza.
"""

import pandas as pd
import os
import joblib
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from typing import List, Dict, Optional, Union, Any

# ==============================================================================
# 1. CONFIGURAZIONE E COSTANTI
# ==============================================================================

INPUT_FILE: str = os.path.join('data', 'kaggle_tickets.csv')
OUTPUT_FILE: str = os.path.join('data', 'tickets_it_augmented.csv')

# Mappatura per normalizzare le categorie del dataset originale in 3 macro-classi
CATEGORY_MAP: Dict[str, str] = {
    "Technical Support": "Tecnico",
    "IT Support": "Tecnico",
    "Service Outages and Maintenance": "Tecnico",
    "Product Support": "Tecnico",
    "Billing and Payments": "Amministrativo",
    "Returns and Exchanges": "Amministrativo",
    "Human Resources": "Amministrativo",
    "General Inquiry": "Commerciale",
    "Customer Service": "Commerciale",
    "Sales and Pre-Sales": "Commerciale"
}

# Mappatura per normalizzare i livelli di priorità in 3 classi standard
PRIORITY_MAP: Dict[str, str] = {
    "Critical": "Alta",
    "critical": "Alta",
    "High": "Alta",
    "high": "Alta",
    "Medium": "Media",
    "medium": "Media",
    "Normal": "Media",
    "normal": "Media",
    "Low": "Bassa",
    "low": "Bassa"
}

def translate_text(text: Union[str, Any]) -> str:
    """
    Traduce una stringa di testo dall'inglese all'italiano.

    Include controlli di validità per evitare chiamate API inutili su input vuoti o troppo brevi.

    Args:
        text (Union[str, Any]): Il testo da tradurre.

    Returns:
        str: Il testo tradotto in italiano o il testo originale in caso di errore/input non valido.
    """
    if not isinstance(text, str) or len(text) < 3:
        return str(text) if text is not None else ""
    
    try:
        translator = GoogleTranslator(source='en', target='it')
        return translator.translate(text)
    except Exception as e:
        return text 


def process_batch(texts: List[str]) -> List[str]:
    """
    Esegue la traduzione di una lista di testi in parallelo.

    Utilizza ThreadPoolExecutor per velocizzare le chiamate I/O bound alle API di traduzione.

    Args:
        texts (List[str]): Lista di stringhe da tradurre.

    Returns:
        List[str]: Lista corrispondente di stringhe tradotte.
    """
    translated: List[str] = []

    # Usiamo 5 worker per bilanciare velocità e rispetto dei rate limit
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(tqdm(executor.map(translate_text, texts), total=len(texts), desc="Traduzione in corso"))
        translated = results
    return translated

def main() -> None:
    """
    Pipeline Principale di Preparazione Dati.

    Esegue i seguenti step:
    1. Caricamento dataset raw.
    2. Normalizzazione nomi colonne e valori (Mapping).
    3. Gestione traduzione (riutilizzo file esistente o nuova traduzione).
    4. Pulizia finale e salvataggio dataset aumentato.
    """
    print("=== DATA PREPARATION PIPELINE V2.0 ===")
    print("Obiettivo: Ingestione, Normalizzazione e Traduzione del Dataset Kaggle")
    
    if not os.path.exists(INPUT_FILE):
        print(f"[ERRORE] File '{INPUT_FILE}' non trovato.")
        print("Azione richiesta: Scarica il dataset da Kaggle (Multilingual Customer Support Tickets)")
        print(f"e salvalo come 'kaggle_tickets.csv' nella cartella 'data/'.")
        return

    print(f"[INFO] Caricamento file grezzo da {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, on_bad_lines='skip')
        print(f"[INFO] Dataset caricato con successo: {len(df)} righe iniziali.")
    except Exception as e:
        print(f"[ERRORE] Impossibile leggere il file CSV: {e}")
        return

    # Normalizzazione nomi colonne a minuscolo
    df.columns = [c.lower() for c in df.columns]
    print(f"[DEBUG] Colonne rilevate: {df.columns.tolist()}")
    
    # Identificazione dinamica colonne target
    text_col: str = 'body' if 'body' in df.columns else 'text'
    cat_col: str = 'queue' if 'queue' in df.columns else 'category'
    pri_col: str = 'priority'
    
    if text_col not in df.columns or pri_col not in df.columns:
        print(f"[ERRORE] Colonne fondamentali mancanti. Cerco: '{text_col}' e '{pri_col}'.")
        return

    print("[INFO] Normalizzazione delle etichette (Mapping)...")
    if cat_col in df.columns:
        # Applica mapping categoria, default a 'Tecnico' per NaN
        df['category_mapped'] = df[cat_col].map(CATEGORY_MAP).fillna("Tecnico") 
    else:
        df['category_mapped'] = "Tecnico"

    # Applica mapping priorità, default a 'Bassa' per NaN
    df['priority_mapped'] = df[pri_col].map(PRIORITY_MAP).fillna("Bassa")

    TRANSLATED_FILE: str = os.path.join('data', 'kaggle_tickets_it.csv')
    
    # Logica intelligente: usa il file già tradotto se esiste (creato da translate_data.py)
    if os.path.exists(TRANSLATED_FILE):
        print(f"[INFO] Trovato dataset pre-tradotto: {TRANSLATED_FILE}")
        print("[INFO] Salto la fase di traduzione e utilizzo i dati pronti (Strategia Efficienza).")
        
        df_translated = pd.read_csv(TRANSLATED_FILE, on_bad_lines='skip')
        
        # Gestione disallineamento dimensioni (es. il file tradotto è parziale)
        if len(df_translated) != len(df):
            print("[WARN] Dimensioni diverse tra file raw e file tradotto. Procedo con join sicuro o taglio.")
            min_len = min(len(df), len(df_translated))
            df = df.iloc[:min_len]
            df_translated = df_translated.iloc[:min_len]
        
        df['text_it'] = df_translated['body']
        df['title_it'] = df_translated['subject'].fillna("Ticket Generic")
        
    else:
        print("[WARN] Dataset pre-tradotto NON trovato.")
        print("[INFO] Avvio processo di traduzione interno (Fallback)...")
        
        # Mescola il dataset per evitare bias di ordine
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"[INFO] Avvio traduzione massiva su {len(df)} ticket.")
        
        # Pulizia base del testo
        df[text_col] = df[text_col].astype(str).str.replace(r'\n', ' ', regex=True)
        
        # Traduzione corpo del ticket
        df['text_it'] = process_batch(df[text_col].tolist())
        
        subject_col = 'subject' if 'subject' in df.columns else None
        if subject_col:
            print("[INFO] Traduzione degli Oggetti (Subjects)...")
            df['title_it'] = process_batch(df[subject_col].fillna("").astype(str).tolist())
        else:
            df['title_it'] = "Ticket Generic"

    # Aggiungi ID incrementale
    df['id'] = range(1, len(df) + 1)
    
    # Costruisci DataFrame finale pulito
    final_df = pd.DataFrame({
        'id': df['id'],
        'title': df['title_it'],
        'body': df[text_col], # Testo originale (utile per riferimento)
        'text': df['text_it'], # Testo tradotto (usato per il training)
        'category': df['category_mapped'],
        'priority': df['priority_mapped']
    })
    
    # Filtra record con testo troppo breve (meno di 5 caratteri)
    initial_len = len(final_df)
    final_df = final_df[final_df['text'].str.len() > 5]
    if len(final_df) < initial_len:
         print(f"[WARN] Rimossi {initial_len - len(final_df)} ticket corrotti o troppo brevi.")
    
    # Salvataggio
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("-" * 50)
    print(f"[SUCCESS] Dataset generato e salvato in: {OUTPUT_FILE}")
    print(f"Totale Record Pronti per il Training: {len(final_df)}")
    print("Prossimo step suggerito: python src/train_unified_model.py")

if __name__ == "__main__":
    main()
