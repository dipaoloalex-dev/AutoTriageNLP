"""
Script di Preparazione Dati
===========================

Questo script mi serve per prendere il dataset grezzo scaricato da Kaggle, 
pulirlo, mappare le categorie in italiano e tradurre il testo dei ticket.
Il risultato finale viene salvato in `tickets_it_augmented.csv`, 
che è il file che userò poi per addestrare il modello.
"""

import pandas as pd
import os
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Dict

# --- FILE PATHS ---
INPUT_FILE: str = os.path.join('data', 'kaggle_tickets.csv')
OUTPUT_FILE: str = os.path.join('data', 'tickets_it_augmented.csv')

# Mappatura per ridurre le categorie originali di Kaggle alle mie 3 macro-classi
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

# Uniformo le diciture delle priorità
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

def translate_text(text) -> str:
    """
    Traduce una singola stringa. Ritorna l'originale se ci sono errori o se il testo è troppo corto.
    """
    if not isinstance(text, str) or len(text) < 3:
        return str(text) if text is not None else ""
    
    try:
        translator = GoogleTranslator(source='en', target='it')
        return translator.translate(text)
    except Exception:
        # In caso di blocco delle API, restituisco il testo originale per non far crashare tutto
        return text 

def process_batch(texts: List[str]) -> List[str]:
    """
    Usa i thread per parallelizzare le chiamate di traduzione, altrimenti ci metterebbe ore.
    """
    # 5 worker mi sembrano un buon compromesso per non farmi bloccare da Google
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(tqdm(executor.map(translate_text, texts), total=len(texts), desc="Traduzione ticket"))
    return results

def main() -> None:
    print("Avvio preparazione del dataset...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Errore: Non trovo il file '{INPUT_FILE}'.")
        print("Assicurati di aver scaricato il dataset da Kaggle e di averlo messo nella cartella 'data/'.")
        return

    print("Caricamento del file grezzo...")
    try:
        df = pd.read_csv(INPUT_FILE, on_bad_lines='skip')
        print(f"File caricato: {len(df)} righe trovate.")
    except Exception as e:
        print(f"Errore durante la lettura del CSV: {e}")
        return

    # Metto i nomi delle colonne in minuscolo per comodità
    df.columns = [c.lower() for c in df.columns]
    
    # Cerco di capire come si chiamano le colonne nel CSV di Kaggle
    text_col = 'body' if 'body' in df.columns else 'text'
    cat_col = 'queue' if 'queue' in df.columns else 'category'
    pri_col = 'priority'
    
    if text_col not in df.columns or pri_col not in df.columns:
        print(f"Errore: Mancano le colonne base. Mi servono '{text_col}' e '{pri_col}'.")
        return

    print("Applico il mapping delle categorie e delle priorità...")
    if cat_col in df.columns:
        df['category_mapped'] = df[cat_col].map(CATEGORY_MAP).fillna("Tecnico") 
    else:
        df['category_mapped'] = "Tecnico"

    df['priority_mapped'] = df[pri_col].map(PRIORITY_MAP).fillna("Bassa")

    TRANSLATED_FILE: str = os.path.join('data', 'kaggle_tickets_it.csv')
    
    # Controllo se ho già fatto girare la traduzione in passato per risparmiare tempo
    if os.path.exists(TRANSLATED_FILE):
        print(f"Trovato file già tradotto ({TRANSLATED_FILE}). Salto le chiamate API.")
        
        df_translated = pd.read_csv(TRANSLATED_FILE, on_bad_lines='skip')
        
        # Se i file hanno lunghezze diverse, taglio alla lunghezza minima per evitare errori di allineamento
        if len(df_translated) != len(df):
            print("Attenzione: lunghezze diverse tra file grezzo e tradotto. Allineo i dati.")
            min_len = min(len(df), len(df_translated))
            df = df.iloc[:min_len]
            df_translated = df_translated.iloc[:min_len]
        
        df['text_it'] = df_translated['body']
        df['title_it'] = df_translated['subject'].fillna("Ticket Generico")
        
    else:
        print("File tradotto non trovato. Avvio la traduzione da zero (potrebbe volerci un po')...")
        
        # Do una mischiata ai dati
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Tolgo gli a capo che danno fastidio
        df[text_col] = df[text_col].astype(str).str.replace(r'\n', ' ', regex=True)
        
        df['text_it'] = process_batch(df[text_col].tolist())
        
        subject_col = 'subject' if 'subject' in df.columns else None
        if subject_col:
            print("Traduco anche gli oggetti dei ticket...")
            df['title_it'] = process_batch(df[subject_col].fillna("").astype(str).tolist())
        else:
            df['title_it'] = "Ticket Generico"

    df['id'] = range(1, len(df) + 1)
    
    # Assemblo il dataframe finale pulito
    final_df = pd.DataFrame({
        'id': df['id'],
        'title': df['title_it'],
        'body': df[text_col], # Tengo l'inglese per riferimento
        'text': df['text_it'], # L'italiano che userò per il training
        'category': df['category_mapped'],
        'priority': df['priority_mapped']
    })
    
    # Tolgo i ticket vuoti o con solo due lettere
    initial_len = len(final_df)
    final_df = final_df[final_df['text'].str.len() > 5]
    if len(final_df) < initial_len:
         print(f"Pulizia: scartati {initial_len - len(final_df)} ticket troppo corti.")
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("-" * 30)
    print(f"Finito! Il dataset preparato è stato salvato in: {OUTPUT_FILE}")
    print(f"Totale ticket validi per il training: {len(final_df)}")

if __name__ == "__main__":
    main()
