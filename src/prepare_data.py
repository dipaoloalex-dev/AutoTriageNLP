"""
Script di Preparazione Dati
===========================

Questo script mi serve per prendere il dataset grezzo scaricato da Kaggle,
pulirlo, mappare le categorie in italiano e tradurre il testo dei ticket.
Il risultato finale viene salvato in `tickets_it_augmented.csv`,
che è il file che userò poi per addestrare il modello.
"""

# ========================================
# IMPORT LIBRARY
# ========================================
import pandas as pd
import os
# GoogleTranslator per tradurre automaticamente dall'inglese all'italiano
from deep_translator import GoogleTranslator
# ThreadPoolExecutor per parallelizzare le traduzioni (più veloce)
from concurrent.futures import ThreadPoolExecutor
# tqdm per la barra di progresso durante la traduzione
from tqdm import tqdm
from typing import List, Dict

# ========================================
# CONFIGURAZIONE PATH
# ========================================
# File di input: dataset grezzo da Kaggle
INPUT_FILE: str = os.path.join('data', 'kaggle_tickets.csv')
# File di output: dataset pulito e tradotto
OUTPUT_FILE: str = os.path.join('data', 'tickets_it_augmented.csv')

# ========================================
# MAPPE DI TRADUZIONE ETICHETTE
# ========================================
# Mappatura per ridurre le categorie originali di Kaggle alle mie 3 macro-classi
CATEGORY_MAP: Dict[str, str] = {
    # Tutti i supporti tecnici vanno sotto "Tecnico"
    "Technical Support": "Tecnico",
    "IT Support": "Tecnico",
    "Service Outages and Maintenance": "Tecnico",
    "Product Support": "Tecnico",
    # Tutto ciò che riguarda fatturazione e HR va sotto "Amministrativo"
    "Billing and Payments": "Amministrativo",
    "Returns and Exchanges": "Amministrativo",
    "Human Resources": "Amministrativo",
    # Tutto il resto (inquiry, sales, customer service) va sotto "Commerciale"
    "General Inquiry": "Commerciale",
    "Customer Service": "Commerciale",
    "Sales and Pre-Sales": "Commerciale"
}

# Uniformo le diciture delle priorità (gestisce maiuscole/minuscole)
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

# ========================================
# FUNZIONI DI TRADUZIONE
# ========================================
def translate_text(text) -> str:
    """
    Traduce una singola stringa dall'inglese all'italiano.
    Ritorna l'originale se ci sono errori o se il testo è troppo corto.

    Args:
        text: Testo da tradurre

    Returns:
        Testo tradotto o originale in caso di errore
    """
    # Controllo di validità: deve essere stringa e lungo almeno 3 caratteri
    if not isinstance(text, str) or len(text) < 3:
        return str(text) if text is not None else ""

    try:
        translator = GoogleTranslator(source='en', target='it')
        return translator.translate(text)
    except Exception:
        # In caso di blocco delle API (rate limiting), restituisco il testo originale
        # per non far crashare tutto il processo
        return text

def process_batch(texts: List[str]) -> List[str]:
    """
    Usa i thread per parallelizzare le chiamate di traduzione.
    Senza parallelizzazione, tradurre migliaia di ticket richiederebbe ore.

    Args:
        texts: Lista di testi da tradurre

    Returns:
        Lista di testi tradotti
    """
    # 5 worker mi sembrano un buon compromesso:
    # - Troppi = Google potrebbe bloccare l'IP per rate limiting
    # - Troppo pochi = il processo diventa troppo lento
    with ThreadPoolExecutor(max_workers=5) as executor:
        # tqdm mostra la barra di progresso
        results = list(tqdm(executor.map(translate_text, texts), total=len(texts), desc="Traduzione ticket"))
    return results

# ========================================
# FUNZIONE PRINCIPALE
# ========================================
def main() -> None:
    """Esegue l'intera pipeline di preparazione dei dati."""
    print("Avvio preparazione del dataset...")

    # ----------------------------------------
    # VERIFICA FILE DI INPUT
    # ----------------------------------------
    if not os.path.exists(INPUT_FILE):
        print(f"Errore: Non trovo il file '{INPUT_FILE}'.")
        print("Assicurati di aver scaricato il dataset da Kaggle e di averlo messo nella cartella 'data/'.")
        return

    # ----------------------------------------
    # CARICAMENTO DATI
    # ----------------------------------------
    print("Caricamento del file grezzo...")
    try:
        # on_bad_lines='skip' salta le righe malformate del CSV
        df = pd.read_csv(INPUT_FILE, on_bad_lines='skip')
        print(f"File caricato: {len(df)} righe trovate.")
    except Exception as e:
        print(f"Errore durante la lettura del CSV: {e}")
        return

    # Metto i nomi delle colonne in minuscolo per gestire meglio le variazioni
    df.columns = [c.lower() for c in df.columns]

    # ----------------------------------------
    # IDENTIFICAZIONE COLONNE
    # ----------------------------------------
    # I dataset di Kaggle possono avere nomi colonne diversi.
    # Cerco di capire come si chiamano le colonne nel CSV.
    text_col = 'body' if 'body' in df.columns else 'text'
    cat_col = 'queue' if 'queue' in df.columns else 'category'
    pri_col = 'priority'

    if text_col not in df.columns or pri_col not in df.columns:
        print(f"Errore: Mancano le colonne base. Mi servono '{text_col}' e '{pri_col}'.")
        return

    # ----------------------------------------
    # APPLICAZIONE MAPPE CATEGORIA/PRIORITÀ
    # ----------------------------------------
    print("Applico il mapping delle categorie e delle priorità...")
    if cat_col in df.columns:
        # map() applica la traduzione, fillna() gestisce i casi non mappati
        df['category_mapped'] = df[cat_col].map(CATEGORY_MAP).fillna("Tecnico")
    else:
        # Se non c'è colonna categoria, assegno default "Tecnico"
        df['category_mapped'] = "Tecnico"

    df['priority_mapped'] = df[pri_col].map(PRIORITY_MAP).fillna("Bassa")

    # ----------------------------------------
    # GESTIONE TRADUZIONE (CACHE)
    # ----------------------------------------
    # Percorso del file tradotto (intermedio)
    TRANSLATED_FILE: str = os.path.join('data', 'kaggle_tickets_it.csv')

    # Controllo se ho già fatto girare la traduzione in passato per risparmiare tempo
    if os.path.exists(TRANSLATED_FILE):
        print(f"Trovato file già tradotto ({TRANSLATED_FILE}). Salto le chiamate API.")

        df_translated = pd.read_csv(TRANSLATED_FILE, on_bad_lines='skip')

        # Se i file hanno lunghezze diverse, taglio alla lunghezza minima
        # per evitare errori di allineamento
        if len(df_translated) != len(df):
            print("Attenzione: lunghezze diverse tra file grezzo e tradotto. Allineo i dati.")
            min_len = min(len(df), len(df_translated))
            df = df.iloc[:min_len]
            df_translated = df_translated.iloc[:min_len]

        df['text_it'] = df_translated['body']
        df['title_it'] = df_translated['subject'].fillna("Ticket Generico")

    else:
        # ----------------------------------------
        # TRADUZIONE DA ZERO
        # ----------------------------------------
        print("File tradotto non trovato. Avvio la traduzione da zero (potrebbe volerci un po')...")

        # Do una mischiata ai dati (shuffle) per evitare bias ordinati
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Tolgo gli a capo che possono dare problemi alla traduzione
        df[text_col] = df[text_col].astype(str).str.replace(r'\n', ' ', regex=True)

        # Traduco il corpo dei ticket
        df['text_it'] = process_batch(df[text_col].tolist())

        # Traduco anche gli oggetti (subject) se esistono
        subject_col = 'subject' if 'subject' in df.columns else None
        if subject_col:
            print("Traduco anche gli oggetti dei ticket...")
            df['title_it'] = process_batch(df[subject_col].fillna("").astype(str).tolist())
        else:
            df['title_it'] = "Ticket Generico"

    # ----------------------------------------
    # ASSEMBLAGGIO DATASET FINALE
    # ----------------------------------------
    df['id'] = range(1, len(df) + 1)

    # Creo il dataframe finale pulito con le colonne standardizzate
    final_df = pd.DataFrame({
        'id': df['id'],
        'title': df['title_it'],
        'body': df[text_col],      # Tengo l'inglese per riferimento
        'text': df['text_it'],      # L'italiano che userò per il training
        'category': df['category_mapped'],
        'priority': df['priority_mapped']
    })

    # ----------------------------------------
    # PULIZIA DATI
    # ----------------------------------------
    # Tolgo i ticket vuoti o con solo due lettere (non significativi)
    initial_len = len(final_df)
    final_df = final_df[final_df['text'].str.len() > 5]
    if len(final_df) < initial_len:
         print(f"Pulizia: scartati {initial_len - len(final_df)} ticket troppo corti.")

    # ----------------------------------------
    # SALVATAGGIO
    # ----------------------------------------
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("-" * 30)
    print(f"Finito! Il dataset preparato è stato salvato in: {OUTPUT_FILE}")
    print(f"Totale ticket validi per il training: {len(final_df)}")

if __name__ == "__main__":
    main()
