"""
Script di Traduzione Dati (Batch)
=================================

Script ausiliario che uso per tradurre il dataset di Kaggle riga per riga.
Rispetto a `process_batch` in `prepare_data.py`, questo script è pensato
per girare in background con un ritardo (sleep) tra una chiamata e l'altra 
per evitare che Google mi blocchi l'IP per troppe richieste (Rate limiting).
Ha anche una logica di salvataggio incrementale: se si blocca a metà, riparte da dove ha lasciato.
"""

import pandas as pd
from deep_translator import GoogleTranslator
import time
import os
import argparse
import sys
from typing import List

# --- CONFIGURAZIONE PATH E PARAMETRI ---
INPUT_FILE: str = os.path.join('data', 'kaggle_tickets.csv')          
OUTPUT_FILE: str = os.path.join('data', 'kaggle_tickets_it.csv')      
COLUMNS_TO_TRANSLATE: List[str] = ['subject', 'body', 'answer'] 
DELAY_SECONDS: float = 0.2                             

def translate_text(text: str, translator: GoogleTranslator) -> str:
    """
    Traduce il testo in italiano, gestendo le eccezioni se l'API va in timeout.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    try:
        # Google Translate accetta al massimo 5000 caratteri, taglio a 4900 per sicurezza.
        if len(text) > 4900:
            text = text[:4900]
        return translator.translate(text)

    except Exception as e:
        print(f"\nErrore API su una riga (lascio originale): {e}")
        return text

def main() -> None:
    parser = argparse.ArgumentParser(description='Traduttore progressivo per dataset Kaggle')
    parser.add_argument('--test', action='store_true', help='Esegue solo le prime 5 righe per vedere se funziona')
    args = parser.parse_args()

    print("=== AVVIO SCRIPT TRADUZIONE ===")

    if not os.path.exists(INPUT_FILE):
        print(f"File non trovato: {INPUT_FILE}")
        print("Devi scaricare il file da Kaggle e metterlo in data/")
        sys.exit(1)

    df_source = pd.read_csv(INPUT_FILE)
    total_rows = len(df_source)
    output_path = OUTPUT_FILE

    if args.test:
        total_rows = 5
        print("MODALITÀ TEST: Traduco solo 5 righe.")
        output_path = OUTPUT_FILE.replace('.csv', '_test.csv')

    print(f"Da elaborare: {total_rows} righe")

    # --- LOGICA DI RIPRESA (RESUME) ---
    # Guardo se avevo già iniziato a tradurre il file in precedenza
    processed_rows = 0
    
    if not args.test and os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                processed_rows = sum(1 for _ in f) - 1 # Tolgo l'header
            
            if processed_rows < 0: 
                processed_rows = 0
            
            if processed_rows > 0:
                print(f"Trovato file parziale: riprendo dalla riga {processed_rows + 1}...")

        except Exception as e:
            print(f"Errore lettura file parziale: {e}. Ricomincio da zero.")
            processed_rows = 0 

    # Se parto da zero, creo il file e ci metto le intestazioni delle colonne
    if processed_rows == 0:
        pd.DataFrame(columns=df_source.columns).to_csv(output_path, index=False)
    
    if processed_rows >= total_rows:
        print("File già tradotto al 100%. Esco.")
        sys.exit(0)

    # --- TRADUZIONE VERA E PROPRIA ---
    translator = GoogleTranslator(source='auto', target='it')
    start_time = time.time()

    print("-" * 40)

    try:
        for idx in range(processed_rows, total_rows):
            row = df_source.iloc[idx].copy()
            
            for col in COLUMNS_TO_TRANSLATE:
                if col in row and pd.notna(row[col]):
                    row[col] = translate_text(row[col], translator)
            
            # Salvo riga per riga. Lento ma sicuro: se crasha non perdo i dati.
            pd.DataFrame([row]).to_csv(output_path, mode='a', header=False, index=False)
            
            # Stampa di debug ogni 10 righe per non intasare la console
            if (idx + 1) % 10 == 0 or idx == 0:
                elapsed = time.time() - start_time
                speed = (idx - processed_rows + 1) / elapsed if elapsed > 0 else 0
                remaining_rows = total_rows - idx - 1
                remaining_min = (remaining_rows / speed) / 60 if speed > 0 else 0
                
                print(f"Tradotte: {idx + 1}/{total_rows} | "
                      f"Velocità: {speed:.1f} r/s | "
                      f"Tempo stimato: {remaining_min:.1f} min", end='\r', flush=True)
            
            # Pausa per non farmi bloccare da Google
            time.sleep(DELAY_SECONDS)

    except KeyboardInterrupt:
        print("\n\nInterrotto da tastiera (CTRL+C). I dati tradotti finora sono salvi.")
        sys.exit(0)

    except Exception as e:
        print(f"\n\nErrore critico durante il loop: {e}")
        sys.exit(1)

    print("\n\n" + "="*40)
    print("Traduzione completata!")
    print(f"File salvato in: {output_path}")

if __name__ == "__main__":
    main()
