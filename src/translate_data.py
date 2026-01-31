"""
Pipeline di Traduzione Dati
===========================

Questo script gestisce la traduzione batch del dataset dall'inglese all'italiano utilizzando la 
libreria `deep_translator` (wrapper di Google Translate). Include funzionalità come:
- Salvataggio incrementale (capacità di ripresa).
- Limitazione della velocità (Rate limiting) per evitare ban API.
- Monitoraggio del progresso (Velocità, ETA).
"""

import pandas as pd
from deep_translator import GoogleTranslator
import time
import os
import argparse
import sys
from typing import List, Optional

# ==============================================================================
# 1. CONFIGURAZIONE E COSTANTI
# ==============================================================================

INPUT_FILE: str = os.path.join('data', 'kaggle_tickets.csv')          
OUTPUT_FILE: str = os.path.join('data', 'kaggle_tickets_it.csv')      
COLUMNS_TO_TRANSLATE: List[str] = ['subject', 'body', 'answer'] 
DELAY_SECONDS: float = 0.2                             

def translate_text(text: str, translator: GoogleTranslator) -> str:
    """
    Traduce una singola stringa di testo in italiano.

    Gestisce casi limite come stringhe vuote o lunghezza eccessiva troncando se necessario, 
    assicurando che il processo sia robusto contro input malformati.

    Args:
        text (str): Il contenuto testuale originale (es. corpo del ticket in inglese).
        translator (GoogleTranslator): Un'istanza di traduttore inizializzata.

    Returns:
        str: Il testo tradotto in italiano. Ritorna il testo originale se la traduzione fallisce
             o se i controlli di validazione passano dati non testuali.
    """

    # Programmazione difensiva: assicurati che l'input sia una stringa non vuota valida
    if not isinstance(text, str) or not text.strip():
        return text

    try:
        # L'API di Google Translate ha un limite di caratteri (circa 5000).
        # Tronchiamo a 4900 per proteggerci dal rifiuto della richiesta.
        if len(text) > 4900:
            text = text[:4900]

        return translator.translate(text)

    except Exception as e:
        # Log dell'avviso ma non arrestare il contesto di esecuzione
        # Idealmente, qui useremmo un logger invece di print.
        print(f"\n[WARNING] Errore durante la traduzione del testo: {e}")
        return text

def main() -> None:
    """
    Pipeline di Esecuzione Principale.

    Orchestra il caricamento dei dati, il ciclo di traduzione, la logica di ripresa e il salvataggio.
    Supporta un argomento CLI `--test` per eseguire un piccolo batch a scopo di debug.
    """
    parser = argparse.ArgumentParser(description='Pipeline di Traduzione Dataset')
    parser.add_argument('--test', action='store_true', help='Esegui solo sulle prime 5 righe per debug.')
    args = parser.parse_args()

    print(f"=== AVVIO PIPELINE DI TRADUZIONE ===")

    # --- Fase 1: Caricamento Dati ---
    try:
        df_source = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"[ERRORE CRITICO] File di input {INPUT_FILE} non trovato.")
        print("Assicurati che il dataset Kaggle sia posizionato nella cartella 'data/'.")
        sys.exit(1)

    total_rows: int = len(df_source)
    output_path: str = OUTPUT_FILE

    if args.test:
        total_rows = 5
        print("[MODE] TEST MODE ATTIVA: Limitazione elaborazione alle prime 5 righe.")
        output_path = OUTPUT_FILE.replace('.csv', '_test.csv')

    print(f"Input Sorgente:  {INPUT_FILE}")
    print(f"Output Target:   {output_path}")
    print(f"Righe Totali:    {total_rows}")

    # --- Fase 2: Logica di Ripresa ---
    # Controlla se esiste un file parziale per riprendere da dove abbiamo lasciato.
    processed_rows: int = 0
    
    if not args.test and os.path.exists(output_path):
        try:
            # Conta velocemente le righe esistenti per determinare l'indice di partenza
            with open(output_path, 'r') as f:
                processed_rows = sum(1 for _ in f) - 1 # Sottrai intestazione
            
            if processed_rows < 0: 
                processed_rows = 0
            
            if processed_rows > 0:
                print(f"[RESUME] Trovato file parziale con {processed_rows} righe.")
                print(f"         Riprendo la traduzione dalla riga {processed_rows + 1}...")

        except Exception as e:
            print(f"[ERROR] Controllo integrità fallito sul file esistente: {e}. Reset.")
            processed_rows = 0 

    # Se si inizia da zero, scrivi l'intestazione
    if processed_rows == 0:
        pd.DataFrame(columns=df_source.columns).to_csv(output_path, index=False)
    
    # Controlla completamento anticipato
    if processed_rows >= total_rows:
        print("Traduzione già completata al 100%. Nessuna azione richiesta.")
        sys.exit(0)

    # --- Fase 3: Ciclo di Traduzione ---
    translator = GoogleTranslator(source='auto', target='it')
    start_time = time.time()

    print("-" * 60)

    try:
        for idx in range(processed_rows, total_rows):
            # Elabora riga per riga per garantire il salvataggio atomico
            row = df_source.iloc[idx].copy()
            
            for col in COLUMNS_TO_TRANSLATE:
                if col in row and pd.notna(row[col]):
                    row[col] = translate_text(row[col], translator)
            
            # Append Atomico: Scrivi immediatamente su disco
            pd.DataFrame([row]).to_csv(output_path, mode='a', header=False, index=False)
            
            # Monitoraggio Progresso
            if (idx + 1) % 10 == 0 or idx == 0:
                elapsed = time.time() - start_time
                # Calcola velocità di elaborazione (righe al secondo)
                speed = (idx - processed_rows + 1) / elapsed if elapsed > 0 else 0
                remaining_rows = total_rows - idx - 1
                remaining_min = (remaining_rows / speed) / 60 if speed > 0 else 0
                
                print(f"Progresso: {idx + 1}/{total_rows} | "
                      f"Velocità: {speed:.2f} r/s | "
                      f"ETA: {remaining_min:.1f} min", end='\r', flush=True)
            
            # Rate Limiting
            time.sleep(DELAY_SECONDS)

    except KeyboardInterrupt:
        print("\n\n[USER STOP] Esecuzione processo interrotta manualmente. Dati salvati fino all'ultimo record.")
        sys.exit(0)

    except Exception as e:
        print(f"\n\n[CRASH] Errore imprevisto nel ciclo principale: {e}")
        sys.exit(1)

    print(f"\n\n{'='*60}")
    print(f"TRADUZIONE COMPLETATA CON SUCCESSO")
    print(f"File salvato in: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
