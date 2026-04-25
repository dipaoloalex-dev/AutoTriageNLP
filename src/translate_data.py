"""
Script di Traduzione Dati (Batch)
=================================

Script ausiliario che uso per tradurre il dataset di Kaggle riga per riga.
Rispetto a `process_batch` in `prepare_data.py`, questo script è pensato
per girare in background con un ritardo (sleep) tra una chiamata e l'altra
per evitare che Google mi blocchi l'IP per troppe richieste (Rate limiting).
Ha anche una logica di salvataggio incrementale: se si blocca a metà, riparte da dove ha lasciato.
"""

# ========================================
# IMPORT LIBRARY
# ========================================
import pandas as pd
from deep_translator import GoogleTranslator
# time per gestire il delay tra le chiamate API
import time
import os
# argparse per gestire i parametri da riga di comando (es: --test)
import argparse
import sys
from typing import List

# ========================================
# CONFIGURAZIONE PATH E PARAMETRI
# ========================================
# File di input: dataset grezzo da Kaggle
INPUT_FILE: str = os.path.join('data', 'kaggle_tickets.csv')
# File di output: dataset tradotto (cache intermedia)
OUTPUT_FILE: str = os.path.join('data', 'kaggle_tickets_it.csv')
# Colonne da tradurre
COLUMNS_TO_TRANSLATE: List[str] = ['subject', 'body', 'answer']
# Delay tra una chiamata e l'altra (per evitare rate limiting)
DELAY_SECONDS: float = 0.2

# ========================================
# FUNZIONI DI TRADUZIONE
# ========================================
def translate_text(text: str, translator: GoogleTranslator) -> str:
    """
    Traduce il testo in italiano, gestendo le eccezioni se l'API va in timeout.

    Args:
        text: Testo da tradurre
        translator: Istanza di GoogleTranslator configurata

    Returns:
        Testo tradotto o originale in caso di errore
    """
    # Controllo validità input
    if not isinstance(text, str) or not text.strip():
        return text

    try:
        # Google Translate accetta al massimo 5000 caratteri
        # Taglio a 4900 per sicurezza (margin)
        if len(text) > 4900:
            text = text[:4900]
        return translator.translate(text)

    except Exception as e:
        # Se l'API fallisce, lascio il testo originale e continuo
        print(f"\nErrore API su una riga (lascio originale): {e}")
        return text

# ========================================
# FUNZIONE PRINCIPALE
# ========================================
def main() -> None:
    """Esegue la traduzione progressiva del dataset."""
    # ----------------------------------------
    # PARSING ARGOMENTI DA RIGA DI COMANDO
    # ----------------------------------------
    parser = argparse.ArgumentParser(description='Traduttore progressivo per dataset Kaggle')
    # --test: esegue solo le prime 5 righe per verificare che funzioni
    parser.add_argument('--test', action='store_true', help='Esegue solo le prime 5 righe per vedere se funziona')
    args = parser.parse_args()

    print("=== AVVIO SCRIPT TRADUZIONE ===")

    # ----------------------------------------
    # VERIFICA FILE DI INPUT
    # ----------------------------------------
    if not os.path.exists(INPUT_FILE):
        print(f"File non trovato: {INPUT_FILE}")
        print("Devi scaricare il file da Kaggle e metterlo in data/")
        sys.exit(1)

    df_source = pd.read_csv(INPUT_FILE)
    total_rows = len(df_source)
    output_path = OUTPUT_FILE

    # ----------------------------------------
    # GESTIONE MODALITÀ TEST
    # ----------------------------------------
    if args.test:
        total_rows = 5
        print("MODALITÀ TEST: Traduco solo 5 righe.")
        output_path = OUTPUT_FILE.replace('.csv', '_test.csv')

    print(f"Da elaborare: {total_rows} righe")

    # ========================================
    # LOGICA DI RIPRESA (RESUME)
    # ========================================
    # Se lo script è stato interrotto, riprende da dove aveva lasciato
    processed_rows = 0

    if not args.test and os.path.exists(output_path):
        try:
            # Conto quante righe sono già state tradotte
            with open(output_path, 'r', encoding='utf-8') as f:
                # Tolgo 1 per l'header
                processed_rows = sum(1 for _ in f) - 1

            if processed_rows < 0:
                processed_rows = 0

            if processed_rows > 0:
                print(f"Trovato file parziale: riprendo dalla riga {processed_rows + 1}...")

        except Exception as e:
            print(f"Errore lettura file parziale: {e}. Ricomincio da zero.")
            processed_rows = 0

    # ----------------------------------------
    # CREAZIONE FILE DI OUTPUT (se necessario)
    # ----------------------------------------
    # Se parto da zero, creo il file e ci metto le intestazioni delle colonne
    if processed_rows == 0:
        pd.DataFrame(columns=df_source.columns).to_csv(output_path, index=False)

    # Se ho già finito, esco
    if processed_rows >= total_rows:
        print("File già tradotto al 100%. Esco.")
        sys.exit(0)

    # ========================================
    # TRADUZIONE VERA E PROPRIA
    # ========================================
    # Inizializzo il traduttore (auto-detect source language)
    translator = GoogleTranslator(source='auto', target='it')
    start_time = time.time()

    print("-" * 40)

    try:
        # Loop dalle ultime righe processate fino alla fine
        for idx in range(processed_rows, total_rows):
            # Copio la riga per non modificare l'originale
            row = df_source.iloc[idx].copy()

            # Traduco ogni colonna specificata
            for col in COLUMNS_TO_TRANSLATE:
                if col in row and pd.notna(row[col]):
                    row[col] = translate_text(row[col], translator)

            # ----------------------------------------
            # SALVATAGGIO INCREMENTALE
            # ----------------------------------------
            # Salvo riga per riga in append (mode='a')
            # Lento ma sicuro: se crasha non perdo i dati già tradotti
            pd.DataFrame([row]).to_csv(output_path, mode='a', header=False, index=False)

            # ----------------------------------------
            # PROGRESS BAR E STATISTICHE
            # ----------------------------------------
            # Stampa di debug ogni 10 righe per non intasare la console
            if (idx + 1) % 10 == 0 or idx == 0:
                elapsed = time.time() - start_time
                # Calcolo velocità (righe al secondo)
                speed = (idx - processed_rows + 1) / elapsed if elapsed > 0 else 0
                # Calcolo tempo rimanente stimato
                remaining_rows = total_rows - idx - 1
                remaining_min = (remaining_rows / speed) / 60 if speed > 0 else 0

                # Stampa progress bar (uso \r per sovrascrivere la stessa riga)
                print(f"Tradotte: {idx + 1}/{total_rows} | "
                      f"Velocità: {speed:.1f} r/s | "
                      f"Tempo stimato: {remaining_min:.1f} min", end='\r', flush=True)

            # ----------------------------------------
            # DELAY PER EVITARE RATE LIMITING
            # ----------------------------------------
            # Pausa tra una chiamata e l'altra per non farmi bloccare da Google
            time.sleep(DELAY_SECONDS)

    # Gestione interruzione manuale (CTRL+C)
    except KeyboardInterrupt:
        print("\n\nInterrotto da tastiera (CTRL+C). I dati tradotti finora sono salvi.")
        sys.exit(0)

    # Gestione altri errori
    except Exception as e:
        print(f"\n\nErrore critico durante il loop: {e}")
        sys.exit(1)

    # ========================================
    # COMPLETAMENTO
    # ========================================
    print("\n\n" + "="*40)
    print("Traduzione completata!")
    print(f"File salvato in: {output_path}")

if __name__ == "__main__":
    main()
