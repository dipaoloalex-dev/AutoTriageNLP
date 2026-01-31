# 🤖 AutoTriage NLP

Questo repository contiene il codice sorgente completo per **AutoTriage NLP**, un prototipo software modulare progettato per automatizzare il processo di smistamento (*triage*) dei ticket di assistenza aziendale.

Il sistema sfrutta tecniche di **Natural Language Processing (NLP)** e **Machine Learning (ML)** per analizzare il contenuto semantico delle richieste, classificandole automaticamente per reparto di competenza e stimandone la priorità operativa con un approccio ibrido (statistico e *rule-based*).

## 🌟 Funzionalità Chiave

1. **Data Engineering & Real-World Simulation:**
   - **Pipeline di Preparazione Dati:** Modulo avanzato (`prepare_data.py`) che non si limita a dati sintetici, ma ingerisce dataset reali di customer care (es. Kaggle).
   - **Traduzione Automatica Neurale:** Integrazione della libreria `deep-translator` con parallelizzazione (`ThreadPoolExecutor`) per convertire massivamente dataset dall'inglese all'italiano, simulando una morfologia linguistica complessa e realistica.
   - **Mappatura Intelligente:** Algoritmi di normalizzazione che convertono le etichette originali (es. "Refund", "Hardware") nelle macro-categorie target (Amministrazione, Tecnico, Commerciale).

2. **Pipeline di Machine Learning (Scikit-Learn):**
   - **Preprocessing:** Pulizia automatica del testo (conversione in minuscolo, rimozione punteggiatura e *stopwords* italiane estese).
   - **Vettorizzazione:** Trasformazione del testo in matrici numeriche sparse tramite algoritmi **TF-IDF** con supporto n-grammi (unigrammi e bigrammi) per catturare il contesto locale.
   - **Classificazione Multi-Task Probabilistica:** Utilizzo di un'architettura **UnifiedModel** basata su **Logistic Regression** incapsulata in un *MultiOutputClassifier*. Questa scelta permette di ottenere non solo la classe predetta, ma anche le **probabilità di confidenza** (`predict_proba`), fondamentali per la gestione delle soglie di rischio.

3. **Logica di Priorità "Risk-Averse" (Ibrida):**
   - Il sistema combina le probabilità del modello ML con un motore di regole basato su keyword critiche (es. "fermo", "hacker", "scadenza").
   - Questo approccio garantisce che ticket ad alto rischio non vengano mai sottovalutati, forzando un'escalation ("Alta Priorità") anche se il modello statistico è incerto.

4. **Explainable AI (LIME Integration):**
   - Integrazione di **LIME (Local Interpretable Model-agnostic Explanations)**.
   - A differenza della semplice analisi dei pesi globali, il sistema spiega ogni *singola* predizione perturbando il testo e identificando le parole specifiche che hanno determinato l'output per quel preciso ticket.

5. **Interfaccia Utente Interattiva (Streamlit):**
   - Dashboard web reattiva e *user-friendly* (`app.py`) con design personalizzato (CSS *Dark Blue/Inter font*).
   - **Analisi Live:** Inserimento manuale di ticket per una classificazione istantanea.
   - **Batch Processing:** Caricamento di file CSV massivi per l'analisi automatica di interi lotti.
   - **Reportistica:** Visualizzazione grafica delle metriche di performance (Confusion Matrix, Accuracy, F1-Score).

## 🏗️ Architettura del Progetto

La struttura del repository è organizzata in moduli logici distinti per garantire manutenibilità e scalabilità:

```plaintext
/
├── assets/                  # Risorse statiche
│   ├── css/                 # Fogli di stile personalizzati (style.css)
│   └── img/                 # Loghi e grafici generati (png/ico)
├── data/                    # Dataset
│   ├── kaggle_tickets.csv      # Dataset originale (EN/DE)
│   ├── kaggle_tickets_it.csv   # Dataset tradotto (IT - Ready-to-use)
│   └── tickets_it_augmented.csv # Dataset finale per training
├── models/                  # Modelli ML serializzati (.pkl)
├── src/                     # Codice Sorgente Python
│   ├── app.py               # Frontend (Streamlit Dashboard & LIME logic)
│   ├── translate_data.py    # [CORE] Script traduzione automatica dataset
│   ├── prepare_data.py      # Pipeline pulizia e preparazione dati training
│   ├── unified_model.py     # Definizione della classe modello
│   └── train_unified_model.py # Script di addestramento e valutazione
├── metrics_report_unified.txt # Report testuale automatico delle performance
└── requirements.txt         # Elenco delle dipendenze Python
```

### Strategia Gestione Dati
Il progetto adotta un approccio ibrido per bilanciare **Efficienza (Developer Experience)** e **Riproducibilità Tecnica**:
1.  **Dataset Pre-Tradotto (`data/kaggle_tickets_it.csv`):** Fornito direttamente nel repository per permettere l'immediato utilizzo e training del modello senza attese (zero-config).
2.  **Pipeline di Traduzione (`src/translate_data.py`):** Viene fornito lo script completo che ha generato il dataset italiano partendo dai dati grezzi. Questo dimostra la completa padronanza della pipeline di Data Engineering e permette di rigenerare i dati da zero se necessario.

### Flusso Operativo (Workflow)
1. **Data Translation (Opzionale):** `translate_data.py` → Legge Raw Data → Chiama API Translator → Genera CSV Italiano. *(Già eseguito per comodità)*.
2. **Data Preparation:** `prepare_data.py` → Legge CSV Italiano → Normalizza → Crea `data/tickets_it_augmented.csv`.
3. **Model Training:** `train_unified_model.py` → Carica CSV → Split Train/Test → Training (MultiOutput Logistic Regression) → Salva `.pkl` in `models/` + Grafici in `assets/`.
4. **Deployment:** `app.py` → Carica `.pkl` → Interfaccia Web per l'utente finale con supporto LIME.

## 📦 Installazione e Configurazione

### Prerequisiti
- **Python 3.10+** installato sul sistema.
- Un ambiente virtuale (consigliato).

### Setup Rapido (Consigliato con Conda)

Per garantire la massima compatibilità e replicabilità dell'ambiente (evitando problemi di stile o librerie mancanti), si consiglia l'uso di **Conda**.

1. **Clona il repository:**
   ```bash
   git clone https://github.com/dipaoloalex-dev/AutoTriageNLP.git
   cd AutoTriageNLP
   ```

2. **Crea l'ambiente dedicato (`autotriage`):**
   Questo comando installerà Python 3.10 e tutte le librerie necessarie (pandas, scikit-learn, streamlit, ecc.) in un ambiente isolato e pulito.
   ```bash
   conda env create -f environment.yml
   ```
   **Nota**: L'opzione `--force` è utile se hai già provato a creare l'ambiente in precedenza; sovrascriverà eventuali installazioni parziali.

3. **Attiva l'ambiente:**
   ```bash
   conda activate autotriage
   ```
   **Nota**: Assicurati di vedere `(autotriage)` nel tuo terminale prima di proseguire. Se usi ancora `(base)`, le dipendenze potrebbero essere errate.

## 🚀 Guida all'Esecuzione

Il sistema è progettato per essere eseguito in sequenza logica.

### Passo 1: Preparazione dei Dati (Data Prep)
Poiché il dataset tradotto è già fornito (`kaggle_tickets_it.csv`), puoi passare direttamente alla preparazione per il training (pulizia e normalizzazione):

```bash
python src/prepare_data.py
```
*Output atteso:* Generazione di `tickets_it_augmented.csv` pronto per l'addestramento.

*(Nota: Se volessi rigenerare le traduzioni da zero, puoi lanciare `python src/translate_data.py`, ma richiede diverse ore).*

### Passo 2: Addestramento del Modello
Avvia la pipeline di Machine Learning. Lo script eseguirà il preprocessing, addestrerà il `UnifiedModel`, calcolerà le metriche di validazione e salverà i file necessari.

```bash
python src/train_unified_model.py
```

*Output atteso:* Report di classificazione a terminale, generazione grafici in `assets/img/png` e salvataggio modello.

### Passo 3: Esperimento di Confronto (Dati Sintetici vs Reali)
**⚠️ PASSO OBBLIGATORIO** - Questo esperimento genera i grafici comparativi visualizzati nel tab "Confronto Dati Sintetici vs Reali" della dashboard.

#### 3.1 Generazione Dataset Sintetico
Crea un dataset di 500 ticket sintetici con regole fisse:
```bash
python src/generate_synthetic_data.py
```

#### 3.2 Esecuzione Esperimento Cross-Validation
Addestra due modelli (uno su dati sintetici, uno su dati reali) e valuta le performance incrociate:
```bash
python src/compare_models.py
```

*Output atteso:* Report testuale con metriche comparative e 3 grafici salvati in `assets/img/comparison/`.

**Nota**: Senza questo step, il quarto tab della dashboard mostrerà un messaggio informativo ma nessun grafico.

---

### Passo 4: Avvio della Dashboard
Lancia l'applicazione web con Streamlit.

```bash
python -m streamlit run src/app.py
```

- Il browser si aprirà automaticamente all'indirizzo `http://localhost:8501`.
- **Nota Importante:** Usiamo `python -m streamlit` invece di `streamlit` direttamente per garantire che venga utilizzato lo Streamlit installato nel virtual environment e non quello eventualmente presente in Anaconda globale. Questo previene conflitti di versione tra le librerie.
- Se riscontri errori di percorso, assicurati di eseguire il comando dalla root del progetto.

## 🖊️ Funzionalità della Dashboard

### Tab 1: Inserimento Manuale
- **Input:** Area di testo per scrivere o incollare un ticket.
- **Azione:** Cliccando su "Analizza", il modello elabora il testo in tempo reale.
- **Output:**
  - **Reparto Consigliato:** Categoria predetta (es. Tecnico).
  - **Livello d'Urgenza:** Priorità stimata (es. Alta - colorato semanticamente).
  - **Explainable AI:** Grafico a barre con le parole chiave che hanno influenzato la decisione (es. "database", "errore" -> Tecnico).

### Tab 2: Importazione da File (Batch Processing)
- **Input:** Caricamento di un file CSV.
- **Formato Richiesto:** Il file deve contenere almeno una colonna `text` con i messaggi dei ticket.
- **Esempio:** Puoi trovare un file di test pronto all'uso in:
  `assets/csv/ticket.csv`
- **Output:** Tabella interattiva con Classificazione, Priorità e Confidenza per ogni riga. Possibilità di scaricare il report finale come nuovo CSV.
