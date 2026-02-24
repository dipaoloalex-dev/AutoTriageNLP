<div align="center">

<img src="assets/img/png/logo.png" alt="AutoTriage NLP Logo" width="200"/>

# AutoTriage NLP

**Sistema intelligente di classificazione e prioritarizzazione ticket per l'assistenza aziendale**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/sklearn-1.3-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

`⭐ Aggiungi una stella su GitHub per supportare il mio progetto universitario!`

</div>

---

## 📌 Panoramica del Progetto

Questo repository contiene il codice sorgente del mio Project Work per il corso di Informatica per le Aziende Digitali (L-31). Il sistema, chiamato AutoTriage NLP, utilizza il Machine Learning per analizzare il testo dei ticket di supporto clienti e smistarli automaticamente. 

Le caratteristiche principali includono:

* **Modello Unificato**: Un singolo algoritmo (`MultiOutputClassifier`) stima contemporaneamente la **Categoria di destinazione** (Amministrazione, Tecnico, Commerciale) e la **Priorità del ticket** (Alta, Media, Bassa).
* **Logica Ibrida per le Priorità**: Per evitare che l'algoritmo sottostimi ticket critici, le predizioni statistiche sono affiancate da regole fisse basate su keyword (es. forzatura a priorità "Alta" se il testo contiene parole come "virus" o "blocco").
* **Interpretabilità (LIME)**: L'integrazione della libreria LIME permette di visualizzare a schermo quali parole esatte hanno spinto il modello a prendere una determinata decisione, rendendolo trasparente per l'operatore.
* **Addestramento su Dati Reali**: Il modello è addestrato su un dataset pubblico di Kaggle contenente oltre 20.000 ticket reali, massivamente tradotti in italiano.

---

## 📂 Architettura del Repository

```plaintext
/
├── assets/                         # Risorse statiche (CSS, Immagini, Documentazione PDF)
├── data/                           # Dataset (Kaggle grezzo, Tradotto, Sintetico)
├── models/                         # Modelli ML serializzati (.pkl) salvati dopo il training
├── src/                            # Codice Sorgente Python
│   ├── app.py                      # Interfaccia Web Streamlit
│   ├── unified_model.py            # Classe principale del Modello ML
│   ├── train_unified_model.py      # Script di addestramento
│   ├── prepare_data.py             # Pipeline di pulizia dati
│   ├── translate_data.py           # Script per la traduzione massiva tramite API
│   ├── generate_synthetic_data.py  # Generatore di dati fittizi
│   └── compare_models.py           # Test comparativo (Dati Reali vs Sintetici)
└── environment.yml                 # Dipendenze per Conda
```

---

## 📦 Installazione e Setup

### Prerequisiti
- **Python 3.10+** installato sul sistema.
- Gestore di pacchetti Conda (consigliato per isolare le dipendenze).

### Comandi per il setup
Apri il terminale e digita:

   ```bash
   # Clona il repository
   git clone https://github.com/dipaoloalex-dev/AutoTriageNLP.git
   cd AutoTriageNLP

   # Crea l'ambiente virtuale leggendo il file environment.yml
   conda env create -f environment.yml

   # Attiva l'ambiente appena creato
   conda activate autotriage
   ```

---

## Riproduzione del progetto
Per far funzionare il sistema da zero, è necessario seguire questi passaggi in ordine logico:

### Passo 1: Preparazione del Dataset

```bash
python src/prepare_data.py
```
Questo script prende i dati grezzi, li pulisce, mappa le categorie e prepara il file CSV finale in italiano pronto per l'addestramento. *(Nota: la traduzione da zero richiede ore, quindi lo script utilizza un file parzialmente pre-tradotto se presente).*

### Passo 2: Addestramento del Modello

```bash
python src/train_unified_model.py
```
Questo script legge il file CSV preparato al passo precedente, effettua lo split dei dati (80/20 per Train e Test), addestra il modello logistico e salva le metriche (Accuracy, F1-Score) e il file binario `.pkl`.

### Passo 3: Esecuzione del Test Comparativo

```bash
python src/generate_synthetic_data.py
python src/compare_models.py
```
Esegui questi due script per riprodurre il test documentato nella relazione, dove viene dimostrato che l'addestramento su dati sintetici porta all'overfitting.

### Passo 4: Avvio dell'Applicazione Web

```bash
python -m streamlit run src/app.py
```
✅ Una volta generato il file `.pkl` del modello, puoi avviare l'interfaccia grafica per testare l'algoritmo in modo interattivo..

---

## ​Utilizzo dell'Interfaccia

L'applicazione web astrae la complessità del codice e permette di testare il modello in due modi:

### Inserimento Manuale (Tab 1)
Puoi incollare il testo di un ticket qualsiasi. L'app ti restituirà la Categoria consigliata, la Priorità e il grafico LIME con le keyword rilevanti.

### Importazione Batch (Tab 2)
Puoi caricare un intero file CSV contenente decine di ticket. Il sistema elaborerà tutte le righe simulando l'attività di smistamento automatico di un Help Desk. Le tab successive mostrano i risultati e i grafici dei test.

---

## ​📄​ License

Vedi [LICENSE](LICENSE) per dettagli.

---

<div align="center">
   
**[Documentazione PDF](assets/doc/template.pdf)** · **[Dataset Kaggle](https://www.kaggle.com/)**
   
</div>
