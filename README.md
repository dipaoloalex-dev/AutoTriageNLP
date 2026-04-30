<div align="center">

# AutoTriage NLP

**Sistema intelligente di classificazione e prioritarizzazione ticket per l'assistenza aziendale**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-0?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/sklearn-1.3-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

`⭐ Aggiungi una stella su GitHub per supportare il mio progetto universitario!`

</div>

---

## 📌 Panoramica del Progetto

Questo repository contiene il codice sorgente del mio Project Work per il corso di Informatica per le Aziende Digitali (L-31). Il sistema, chiamato AutoTriage NLP, utilizza il Machine Learning per analizzare il testo dei ticket di supporto clienti e smistarli automaticamente.

Un'architettura **Full Stack** moderna che combina:

* **Modello Unificato**: Un singolo algoritmo (`MultiOutputClassifier`) stima contemporaneamente la **Categoria di destinazione** (Amministrativa, Commerciale, Tecnica) e la **Priorità del ticket** (Alta, Media, Bassa).
* **Logica Ibrida per le Priorità**: Per evitare che l'algoritmo sottostimi ticket critici, le predizioni statistiche sono affiancate da regole fisse basate su keyword (es. forzatura a priorità "Alta" se il testo contiene parole come "virus" o "blocco").
* **Interpretabilità (LIME)**: L'integrazione della libreria LIME permette di visualizzare a schermo quali parole esatte hanno spinto il modello a prendere una determinata decisione, rendendolo trasparente per l'operatore.
* **Addestramento su Dati Reali**: Il modello è addestrato su un dataset pubblico di Kaggle contenente oltre 20.000 ticket reali, massivamente tradotti in italiano.

---

## 🚀 Guida Rapida - Avvio

### Opzione 1: Docker (Consigliato - Nessuna Installazione)

Con Docker non devi installare Python, Node.js o alcuna libreria sul tuo PC. Tutto gira dentro container isolati.

**Prerequisiti:**
- Docker Desktop installato ([Download](https://www.docker.com/products/docker-desktop/))
- Docker Compose (incluso in Docker Desktop)

---

#### Passo 1: Generazione Modello ML

Il backend necessita del file `models/unified_model.pkl` per funzionare. Puoi generarlo direttamente con Docker:

```bash
# Costruisci l'immagine di training.
docker-compose --profile training build training

# Esegui il training. Questo scarica le dipendenze, prepara i dati e addestra il modello.
docker-compose --profile training run --rm training
```

**Cosa succede:**
- Il container scarica pandas, scikit-learn e le altre librerie
- Legge i dataset dalla cartella `data/`
- Genera il file `models/unified_model.pkl` che resta sul tuo PC
- Il container viene automaticamente rimosso dopo il completamento

---

#### Passo 2: Avvio dell'Applicazione

```bash
# Avvia frontend e backend
docker-compose up --build
```

**Accesso:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/api/v1/docs

> **💡 Nota:** Una volta generato il modello, puoi avviare l'applicazione direttamente con `docker-compose up` (senza `--build`) per essere più veloce. Usa `--build` solo se modifichi il codice.

---

#### Passo 3: Spegnimento

```bash
# Ferma e rimuovi i container
docker-compose down
```

---

### Opzione 2: Manuale (Sviluppo Locale)

**Prerequisiti:**
- Python 3.10+
- Node.js 20+

#### Passo 1: Addestramento Modello ML

```bash
# Installa le dipendenze Python per il training
pip install pandas scikit-learn numpy matplotlib seaborn joblib deep_translator tqdm

# Prepara il dataset tradotto
python src/prepare_data.py

# Addestra il modello
python src/train_unified_model.py
```

Questo creerà il file `models/unified_model.pkl` necessario per il backend.

---

#### Passo 1: Backend (FastAPI)

```bash
# 1. Entra nella cartella backend
cd backend

# 2. Crea ambiente virtuale
python3 -m venv venv

# 3. Attiva ambiente
# Su Mac/Linux:
source venv/bin/activate
# Su Windows:
venv\Scripts\activate

# 4. Aggiorna pip
python -m pip install --upgrade pip

# 5. Installa dipendenze con timeout esteso (per connessioni lente)
pip install -r requirements.txt --timeout 300 --retries 5

# 6. Copia configurazione ambiente
cp .env.example .env

# 7. Avvia backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Passo 2: Frontend (Next.js)

Apri un nuovo terminale (lascia il backend attivo nell'altro):

```bash
# 1. Entra nella cartella frontend
cd frontend

# 2. Installa dipendenze
npm install

# 3. Copia configurazione ambiente
cp .env.local.example .env.local

# 4. Avvia frontend
npm run dev
```

**Accesso:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/api/v1/docs

---

## 📂 Architettura del Repository

```plaintext
/
├── frontend/          
│   ├── src/
│   │   ├── app/      
│   │   │   ├── page.tsx          
│   │   │   ├── manual/           
│   │   │   ├── batch/             
│   │   │   ├── metrics/           
│   │   │   ├── comparison/       
│   │   │   └── layout.tsx         
│   │   ├── components/
│   │   │   ├── ui/              
│   │   │   ├── GlassCard.tsx     
│   │   │   ├── PriorityBadge.tsx 
│   │   │   ├── Navigation.tsx   
│   │   │   └── MetricsChart.tsx  
│   │   ├── lib/     
│   │   │   ├── api.ts           
│   │   │   └── constants.ts     
│   │   └── contexts/ 
│   ├── package.json
│   └── Dockerfile
│
├── backend/           
│   ├── app/
│   │   ├── api/       
│   │   │   └── routes/
│   │   │       ├── ticket.py       
│   │   │       ├── metrics.py       
│   │   │       └── health.py        
│   │   ├── models/   
│   │   │   └── unified_model.py     
│   │   ├── services/ 
│   │   │   ├── ticket_service.py  
│   │   │   └── lime_service.py     
│   │   └── core/     
│   │       └── config.py           
│   ├── requirements.txt
│   └── Dockerfile
│
├── src/              
│   ├── unified_model.py         
│   ├── train_unified_model.py   
│   ├── prepare_data.py        
│   ├── translate_data.py        
│   ├── generate_synthetic_data.py
│   └── compare_models.py       
│
├── img/           
├── data/          
├── models/         
└── docker-compose.yml
```

---

## ​Utilizzo dell'Interfaccia

L'applicazione web con design glassmorphism offre:

1. **Dashboard** (`/`) - Overview con link alle sezioni e Stack Tecnologico
2. **Inserimento Manuale** (`/manual`) - Analisi singola ticket con spiegazioni LIME
   - Inserimento testo ticket
   - Classificazione in tempo reale
   - Visualizzazione parole chiave influenti
   - Esempi di ticket pronti all'uso

3. **Importazione CSV** (`/batch`) - Upload e analisi di gruppi di ticket
   - Drag & drop del file CSV
   - Supporta colonne: text, body, testo, descrizione
   - Risultati dettagliati con summary

4. **Metriche** (`/metrics`) - Performance del modello con grafici interattivi
   - KPI cards (Accuracy, Precision, F1-Score)
   - Grafici Chart.js
   - Confusion Matrix per categoria e priorità
   - Spiegazioni delle metriche

5. **Confronto** (`/comparison`) - Test comparativo dati sintetici vs reali
   - Metodologia del test
   - Risultati test inversion
   - Conclusioni e deduzioni

## ​📄​ License

Vedi [LICENSE](LICENSE) per dettagli.

---

<div align="center">

**[Dataset Kaggle](https://www.kaggle.com/)** 

**Informatica per le Aziende Digitali (L-31) - Alex Di Paolo**

</div>
