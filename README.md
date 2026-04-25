<div align="center">

<img src="img/png/logo.png" alt="AutoTriage NLP Logo" width="200"/>

# AutoTriage NLP

**Sistema intelligente di classificazione e prioritarizzazione ticket per l'assistenza aziendale**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-0?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/sklearn-1.3-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

`⭐ Aggiungi una stella su GitHub per supportare il mio progetto universitario!`

**Full Stack Architecture - Next.js + FastAPI + Design Glassmorphism**

</div>

---

## 📌 Panoramica del Progetto

Questo repository contiene il codice sorgente del mio Project Work per il corso di Informatica per le Aziende Digitali (L-31). Il sistema, chiamato AutoTriage NLP, utilizza il Machine Learning per analizzare il testo dei ticket di supporto clienti e smistarli automaticamente.

Un'architettura **Full Stack moderna** che combina:

* **Modello Unificato**: Un singolo algoritmo (`MultiOutputClassifier`) stima contemporaneamente la **Categoria di destinazione** (Hardware, Software, Reti, Accesso, Altro) e la **Priorità del ticket** (Alta, Media, Bassa).
* **Logica Ibrida per le Priorità**: Per evitare che l'algoritmo sottostimi ticket critici, le predizioni statistiche sono affiancate da regole fisse basate su keyword (es. forzatura a priorità "Alta" se il testo contiene parole come "virus" o "blocco").
* **Interpretabilità (LIME)**: L'integrazione della libreria LIME permette di visualizzare a schermo quali parole esatte hanno spinto il modello a prendere una determinata decisione, rendendolo trasparente per l'operatore.
* **Addestramento su Dati Reali**: Il modello è addestrato su un dataset pubblico di Kaggle contenente oltre 20.000 ticket reali, massivamente tradotti in italiano.
* **Design Glassmorphism**: Interfaccia moderna con effetti vetro, animazioni fluide e dark theme ispirato al portfolio professionale.

---

## 🚀 Guida Rapida - Avvio

### Opzione 1: Docker (Consigliato - Più Semplice)

**Prerequisiti:**
- Docker e Docker Compose installati

```bash
# Avvia entrambi i servizi (frontend + backend)
docker-compose up --build
```

**Accesso:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/api/v1/docs

---

### Opzione 2: Manuale (Development)

**Prerequisiti:**
- Python 3.10+
- Node.js 20+

#### Passo 1: Backend (FastAPI)

Apri il terminale nella cartella `backend/`:

```bash
# Crea ambiente virtuale
python3 -m venv venv

# Attiva ambiente
# Su Mac/Linux:
source venv/bin/activate
# Su Windows:
venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt

# Copia configura ambiente
cp .env.example .env

# Avvia backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend sarà su: **http://localhost:8000**

#### Passo 2: Frontend (Next.js)

Apri un **NUOVO** terminale nella cartella `frontend/`:

```bash
# Installa dipendenze
npm install

# Copia configura ambiente
cp .env.local.example .env.local

# Avvia frontend
npm run dev
```

Frontend sarà su: **http://localhost:3000**

---

## 🐛 Troubleshooting

### Backend non parte?
```bash
# Verifica che il modello esista
ls -la ../models/unified_model.pkl

# Se non esiste, devi prima addestrare il modello:
cd ..
python src/train_unified_model.py
```

### Frontend da errori di connessione?
```bash
# Verifica che .env.local contenga:
NEXT_PUBLIC_API_URL=http://localhost:8000

# Verifica che il backend sia in esecuzione
curl http://localhost:8000/api/v1/health
```

### Porte già in uso?
```bash
# Cambia porte backend: modifica backend/.env
# Cambia porte frontend: modifica frontend/package.json scripts
```

### Docker non funziona?
```bash
# Pulisci cache Docker
docker system prune -a

# Rebuild senza cache
docker-compose build --no-cache
```

---

## 📊 Verifica Funzionamento

1. **Health Check Backend:**
   ```bash
   curl http://localhost:8000/api/v1/health
   ```
   Dovrebbe restituire: `{"status": "healthy", "model_loaded": true}`

2. **Test Classificazione:**
   Visita http://localhost:3000/manual e prova a inserire un ticket

3. **Test API Docs:**
   Visita http://localhost:8000/api/v1/docs

---

## 🛑 Spegnere i Servizi

### Docker:
```bash
docker-compose down
```

### Manuale:
- Chiudi i terminali o premi Ctrl+C in ognuno

---

## 📝 Note

- La prima volta che avvii, l'installazione delle dipendenze potrebbe richiedere qualche minuto
- Assicurati di avere il file `models/unified_model.pkl` prima di avviare il backend
- Per sviluppo, usa `npm run dev` (hot reload attivo)
- Per produzione, usa `npm run build && npm start`

---

## 🆘 Hai problemi?

Controlla:
1. ✅ Python 3.10+ installato?
2. ✅ Node.js 20+ installato?
3. ✅ Modello ML presente in `models/`?
4. ✅ Porte 3000 e 8000 libere?
5. ✅ Environment variables configurate?

---

## 📂 Architettura del Repository

```plaintext
/
├── frontend/           # Next.js 15 Application
│   ├── src/
│   │   ├── app/       # App Router pages
│   │   │   ├── page.tsx           # Dashboard home
│   │   │   ├── manual/            # Inserimento manuale
│   │   │   ├── batch/             # Importazione CSV
│   │   │   ├── metrics/           # Metriche modello
│   │   │   ├── comparison/        # Confronto dati
│   │   │   └── layout.tsx         # Root layout con glassmorphism
│   │   ├── components/ # React components
│   │   │   ├── ui/                # shadcn/ui components
│   │   │   ├── GlassCard.tsx      # Glassmorphism card
│   │   │   ├── PriorityBadge.tsx  # Priority badges
│   │   │   ├── Navigation.tsx     # Back button
│   │   │   └── MetricsChart.tsx   # Chart.js visualizations
│   │   ├── lib/       # API client & utilities
│   │   │   ├── api.ts             # Axios client
│   │   │   └── constants.ts       # App constants
│   │   └── contexts/  # Context API providers
│   ├── package.json
│   └── Dockerfile
│
├── backend/            # FastAPI Application
│   ├── app/
│   │   ├── api/       # API routes
│   │   │   └── routes/
│   │   │       ├── ticket.py        # Classification endpoints
│   │   │       ├── metrics.py       # Metrics endpoints
│   │   │       └── health.py        # Health check
│   │   ├── models/    # ML models wrapper
│   │   │   └── unified_model.py      # Model manager
│   │   ├── services/  # Business logic
│   │   │   ├── ticket_service.py    # Hybrid priority logic
│   │   │   └── lime_service.py      # LIME explanations
│   │   └── core/      # Configuration
│   │       └── config.py             # Settings management
│   ├── requirements.txt
│   └── Dockerfile
│
├── src/               # Codice Python per training e testing
│   ├── unified_model.py         # Classe principale Modello ML
│   ├── train_unified_model.py    # Script di addestramento
│   ├── prepare_data.py          # Pipeline di pulizia dati
│   ├── translate_data.py         # Script traduzione massiva
│   ├── generate_synthetic_data.py # Generatore dati fittizi
│   └── compare_models.py         # Test comparativo modelli
│
├── img/               # Risorse statiche (Immagini)
├── data/              # Dataset (Kaggle grezzo, Tradotto, Sintetico)
├── models/            # Modelli ML serializzati (.pkl)
└── docker-compose.yml # Container orchestration
```

---

## 📡 API Endpoints

### Classificazione
- `POST /api/v1/ticket/classify` - Classifica singolo ticket
  - Request: `{ "text": "string" }`
  - Response: `{ category, priority, confidence, probabilities, lime_explanation }`

- `POST /api/v1/ticket/batch` - Classifica gruppi di ticket
  - Request: `{ "texts": ["string", ...] }`
  - Response: `{ results: [], summary: {} }`

- `POST /api/v1/ticket/upload-csv` - Upload CSV per classificazione
  - Request: `multipart/form-data` con file
  - Response: `{ results: [], summary: {} }`

### Metriche
- `GET /api/v1/metrics/summary` - Metriche del modello
  - Response: `{ category: {accuracy, precision, recall, f1}, priority: {...} }`

- `GET /api/v1/metrics/images` - Path immagini metriche
  - Response: `{ confusion_matrix_category: "path", confusion_matrix_priority: "path" }`

### Health Check
- `GET /api/v1/health` - Stato API e modello
  - Response: `{ status: "ok", model_loaded: true }`

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

---

## 🎨 Stack Tecnologico

### Frontend
- **Next.js 15** - React Framework con App Router
- **React 18** - UI library con hooks
- **TypeScript** - Type safety
- **TailwindCSS** - Styling utility-first
- **Chart.js** - Visualizzazioni interattive
- **Framer Motion** - Animazioni fluide
- **Axios** - API client
- **Context API** - State management nativo
- **Lucide React** - Icon library

### Backend
- **FastAPI** - Modern API framework
- **Python 3.10** - Linguaggio principale
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **Scikit-Learn** - Machine Learning
- **LIME** - Explainable AI
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

---

## 📋 Caratteristiche Tecniche

### Modello Unificato
- **MultiOutputClassifier** per classificazione simultanea
- 5 categorie: Hardware, Software, Reti, Accesso, Altro
- 3 livelli di priorità: Alta, Media, Bassa
- Logica ibrida ML + keyword per priorità accurate

### Sistema Ibrido di Priorità
- **Trigger Keywords**: forzano priorità Alta
  - "virus", "hacker", "bloccato", "fermo", "critico", etc.
- **Dampening Keywords**: riducono la priorità
  - "non urgente", "con calma", "informazione", etc.
- **Happy Keywords**: indicano problemi risolti
  - "perfetto", "risolto", "complimenti", etc.

### Explainable AI (LIME)
- Visualizzazione parole chiave
- Colore rosso per trigger (aumentano priorità)
- Colore blu per supporto (riducono priorità)
- Trasparenza nelle decisioni del modello

---

## 🔬 Training del Modello

Per addestrare il modello da zero:

### Passo 1: Preparazione del Dataset

```bash
python src/prepare_data.py
```
Questo script pulisce i dati grezzi, mappa le categorie e prepara il CSV in italiano.

### Passo 2: Addestramento

```bash
python src/train_unified_model.py
```
Addestra il modello, effettua split 80/20, salva metriche e modello `.pkl`.

### Passo 3: Test Comparativo

```bash
python src/generate_synthetic_data.py
python src/compare_models.py
```
Dimostra che dati sintetici portano a overfitting rispetto a dati reali.

---

## 🎯 Performance del Modello

Il modello addestrato su dati reali (~20.000 ticket da Kaggle) raggiunge:

### Categoria
- **Accuracy**: ~59%
- **Precision**: ~65%
- **F1-Score**: ~61%

### Priorità
- **Accuracy**: ~46%
- Ottima generalizzazione su linguaggio naturale

---

## ​📄​ License

Vedi [LICENSE](LICENSE) per dettagli.

---

<div align="center">

**Full Stack Architecture** - Next.js + FastAPI

**[Documentazione PDF](../data/docs/template.pdf)** · **[Dataset Kaggle](https://www.kaggle.com/)** · **[API Docs](http://localhost:8000/api/v1/docs)**

**Informatica per le Aziende Digitali (L-31) - Alex Di Paolo**

</div>
