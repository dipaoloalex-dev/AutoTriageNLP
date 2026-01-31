<div align="center">

<img src="assets/img/png/logo.png" alt="AutoTriage NLP Logo" width="200"/>

# AutoTriage NLP

**Sistema intelligente di classificazione e prioritarizzazione ticket per l'assistenza aziendale**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/sklearn-1.3-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-green)](LICENSE)

`⭐ Aggiungici una stella su GitHub per supportare il progetto!`

</div>

---

## Caratteristiche Principali

* **Unified Model Architecture**: Un singolo modello (`MultiOutputClassifier`) predice simultaneamente **Reparto** (Tecnico, Commerciale) e **Priorità** (Alta, Media, Bassa).
* **Risk-Averse Logic**: Un layer di sicurezza deterministico (keyword critiche) sovrascrive il modello ML per garantire che le urgenze (es. "Hacker", "Server Down") non vengano mai ignorate.
* **Explainable AI (LIME)**: Ogni predizione è giustificata visivamente. Il sistema mostra *quali parole* hanno determinato la scelta.
* **Real-World Data**: Pipeline ETL avanzata che traduce e adatta dataset reali (Kaggle) invece di usare dati sintetici.

---

## Architettura del Progetto

```plaintext
/
├── assets/                  # Risorse statiche (CSS, Img)
├── data/                    # Dataset (Raw, Translated, Augmented)
├── models/                  # Modelli ML serializzati (.pkl)
├── src/                     # Codice Sorgente Python
│   ├── app.py               # [GUI] Streamlit Dashboard
│   ├── unified_model.py     # [CORE] Classe Modello ML
│   ├── train_unified_model.py # [ML] Training Pipeline
│   ├── compare_models.py    # [TEST] Scientific Experiment
│   └── ...                  # Data Pipelines
└── environment.yml          # Dipendenze Conda
```

---

## 📦 Installazione e Configurazione

### Prerequisiti
- **Python 3.10+** installato sul sistema.
- Un ambiente virtuale (consigliato).

### Setup Rapido

   ```bash
   # Clona il repository
   git clone [https://github.com/dipaoloalex-dev/AutoTriageNLP.git](https://github.com/dipaoloalex-dev/AutoTriageNLP.git)
   cd AutoTriageNLP

   # Crea l'ambiente (include Python 3.10, Pandas, Sklearn, Streamlit)
   conda env create -f environment.yml

   # Attiva l'ambiente
   conda activate autotriage
   ```

---

## Guida all'Esecuzione

### Passo 1: Data Preparation (ETL)

```bash
python src/prepare_data.py
# Output: data/tickets_it_augmented.csv
```
Ingestione, normalizzazione e pulizia del dataset Kaggle tradotto. *(Nota: Se volessi rigenerare le traduzioni da zero, puoi lanciare `python src/translate_data.py`, ma richiede diverse ore).*

### Passo 2: Model Training

```bash
python src/train_unified_model.py
# Output: models/unified_model.pkl + assets/img/png/*.
```
Addestramento del `UnifiedModel`, validazione e salvataggio artefatti.

### Passo 3: Scientific Validation

```bash
python src/generate_synthetic_data.py

python src/compare_models.py
# Output: assets/img/comparison/*.
```
Esecuzione dell'esperimento A/B: confronto tra modello addestrato su dati Reali vs Sintetici.

### Passo 4: Launch Dashboard

```bash
python -m streamlit run src/app.py
```
✅ Avvia l'interfaccia web per l'utente finale.

---

## Funzionalità della Dashboard

L'interfaccia Streamlit è suddivisa in tab funzionali:

### Tab 1: Analisi Live
Input manuale per test rapidi. Include:
- **Classificazione:** Reparto e Priorità.
- **XAI Chart:** Grafico LIME che spiega perché (es. "La parola 'Fattura' pesa 0.4 verso Amministrazione").

### Tab 2: Batch Processing
Caricamento file CSV massivi.
- **Input:** File CSV con colonna `text` .
- **Output:** Tabella interattiva scaricabile con predizioni e score di confidenza.
- **File Test:** Usa `assets/csv/ticket.csv`.

### Tab 3 & 4: Metriche & Confronto
Visualizzazione scientifica delle performance:
- **Confusion Matrix:** Dove sbaglia il modello?
- **Real vs Synthetic:** Dimostrazione grafica della superiorità dei dati reali.

---

## License

Academic Public License - Vedi [LICENSE](LICENSE) per dettagli.

Software rilasciato per scopi didattici e di ricerca (Project Work L-31).

---

<div align="center">
   
**[Documentazione PDF](assets/doc/template.pdf)** · **[Dataset Kaggle](https://www.kaggle.com/)**
   
</div>