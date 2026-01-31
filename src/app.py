"""
AutoTriage NLP - Applicazione Web Streamlit
===========================================

Questa è l'applicazione principale (Frontend) per il progetto di AutoTriage.
Fornisce un'interfaccia utente interattiva per:
1. Analizzare singoli ticket in tempo reale.
2. Caricare file CSV per analisi batch.
3. Visualizzare le metriche di performance del modello.
4. Spiegare le predizioni tramite LIME (Explainable AI).
5. Mostrare i risultati comparativi (Sintetico vs Reale).
"""

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from lime.lime_text import LimeTextExplainer
from typing import Tuple, List, Dict, Union, Any, Optional

# Setup path per importare moduli dal progetto
sys.path.append(os.path.join(os.path.dirname(__file__)))
from unified_model import UnifiedModel

# --- Configurazione Pagina ---
ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    page_icon_path = os.path.join(ROOT_DIR, "assets/img/ico/favicon.ico")
    st.set_page_config(
        page_title="AutoTriage NLP - Advanced",
        page_icon=page_icon_path,
        layout="wide", 
        initial_sidebar_state="expanded"
    )
except Exception:
    st.set_page_config(page_title="AutoTriage NLP", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

def load_css(file_path: str) -> None:
    """
    Carica un file CSS personalizzato e lo inietta nell'applicazione.

    Args:
        file_path (str): Percorso relativo del file CSS.
    """
    full_path = os.path.join(ROOT_DIR, file_path) if not os.path.isabs(file_path) else file_path
    try:
        mtime = os.path.getmtime(full_path)
        with open(full_path) as f:
            css_content = f.read()
            # Timestamp aggiunto per forzare il refresh della cache del browser
            st.markdown(f'<style>/* CSS loaded at {mtime} */ {css_content}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"File CSS non trovato: {full_path}")

# Caricamento stili globali
load_css("assets/css/style.css")

@st.cache_resource
def load_model() -> Optional['UnifiedModel']:
    """
    Carica il modello addestrato dalla cache o dal disco.
    Utilizza `@st.cache_resource` per evitare ricaricamenti a ogni interazione utente.

    Returns:
        Optional[UnifiedModel]: L'istanza del modello caricata, o None se il file non esiste.
    """
    model_path = os.path.join(ROOT_DIR, 'models/unified_model.pkl')
    if not os.path.exists(model_path):
        return None
    return UnifiedModel.load(model_path)

# Inizializzazione Globale del Modello
model = load_model()

def explain_with_lime(model: 'UnifiedModel', text: str, target_idx: int = 0, num_features: int = 5) -> Optional[Any]:
    """
    Genera una spiegazione LIME per una singola predizione.

    Args:
        model (UnifiedModel): Il modello da spiegare.
        text (str): Il testo del ticket.
        target_idx (int): Indice del target (0 = Categoria, 1 = Priorità). Default 0.
        num_features (int): Numero di feature (parole) da mostrare.

    Returns:
        Optional[Any]: Oggetto Explanation di LIME o None in caso di errore.
    """

    # Accesso allo stimatore specifico all'interno del MultiOutputClassifier
    estimator = model.pipeline.named_steps['clf'].estimators_[target_idx]
    explainer = LimeTextExplainer(class_names=estimator.classes_)

    # Wrapper per adattare l'interfaccia predict_proba di UnifiedModel a LIME
    def predict_proba_wrapper(texts: List[str]) -> Any:
        # Ritorna solo le probabilità per il target d'interesse
        return model.predict_proba(texts)[target_idx]

    try:
        exp = explainer.explain_instance(text, predict_proba_wrapper, num_features=num_features)
        return exp
    except Exception:
        return None

def calculate_priority(model: 'UnifiedModel', text: str) -> Tuple[str, float, Dict[str, float], List[str]]:
    """
    Calcola la priorità utilizzando logica Ibrida (ML + Regole).

    Combina le predizioni statistiche del modello con regole deterministiche basate su keyword critiche.

    Args:
        model (UnifiedModel): Modello predittivo.
        text (str): Testo del ticket.

    Returns:
        Tuple contenente:
        - Pred_Class (str): La classe di priorità finale ('Alta', 'Media', 'Bassa').
        - Score (float): Lo score di confidenza (o override rules).
        - Debug_Probs (Dict): Mappa delle probabilità per debug visuale.
        - Triggers (List[str]): Lista di parole chiave che hanno attivato regole (se presenti).
    """

    # Ottieni probabilità raw dal modello
    probs = model.predict_proba([text])[1][0]
    classes = model.pipeline.named_steps['clf'].estimators_[1].classes_
    prob_map = {c: p for c, p in zip(classes, probs)}
    
    score_alta = prob_map.get('Alta', 0.0)
    debug_probs = prob_map.copy()

    # Dizionari di regole (Rule-Engine)
    critical_keywords = [
        "virus", "hacker", "attacco", "violazione", "perso dati", "cancellato", 
        "fermo", "blocco", "bloccato", "scadenza", "entro domani", "urgent", 
        "subito", "velocemente", "critico", "panico", "terribile"
    ]
    dampening_keywords = [
        "con calma", "non è urgente", "non urgente", "nessuna fretta", 
        "quando potete", "appena possibile", "senza urgenza",
        "normale", "nessun problema", "funziona", "tutto ok", "informazione"
    ]

    text_lower = text.lower()
    is_critical = any(kw in text_lower for kw in critical_keywords)
    is_dampener = any(kw in text_lower for kw in dampening_keywords)

    # REGOLA 1: Override Critico (Se contiene keyword urgenti e nessuna keyword 'calma')
    if is_critical and not is_dampener:
        trigger_word = next((kw for kw in critical_keywords if kw in text_lower), "keyword")
        # Forza probabilità per visualizzazione
        debug_probs['Alta'] = 0.99
        debug_probs['Media'] = 0.01
        debug_probs['Bassa'] = 0.00
        return "Alta", 0.99, debug_probs, [trigger_word] 

    # REGOLA 2: Rischio Statistico (Se il modello è abbastanza sicuro dell'Alta priorità)
    if score_alta > 0.60 and not is_dampener:
         return "Alta", score_alta, debug_probs, ["Rischio Statistico Alta Priorità"]

    # REGOLA 3: Feedback Positivo (Ticket di 'Grazie' sono sempre bassa priorità)
    happy_keywords = ["perfetto", "ottimo", "risolto", "funziona tutto", "eccellente", "bravi", "complimenti"]
    is_happy = any(kw in text_lower for kw in happy_keywords)
    if is_happy and not is_critical:
         debug_probs['Alta'] = 0.00
         debug_probs['Media'] = 0.01
         debug_probs['Bassa'] = 0.99
         return "Bassa", 0.99, debug_probs, ["Feedback Positivo"]

    # FALLBACK: Usa la predizione standard del modello (Argmax)
    pred_class = model.predict([text])[0][1] 
    score = prob_map.get(pred_class, 0.0)
    return pred_class, score, debug_probs, []

# --- INTERFACCIA UTENTE ---
st.title("AutoTriage NLP")
st.markdown("##### Sistema intelligente di classificazione e prioritarizzazione ticket per l'assistenza aziendale")
st.markdown("---")

if model is None:
    st.error("⚠️ Modello non trovato! Esegui prima il training (`python src/train_unified_model.py`).")
    st.stop()

# Layout a Schede (Tabs)
tab1, tab2, tab3, tab4 = st.tabs([
    "Inserimento Manuale", "Importazione da File", 
    "Metriche Modello", "Confronto Dati Sintetici vs Reali"
])

# --- SIDEBAR ---

with st.sidebar:
    st.image(os.path.join(ROOT_DIR, "assets/img/png/logo.png"), width=60)
    st.title("Scheda Progetto")
    st.markdown("---")
    
    st.info(
        """
        **Obiettivo del Progetto**:
        Sistema di Machine Learning per l'analisi e lo smistamento automatico 
        dei ticket di assistenza aziendale.
        
        **Tecnologie Utilizzate**:
        * **Python**: Backend logic
        * **Scikit-Learn**: Modelli ML (Logistic Regression)
        * **Pandas**: Data Manipulation
        * **Streamlit**: Frontend UI
        * **LIME**: Explainable AI
        """
    )
    
    # Download PDF Tesi/Documentazione
    pdf_path = os.path.join(ROOT_DIR, "assets/doc/template.pdf")
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
            st.download_button(
                label="Scarica Documentazione",
                data=pdf_data,
                file_name="documentazione_progetto.pdf",
                mime="application/pdf"
            )
    st.markdown("---")
    st.caption("""
        © 2026 Project Work Universitario<br>
        Informatica per le Aziende Digitali (L-31)<br>
        Alex Di Paolo
    """, unsafe_allow_html=True)

# --- TAB 1: INSERIMENTO MANUALE ---
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.subheader("Caricamento Ticket")
        ticket_text = st.text_area(
            "Inserisci il contenuto del ticket:", height=200,
            placeholder="Esempio: Salve, il sistema gestionale è bloccato e non riusciamo a emettere fatture...",
            help="Scrivi o incolla qui il testo della richiesta per analizzarla in tempo reale."
        )
        analyze_btn = st.button("Analizza", type="primary")

    with col2:
        st.subheader("Risultato Analisi")
        if not analyze_btn:
             st.info("Inserisci un ticket e clicca su 'Analizza' per visualizzare la classificazione.")

        if analyze_btn and ticket_text:
            with st.spinner('Elaborazione NLP in corso (Inference + Explainability)...'):
                # Predizione Categoria (Standard ML)
                cat_pred = model.predict([ticket_text])[0][0]
                # Predizione Priorità (Hybrid: ML + Rules)
                pri_pred, pri_score, pri_probs, triggers = calculate_priority(model, ticket_text)

                st.markdown("###### SINTESI CLASSIFICAZIONE")
                col_res1, col_res2 = st.columns(2)
                
                # Cards risultati
                with col_res1:
                    st.markdown(f"""
                        <div class="prediction-card">
                            <div class="card-label">REPARTO CONSIGLIATO</div>
                            <div class="category-value" style="font-size: 1.8rem;">{cat_pred}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col_res2:
                    st.markdown(f"""
                        <div class="prediction-card">
                            <div class="card-label">LIVELLO D'URGENZA</div>
                            <span class="badge badge-{pri_pred}" style="font-size: 1.2rem; padding: 5px 15px;">{pri_pred}</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # --- Sezione Explainability (LIME) ---
                st.markdown("**Perché questo risultato? (LIME Analysis)**")
                keywords_data: List[Tuple[str, float]] = []

                if triggers:
                    for t in triggers:
                         keywords_data.append((t, 1.0)) # Massimo peso per le regole esplicite

                # Parole comuni da escludere dalla visualizzazione
                visual_stopwords = ['il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'in', 'di', 'a', 'è', 'ho', 'sono']

                try:
                    exp = explain_with_lime(model, ticket_text, target_idx=1, num_features=12) 
                    if exp:
                        for word, weight in exp.as_list():
                            if len(keywords_data) >= 6: break 
                            w_clean = word.lower().strip()
                            if (w_clean not in [k[0] for k in keywords_data] 
                                and w_clean not in visual_stopwords 
                                and len(w_clean) > 2): # Filtra parole troppo corte
                                keywords_data.append((word, abs(weight))) 
                except Exception:
                    pass

                if keywords_data and any(weight > 0.001 for _, weight in keywords_data):
                    keywords_data.sort(key=lambda x: x[1], reverse=True)
                    top_k = keywords_data[:6]
                    
                    kw_html = "Parole chiave rilevate: " + ", ".join([f"**{k[0]}**" for k in top_k])
                    st.markdown(kw_html)
                    
                    explain_df = pd.DataFrame(top_k, columns=['Parola', 'Impatto'])
                    st.bar_chart(explain_df.set_index('Parola'), color="#1a73e8")
                else:
                    st.info("Testo troppo breve o generico. Inserisci una frase più dettagliata per vedere l'analisi LIME.")

                st.divider()
                
                # Barre di confidenza del modello
                st.markdown("Analisi probabilità (Confidenza Statistica):")
                for cls in ['Bassa', 'Media', 'Alta']:
                     p = pri_probs.get(cls, 0.0)
                     st.progress(p, text=f"{cls}: {p:.1%}")

# --- TAB 2: IMPORTAZIONE DA FILE ---
with tab2:
    st.subheader("Caricamento Dati")
    uploaded_file = st.file_uploader("Carica file CSV", type="csv", label_visibility="collapsed")
    st.caption("Richiesto file CSV con colonna 'text' o 'body'.")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        text_col = None
        possible_cols = ['text', 'body', 'testo', 'descrizione']
        
        # Identificazione automatica colonna testo
        for col in possible_cols:
            if col in df.columns:
                text_col = col
                break
        
        if text_col:
            # Batch Prediction
            preds = model.predict(df[text_col])
            df['Categoria'] = preds[:, 0]
            
            results = [calculate_priority(model, txt) for txt in df[text_col]]
            df['Priorità'] = [r[0] for r in results]
            df['Confidenza'] = [r[1] for r in results]
            
            display_df = df.rename(columns={text_col: 'Descrizione'})
            
            st.data_editor(
                display_df[['Descrizione', 'Categoria', 'Priorità', 'Confidenza']], 
                width=700, 
                height=600,
                disabled=True 
            )
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Scarica Report Completo", csv, "report_analysis.csv", "text/csv")
        else:
            st.error("Errore: Colonna testo mancante (cercato: text, body, testo, descrizione).")

# --- TAB 3: METRICHE MODELLO ---
with tab3:
    st.subheader("Performance del Modello")
    
    # Caricamento JSON metriche salvato da train_unified_model.py
    metrics = None
    metrics_path = os.path.join(ROOT_DIR, "metrics_summary.json")
    
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    if metrics:
        with col_kpi1: st.metric("Accuracy", f"{metrics['category']['accuracy']:.0%}")
        with col_kpi2: st.metric("Precision", f"{metrics['category']['precision']:.0%}")
        with col_kpi3: st.metric("Recall", f"{metrics['category']['recall']:.0%}")
        with col_kpi4: st.metric("F1-Score", f"{metrics['category']['f1']:.0%}")
    else:
        st.warning("Metriche non disponibili. Esegui prima il training.")

    st.markdown("---")
    with st.expander("Guida alla lettura delle metriche"):
        st.markdown("""
        * **Confusion Matrix**: Confronta predizioni vs realtà. Diagonale = corretti, fuori diagonale = errori.
        * **Accuracy**: Percentuale totale di risposte corrette.
        * **Precision**: Affidabilità delle predizioni positive (quanti "Allarmi" sono veri).
        * **Recall**: Capacità di trovare tutti i casi positivi (quanti casi reali ho trovato).
        * **F1-Score**: Media armonica tra Precision e Recall, ottima per bilanciare i due aspetti.
        """)

    h1, h2 = st.columns(2)
    with h1: st.markdown("### Metriche Categoria")
    with h2: st.markdown("### Metriche Priorità")

    def show_metric_row(filename_cat: str, filename_pri: str, label: str) -> None:
        c1, c2 = st.columns(2)
        with c1:
            img_path_cat = os.path.join(ROOT_DIR, f"assets/img/png/{filename_cat}")
            if os.path.exists(img_path_cat):
                st.image(img_path_cat, use_column_width=True)
                st.markdown(f"<p style='text-align: center; color: #888; font-size: 0.9rem;'>{label}</p>", unsafe_allow_html=True)
        with c2:
            img_path_pri = os.path.join(ROOT_DIR, f"assets/img/png/{filename_pri}")
            if os.path.exists(img_path_pri):
                st.image(img_path_pri, use_column_width=True)
                st.markdown(f"<p style='text-align: center; color: #888; font-size: 0.9rem;'>{label}</p>", unsafe_allow_html=True)

    show_metric_row("confusion_matrix_category.png", "confusion_matrix_priority.png", "Confusion Matrix")
    show_metric_row("accuracy_category.png", "accuracy_priority.png", "Accuracy")
    show_metric_row("precision_category.png", "precision_priority.png", "Precision")
    show_metric_row("recall_category.png", "recall_priority.png", "Recall")
    show_metric_row("f1score_category.png", "f1score_priority.png", "F1-score")

# --- TAB 4: CONFRONTO SPERIMENTALE ---
with tab4:
    st.markdown(""" ### Metodologia
    Questo esperimento dimostra scientificamente **perché abbiamo scelto di utilizzare un dataset reale tradotto** 
    invece di creare dati sintetici (come richiesto dalla traccia base).
    Abbiamo addestrato **due modelli gemelli** (stessa architettura, stesso algoritmo):
    """)
    
    col_setup1, col_setup2 = st.columns(2, gap="medium")
    
    with col_setup1:
        st.error("**Modello A (Approccio Base)**")
        st.markdown("""
        * **Training Set**: 500 ticket sintetici.
        * **Generazione**: Script Python con regole fisse.
        * **Caratteristiche**: Frasi semplici, keyword ripetitive.
        * **Ipotesi**: Impara "a memoria" le regole.
        """)
        
    with col_setup2:
        st.success("**Modello B (Il Nostro Approccio)**")
        st.markdown("""
        * **Training Set**: 20.000 ticket reali (Kaggle).
        * **Generazione**: Traduzione neuronale di ticket veri.
        * **Caratteristiche**: Slang, errori di battitura, sinonimi complessi.
        * **Ipotesi**: Impara il *significato* del linguaggio.
        """)
    
    st.markdown("Abbiamo scambiato i dataset di test per valutare la **capacità di generalizzazione**.")
    
    st.divider()

    comparison_img_dir = os.path.join(ROOT_DIR, "assets/img/comparison")
    
    if os.path.exists(comparison_img_dir):
        col_res1, col_res2 = st.columns(2, gap="large")
        
        with col_res1:
            st.markdown(" ### Modello A | Addestrato su Sintetico e Testato su REALE")
            
            if os.path.exists(f"{comparison_img_dir}/conf_matrix_A_on_Real.png"):
                st.image(f"{comparison_img_dir}/conf_matrix_A_on_Real.png", use_column_width=True, caption="Performance CROLLATE su Dati Reali")
            
            st.error("""
            **FALLIMENTO (Accuracy ~50%)**

            Il modello non riconosce i ticket veri perché non contengono le keyword esatte che ha imparato a memoria. 
            È inutile in un contesto aziendale.
            """)

        with col_res2:
            st.markdown(" ### Modello B | Addestrato su Reale e Testato su SINTETICO")
            
            if os.path.exists(f"{comparison_img_dir}/conf_matrix_B_on_Synth.png"):
                st.image(f"{comparison_img_dir}/conf_matrix_B_on_Synth.png", use_column_width=True, caption="Performance ROBUSTE su Dati Sintetici")
            
            st.success("""
            **SUCCESSO (Accuracy ~65%)**

            Il modello reale è robusto: capisce anche i dati sintetici (che non ha mai visto) perché ha imparato la semantica generale del linguaggio.
            """)

        st.divider()
        
        col_final_chart, col_final_text = st.columns([1, 1], gap="medium")
        
        with col_final_chart:
             if os.path.exists(f"{comparison_img_dir}/comparison_chart.png"):
                 st.image(f"{comparison_img_dir}/comparison_chart.png", caption="Confronto Generalizzazione Modelli", use_column_width=True)
                 
        with col_final_text:
            st.markdown("### Conclusioni Finali")
            st.markdown("""
            Spesso nei progetti accademici o POC si tende a usare generatori di dati per "fare prima".
            Il nostro esperimento dimostra che questa scorciatoia è fatale: il modello addestrato sinteticamente non impara a *ragionare*, ma solo a riconoscere pattern fissi (overfitting).
            
            I dati reali di Kaggle, anche se tradotti automaticamente, mantengono l'elemento fondamentale: l'imprevedibilità umana:
            *   Errori grammaticali e di battitura.
            *   Sinonimi non standard ("il pc non va" vs "errore di sistema").
            *   Ambiguità semantiche.
            
            Addestrare su questo "rumore" costringe il modello a trovare pattern profondi e robusti, rendendolo pronto per l'uso in produzione.
            """)

    else:
        st.warning("Grafici non trovati.")
        st.markdown("Per generare i risultati dell'esperimento, esegui:")
        st.code("python src/compare_models.py", language="bash")