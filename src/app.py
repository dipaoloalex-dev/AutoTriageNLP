"""
AutoTriage NLP - Applicazione Web Streamlit
===========================================

Frontend del progetto AutoTriage. Gestisce l'interfaccia utente per:
1. Analizzare i ticket in tempo reale.
2. Caricare CSV per l'analisi batch.
3. Visualizzare le metriche del modello.
4. Integrare LIME per l'interpretabilità delle predizioni.
5. Mostrare il test comparativo tra dati sintetici e reali.
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

# Aggiungo la directory corrente al path per importare correttamente la classe UnifiedModel
sys.path.append(os.path.join(os.path.dirname(__file__)))
from unified_model import UnifiedModel

# --- CONFIGURAZIONE PAGINA ---
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
    # Fallback in caso di problemi con l'icona custom
    st.set_page_config(page_title="AutoTriage NLP", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

def load_css(file_path: str) -> None:
    """Carica e inietta il foglio di stile personalizzato per l'interfaccia."""
    full_path = os.path.join(ROOT_DIR, file_path) if not os.path.isabs(file_path) else file_path
    try:
        mtime = os.path.getmtime(full_path)
        with open(full_path) as f:
            css_content = f.read()
            # Uso il timestamp per forzare l'aggiornamento della cache del browser se modifico il CSS
            st.markdown(f'<style>/* CSS loaded at {mtime} */ {css_content}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Attenzione: File CSS non trovato in {full_path}")

load_css("assets/css/style.css")

@st.cache_resource
def load_model() -> Optional['UnifiedModel']:
    """
    Recupera il modello salvato. 
    Uso @st.cache_resource per evitare che Streamlit ricarichi il file .pkl 
    a ogni interazione o ricaricamento della pagina.
    """
    model_path = os.path.join(ROOT_DIR, 'models/unified_model.pkl')
    if not os.path.exists(model_path):
        return None
    return UnifiedModel.load(model_path)

# Inizializzazione del modello all'avvio dell'app
model = load_model()

def explain_with_lime(model: 'UnifiedModel', text: str, target_idx: int = 0, num_features: int = 5) -> Optional[Any]:
    """
    Applica LIME per estrarre le parole che hanno influenzato di più la predizione.
    """
    # Vado a pescare lo stimatore specifico dal MultiOutputClassifier (Categoria o Priorità)
    estimator = model.pipeline.named_steps['clf'].estimators_[target_idx]
    explainer = LimeTextExplainer(class_names=estimator.classes_)

    # Adatto l'output del mio modello per farlo digerire a LIME
    def predict_proba_wrapper(texts: List[str]) -> Any:
        return model.predict_proba(texts)[target_idx]

    try:
        exp = explainer.explain_instance(text, predict_proba_wrapper, num_features=num_features)
        return exp
    except Exception:
        return None

def calculate_priority(model: 'UnifiedModel', text: str) -> Tuple[str, float, Dict[str, float], List[str]]:
    """
    Logica ibrida per la priorità: unisce l'output probabilistico del modello ML 
    con un set di regole deterministiche basate su keyword critiche.
    """
    probs = model.predict_proba([text])[1][0]
    classes = model.pipeline.named_steps['clf'].estimators_[1].classes_
    prob_map = {c: p for c, p in zip(classes, probs)}
    
    score_alta = prob_map.get('Alta', 0.0)
    debug_probs = prob_map.copy()

    # Dizionari per il controllo regole
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

    # Controllo 1: Presenza di termini molto critici senza formule di cortesia/calma
    if is_critical and not is_dampener:
        trigger_word = next((kw for kw in critical_keywords if kw in text_lower), "keyword")
        debug_probs['Alta'] = 0.99
        debug_probs['Media'] = 0.01
        debug_probs['Bassa'] = 0.00
        return "Alta", 0.99, debug_probs, [trigger_word] 

    # Controllo 2: Il modello propende per l'Alta priorità in modo marcato
    if score_alta > 0.60 and not is_dampener:
         return "Alta", score_alta, debug_probs, ["Rischio Statistico Alta Priorità"]

    # Controllo 3: Ticket di ringraziamento o feedback positivo
    happy_keywords = ["perfetto", "ottimo", "risolto", "funziona tutto", "eccellente", "bravi", "complimenti"]
    is_happy = any(kw in text_lower for kw in happy_keywords)
    if is_happy and not is_critical:
         debug_probs['Alta'] = 0.00
         debug_probs['Media'] = 0.01
         debug_probs['Bassa'] = 0.99
         return "Bassa", 0.99, debug_probs, ["Feedback Positivo"]

    # Fallback: se non scatta nessuna regola, mi fido della predizione base del modello
    pred_class = model.predict([text])[0][1] 
    score = prob_map.get(pred_class, 0.0)
    return pred_class, score, debug_probs, []

# --- INTERFACCIA UTENTE ---
st.title("AutoTriage NLP")
st.markdown("##### Sistema intelligente di classificazione e prioritarizzazione ticket per l'assistenza aziendale")
st.markdown("---")

if model is None:
    st.error("Modello non trovato. Avvia prima lo script di training (`python src/train_unified_model.py`).")
    st.stop()

# Layout principale
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
        * **Python**: Logica Backend
        * **Scikit-Learn**: Modelli ML (Logistic Regression)
        * **Pandas**: Data Manipulation
        * **Streamlit**: Frontend UI
        * **LIME**: Explainable AI
        """
    )
    
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
            help="Scrivi o incolla qui il testo per fare una prova di classificazione."
        )
        analyze_btn = st.button("Analizza", type="primary")

    with col2:
        st.subheader("Risultato Analisi")
        if not analyze_btn:
             st.info("Inserisci un ticket e clicca su 'Analizza' per vedere il risultato.")

        if analyze_btn and ticket_text:
            with st.spinner('Elaborazione testuale in corso...'):
                cat_pred = model.predict([ticket_text])[0][0]
                pri_pred, pri_score, pri_probs, triggers = calculate_priority(model, ticket_text)

                st.markdown("###### SINTESI CLASSIFICAZIONE")
                col_res1, col_res2 = st.columns(2)
                
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
                
                # --- SEZIONE LIME ---
                st.markdown("**Perché questo risultato? (LIME Analysis)**")
                keywords_data: List[Tuple[str, float]] = []

                if triggers:
                    for t in triggers:
                         keywords_data.append((t, 1.0))

                visual_stopwords = ['il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'in', 'di', 'a', 'è', 'ho', 'sono']

                try:
                    exp = explain_with_lime(model, ticket_text, target_idx=1, num_features=12) 
                    if exp:
                        for word, weight in exp.as_list():
                            if len(keywords_data) >= 6: break 
                            w_clean = word.lower().strip()
                            if (w_clean not in [k[0] for k in keywords_data] 
                                and w_clean not in visual_stopwords 
                                and len(w_clean) > 2):
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
                    st.info("Testo troppo breve. Inserisci una frase più lunga per permettere a LIME di generare il grafico.")

                st.divider()
                
                st.markdown("Confidenza del modello sulle priorità:")
                for cls in ['Bassa', 'Media', 'Alta']:
                     p = pri_probs.get(cls, 0.0)
                     st.progress(p, text=f"{cls}: {p:.1%}")

# --- TAB 2: IMPORTAZIONE DA FILE ---
with tab2:
    st.subheader("Caricamento Dati in Batch")
    uploaded_file = st.file_uploader("Carica file CSV", type="csv", label_visibility="collapsed")
    st.caption("Il file CSV deve avere una colonna denominata 'text', 'body', 'testo' o 'descrizione'.")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        text_col = None
        possible_cols = ['text', 'body', 'testo', 'descrizione']
        
        for col in possible_cols:
            if col in df.columns:
                text_col = col
                break
        
        if text_col:
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
            st.download_button("Scarica Report CSV", csv, "report_analisi_ticket.csv", "text/csv")
        else:
            st.error("Errore: Impossibile trovare la colonna col testo del ticket.")

# --- TAB 3: METRICHE MODELLO ---
with tab3:
    st.subheader("Performance del Modello")
    
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
        st.warning("File delle metriche non trovato. Addestra il modello per generarlo.")

    st.markdown("---")
    with st.expander("Come leggere questi dati"):
        st.markdown("""
        * **Confusion Matrix**: Visualizza gli errori di smistamento (la diagonale rappresenta le previsioni corrette).
        * **Accuracy**: Percentuale complessiva di classificazioni esatte.
        * **Precision**: Affidabilità del modello quando assegna una determinata categoria.
        * **Recall**: Capacità del modello di non perdersi per strada i ticket di una certa categoria.
        * **F1-Score**: Una media bilanciata tra Precision e Recall.
        """)

    h1, h2 = st.columns(2)
    with h1: st.markdown("<h3 style='text-align: center;'>Categoria</h3>", unsafe_allow_html=True)
    with h2: st.markdown("<h3 style='text-align: center;'>Priorità</h3>", unsafe_allow_html=True)

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
    st.markdown(""" ### Metodologia del Test
    In questa sezione ho voluto mettere alla prova l'idea alla base del mio progetto: 
    utilizzare un dataset reale (anche se tradotto) è davvero meglio che usare dei dati sintetici creati ad hoc?
    Per verificarlo, ho addestrato due modelli identici:
    """)
    
    col_setup1, col_setup2 = st.columns(2, gap="medium")
    
    with col_setup1:
        st.error("**Modello A (Baseline con dati sintetici)**")
        st.markdown("""
        * Addestrato su 500 ticket generati via script.
        * Linguaggio molto rigido e ripetitivo.
        """)
        
    with col_setup2:
        st.success("**Modello B (L'approccio scelto per il progetto)**")
        st.markdown("""
        * Addestrato su ~20.000 ticket reali estratti da Kaggle.
        * Linguaggio naturale, comprensivo di errori e slang.
        """)
    
    st.markdown("Ho poi invertito i dataset di test per vedere come i due modelli se la cavavano su terreni sconosciuti.")
    st.divider()

    comparison_img_dir = os.path.join(ROOT_DIR, "assets/img/png")
    
    if os.path.exists(comparison_img_dir):
        col_res1, col_res2 = st.columns(2, gap="large")
        
        with col_res1:
            st.markdown(" ### Modello A testato sui Dati Reali")
            
            if os.path.exists(f"{comparison_img_dir}/confusion_matrix_a_on_real.png"):
                st.image(f"{comparison_img_dir}/confusion_matrix_a_on_real.png", use_column_width=True, caption="Forte calo delle prestazioni")
            
            st.error("""
            **Risultato deludente (Accuracy ~50%)**
            Il modello addestrato sui dati finti va in crisi con il linguaggio umano reale perché cerca pattern esatti che non trova.
            """)

        with col_res2:
            st.markdown(" ### Modello B testato sui Dati Sintetici")
            
            if os.path.exists(f"{comparison_img_dir}/confusion_matrix_b_on_synth.png"):
                st.image(f"{comparison_img_dir}/confusion_matrix_b_on_synth.png", use_column_width=True, caption="Prestazioni stabili")
            
            st.success("""
            **Risultato solido (Accuracy ~60%)**
            Il modello addestrato sui dati di Kaggle riesce a gestire bene anche i ticket sintetici, dimostrando un'ottima capacità di generalizzazione.
            """)

        st.divider()
        
        col_final_chart, col_final_text = st.columns([1, 1], gap="medium")
        
        with col_final_chart:
             if os.path.exists(f"{comparison_img_dir}/comparison_chart.png"):
                 st.image(f"{comparison_img_dir}/comparison_chart.png", caption="Confronto Generalizzazione Modelli", use_column_width=True)
                 
        with col_final_text:
            st.markdown("### Cosa ho dedotto da questo test")
            st.markdown("""
            I dati confermano che usare dei generatori testuali per fare prima è sconsigliato in un contesto aziendale. 
            Il modello sintetico va in overfitting, imparando le regole a memoria ma senza generalizzare.
            
            I dati reali (quelli del Modello B) sono essenziali perché il "rumore" di fondo (errori di battitura, sinonimi, ambiguità) costringe 
            l'algoritmo a cercare pattern semantici più profondi, rendendolo molto più stabile quando messo in produzione.
            """)

    else:
        st.warning("Grafici del test non trovati.")
        st.markdown("Per generare le metriche di confronto, esegui:")
        st.code("python src/compare_models.py", language="bash")
