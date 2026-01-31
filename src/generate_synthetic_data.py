"""
Modulo Generazione Dati Sintetici
=================================

Questo modulo genera un dataset di ticket di assistenza IT sintetici.
Utilizza template basati su regole e campionamento casuale per creare esempi etichettati per
l'addestramento di modelli di machine learning (Modello A nell'esperimento di confronto).

I dati generati imitano pattern semplici e distinti per dimostrare i rischi di overfitting
del modello quando confrontati con dati del mondo reale.
"""

import pandas as pd
import random
import os
from typing import List, Dict, Any, Union

# ==============================================================================
# 1. COSTANTI E TEMPLATE
# ==============================================================================

OUTPUT_FILE: str = os.path.join('data', 'synthetic_tickets.csv')
NUM_TICKETS: int = 500

# Vocabolari specifici del dominio per il riempimento dei template
TECH_SUBJECTS: List[str] = [
    "il server aziendale", "il mio PC fisso", "la stampante del 2° piano", 
    "la connessione VPN", "il software di contabilità", "l'app mobile", 
    "il database clienti", "monitor esterno", "mouse e tastiera", "tastiera wireless"
]

TECH_PROBLEMS: List[str] = [
    "non ne vuole sapere di accendersi", "è diventato lentissimo", "restituisce errore 404", 
    "si è bloccato all'improvviso", "cade in continuazione", "fa un rumore strano molto forte", 
    "non stampa i colori", "mostra una schermata blu", "sembra infetto da malware"
]

TECH_ACTIONS: List[str] = [
    "fare un riavvio forzato", "formattare tutto", "sostituire il pezzo", 
    "fare l'aggiornamento driver", "verificare i log", "reinstallare da zero", "configurare la rete"
]

ADMIN_SUBJECTS: List[str] = [
    "la fattura n. 203", "il pagamento dello stipendio", "il bonifico fornitori", 
    "la mia busta paga", "la nota spese di marzo", "il contratto di consulenza", 
    "il rimborso km", "il piano ferie", "i permessi retribuiti"
]

ADMIN_PROBLEMS: List[str] = [
    "è scaduta da due giorni", "non risulta ancora arrivato", "contiene un importo errato", 
    "non si trova nell'archivio", "deve essere ancora approvato", 
    "non corrisponde agli accordi", "è stato rifiutato dal sistema"
]

ADMIN_ACTIONS: List[str] = [
    "fare un controllo incrociato", "inviare la copia corretta", "rettificare l'importo", 
    "dare l'ok finale", "firmare digitalmente", "verificare con la banca"
]

COMM_SUBJECTS: List[str] = [
    "il preventivo per Rossi Srl", "la nuova offerta Summer", "il listino prezzi 2026", 
    "la promo dedicata ai partner", "il catalogo prodotti aggiornato", 
    "la demo per il nuovo cliente", "il lead qualificato", "la partnership strategica"
]

COMM_PROBLEMS: List[str] = [
    "necessita di più dettagli tecnici", "è troppo alto per il loro budget", 
    "ha convinto il cliente all'acquisto", "sembra molto promettente", 
    "richiede un incontro di persona", "vuole disdire il contratto attuale"
]

COMM_ACTIONS: List[str] = [
    "ricontattare telefonicamente", "inviare la brochure", "fissare una call conoscitiva", 
    "preparare una proposta personalizzata", "aggiornare lo stato sul CRM"
]

# Parole chiave che segnalano fortemente un'alta priorità (feature per rule-learning)
HIGH_KEYWORDS: List[str] = [
    "URGENTE", "SUBITO", "BLOCCATO", "CRITICO", "FERMO", "SCADENZA", "OGGI", "IMMEDIATO"
]

MEDIUM_KEYWORDS: List[str] = [
    "appena possibile", "problema fastidioso", "attenzione richiesta", "verifica necessaria", "gentilmente"
]

LOW_KEYWORDS: List[str] = [
    "con calma", "solo informazione", "quando avete tempo", "nessuna fretta", "curiosità", "info generica"
]

def generate_ticket() -> Dict[str, Union[str, int]]:
    """
    Genera un singolo punto dati ticket sintetico.

    Seleziona una categoria casuale (Tecnico, Amministrativo, Commerciale) e costruisce
    il corpo e il titolo del ticket usando template e vocabolari predefiniti.
    Inietta anche parole chiave specifiche per la priorità per creare correlazioni.

    Returns:
        Dict[str, Union[str, int]]: Un dizionario rappresentante il record del ticket con chiavi:
                                    ['title', 'body', 'text', 'category', 'priority']
    """
    category = random.choice(["Tecnico", "Amministrativo", "Commerciale"])
    priority = random.choice(["Bassa", "Media", "Alta"])
    
    # Variabili di contesto
    subj: str = ""
    prob: str = ""
    act: str = ""
    body_template: List[str] = []

    # Logica di Dominio: Seleziona il vocabolario appropriato basato sulla Categoria
    if category == "Tecnico":
        subj = random.choice(TECH_SUBJECTS)
        prob = random.choice(TECH_PROBLEMS)
        act = random.choice(TECH_ACTIONS)
        body_template = [
            f"Ciao team IT, ho un problema con {subj}: {prob}. Potete venire a {act}?",
            f"Segnalazione guasto: {subj} {prob}. Credo sia necessario {act} al più presto.",
            f"Non riesco a lavorare, {subj} {prob}. Ho provato a spegnere e riaccendere ma nulla. Bisogna {act}.",
            f"Vi scrivo perché {subj} {prob}. Attendo voi per {act}."
        ]
    elif category == "Amministrativo":
        subj = random.choice(ADMIN_SUBJECTS)
        prob = random.choice(ADMIN_PROBLEMS)
        act = random.choice(ADMIN_ACTIONS)
        body_template = [
            f"Buongiorno Amministrazione, vi scrivo in merito a {subj} che {prob}. Potreste {act}?",
            f"Per info: {subj} {prob}. Resto in attesa per {act}.",
            f"Salve, {subj} {prob}. Verificate per favore? Bisognerebbe {act}.",
            f"Richiesta supporto amministrativo su {subj}. Il problema è che {prob}."
        ]
    else: 
        subj = random.choice(COMM_SUBJECTS)
        prob = random.choice(COMM_PROBLEMS)
        act = random.choice(COMM_ACTIONS)
        body_template = [
            f"Aggiornamento vendita: {subj} {prob}. Prossimo step: {act}.",
            f"Il cliente ci ha scritto riguardo {subj}. Dice che {prob}. Penso sia il caso di {act}.",
            f"Info commerciale: {subj} {prob}. Procedo a {act}?",
            f"Nuova opportunità: {subj} {prob}. Da {act}."
        ]

    base_body = random.choice(body_template)
    final_body: str = base_body

    # Logica di Priorità: Inietta keyword basate sul Livello di Priorità
    if priority == "Alta":
        kw = random.choice(HIGH_KEYWORDS)
        # Iniezione deterministica di keyword ad Alta Priorità
        if random.random() > 0.5:
             final_body = f"[{kw}] {base_body} È una situazione {kw}!"
        else:
             final_body = f"{base_body} Intervenire {kw}!!"
    
    elif priority == "Media":
        if random.random() > 0.6: 
            final_body = f"{base_body} {random.choice(MEDIUM_KEYWORDS)}."
    
    else: 
        kw = random.choice(LOW_KEYWORDS)
        final_body = f"{base_body} Comunque {kw}."

    title = f"Ticket {category} - {priority}"

    # Ritorna dizionario strutturato
    return {
        "title": title,
        "body": final_body, 
        "text": final_body, # Duplicazione body come 'text' per coerenza API
        "category": category,
        "priority": priority
    }

def main() -> None:
    """
    Funzione principale per generare e salvare il dataset.
    """
    print(f"Generazione di {NUM_TICKETS} ticket sintetici in corso...")

    data = [generate_ticket() for _ in range(NUM_TICKETS)]
    df = pd.DataFrame(data)

    # Aggiungi colonna ID
    df['id'] = range(1, len(df) + 1)

    # Ordinamento colonne per leggibilità
    df = df[['id', 'title', 'body', 'text', 'category', 'priority']]

    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Dataset sintetico salvato in: {OUTPUT_FILE}")
    print("\nDistribuzione Classi (Categoria):")
    print(df['category'].value_counts())
    print("\nDistribuzione Classi (Priorità):")
    print(df['priority'].value_counts())

if __name__ == "__main__":
    main()
