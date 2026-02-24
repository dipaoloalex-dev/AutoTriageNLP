"""
Modulo Generazione Dati Sintetici
=================================

Script di supporto per creare un dataset di ticket IT "finti".
Uso questi dati per addestrare il Modello A nel test comparativo, 
in modo da dimostrare come un modello impari a memoria le regole 
(andando in overfitting) se non viene addestrato su testo reale.
"""

import pandas as pd
import random
import os
from typing import List, Dict, Union

# --- CONFIGURAZIONE ---
OUTPUT_FILE: str = os.path.join('data', 'synthetic_tickets.csv')
NUM_TICKETS: int = 500

# --- VOCABOLARI PER I TEMPLATE ---

# 1. Tecnico
TECH_SUBJECTS = [
    "il server aziendale", "il mio PC fisso", "la stampante del 2° piano", 
    "la connessione VPN", "il software di contabilità", "l'app mobile", 
    "il database clienti", "monitor esterno", "mouse e tastiera", "tastiera wireless"
]
TECH_PROBLEMS = [
    "non ne vuole sapere di accendersi", "è diventato lentissimo", "restituisce errore 404", 
    "si è bloccato all'improvviso", "cade in continuazione", "fa un rumore strano molto forte", 
    "non stampa i colori", "mostra una schermata blu", "sembra infetto da malware"
]
TECH_ACTIONS = [
    "fare un riavvio forzato", "formattare tutto", "sostituire il pezzo", 
    "fare l'aggiornamento driver", "verificare i log", "reinstallare da zero", "configurare la rete"
]

# 2. Amministrativo
ADMIN_SUBJECTS = [
    "la fattura n. 203", "il pagamento dello stipendio", "il bonifico fornitori", 
    "la mia busta paga", "la nota spese di marzo", "il contratto di consulenza", 
    "il rimborso km", "il piano ferie", "i permessi retribuiti"
]
ADMIN_PROBLEMS = [
    "è scaduta da due giorni", "non risulta ancora arrivato", "contiene un importo errato", 
    "non si trova nell'archivio", "deve essere ancora approvato", 
    "non corrisponde agli accordi", "è stato rifiutato dal sistema"
]
ADMIN_ACTIONS = [
    "fare un controllo incrociato", "inviare la copia corretta", "rettificare l'importo", 
    "dare l'ok finale", "firmare digitalmente", "verificare con la banca"
]

# 3. Commerciale
COMM_SUBJECTS = [
    "il preventivo per Rossi Srl", "la nuova offerta Summer", "il listino prezzi 2026", 
    "la promo dedicata ai partner", "il catalogo prodotti aggiornato", 
    "la demo per il nuovo cliente", "il lead qualificato", "la partnership strategica"
]
COMM_PROBLEMS = [
    "necessita di più dettagli tecnici", "è troppo alto per il loro budget", 
    "ha convinto il cliente all'acquisto", "sembra molto promettente", 
    "richiede un incontro di persona", "vuole disdire il contratto attuale"
]
COMM_ACTIONS = [
    "ricontattare telefonicamente", "inviare la brochure", "fissare una call conoscitiva", 
    "preparare una proposta personalizzata", "aggiornare lo stato sul CRM"
]

# Parole chiave che l'algoritmo dovrà "imparare" a legare alle priorità
HIGH_KEYWORDS = ["URGENTE", "SUBITO", "BLOCCATO", "CRITICO", "FERMO", "SCADENZA", "OGGI", "IMMEDIATO"]
MEDIUM_KEYWORDS = ["appena possibile", "problema fastidioso", "attenzione richiesta", "verifica necessaria", "gentilmente"]
LOW_KEYWORDS = ["con calma", "solo informazione", "quando avete tempo", "nessuna fretta", "curiosità", "info generica"]

def generate_ticket() -> Dict[str, Union[str, int]]:
    """
    Costruisce un singolo ticket fittizio pescando pezzi dai vocabolari
    e incollandoli in un template di base.
    """
    category = random.choice(["Tecnico", "Amministrativo", "Commerciale"])
    priority = random.choice(["Alta", "Media", "Bassa"])
    
    # Preparo le frasi in base alla categoria
    if category == "Tecnico":
        subj = random.choice(TECH_SUBJECTS)
        prob = random.choice(TECH_PROBLEMS)
        act = random.choice(TECH_ACTIONS)
        templates = [
            f"Ciao team IT, ho un problema con {subj}: {prob}. Potete venire a {act}?",
            f"Segnalazione guasto: {subj} {prob}. Credo sia necessario {act} al più presto.",
            f"Non riesco a lavorare, {subj} {prob}. Ho provato a spegnere e riaccendere ma nulla. Bisogna {act}.",
            f"Vi scrivo perché {subj} {prob}. Attendo voi per {act}."
        ]
    elif category == "Amministrativo":
        subj = random.choice(ADMIN_SUBJECTS)
        prob = random.choice(ADMIN_PROBLEMS)
        act = random.choice(ADMIN_ACTIONS)
        templates = [
            f"Buongiorno Amministrazione, vi scrivo in merito a {subj} che {prob}. Potreste {act}?",
            f"Per info: {subj} {prob}. Resto in attesa per {act}.",
            f"Salve, {subj} {prob}. Verificate per favore? Bisognerebbe {act}.",
            f"Richiesta supporto amministrativo su {subj}. Il problema è che {prob}."
        ]
    else: 
        subj = random.choice(COMM_SUBJECTS)
        prob = random.choice(COMM_PROBLEMS)
        act = random.choice(COMM_ACTIONS)
        templates = [
            f"Aggiornamento vendita: {subj} {prob}. Prossimo step: {act}.",
            f"Il cliente ci ha scritto riguardo {subj}. Dice che {prob}. Penso sia il caso di {act}.",
            f"Info commerciale: {subj} {prob}. Procedo a {act}?",
            f"Nuova opportunità: {subj} {prob}. Da {act}."
        ]

    base_body = random.choice(templates)

    # Inserisco in modo forzato le keyword di priorità per creare il pattern da far imparare al modello
    if priority == "Alta":
        kw = random.choice(HIGH_KEYWORDS)
        if random.random() > 0.5:
             final_body = f"[{kw}] {base_body} È una situazione {kw}!"
        else:
             final_body = f"{base_body} Intervenire {kw}!!"
    
    elif priority == "Media":
        kw = random.choice(MEDIUM_KEYWORDS)
        if random.random() > 0.6: 
            final_body = f"{base_body} Da risolvere {kw}."
        else:
            final_body = base_body # Lascio alcuni ticket medi "puliti"
    
    else: # Bassa
        kw = random.choice(LOW_KEYWORDS)
        final_body = f"{base_body} Comunque da fare {kw}."

    title = f"Ticket {category} - {priority}"

    # Ritorno la riga per il Dataframe
    return {
        "title": title,
        "body": final_body, 
        "text": final_body, # Duplico body in text perché gli altri script leggono la colonna 'text'
        "category": category,
        "priority": priority
    }

def main() -> None:
    print(f"Generazione di {NUM_TICKETS} ticket sintetici...")

    data = [generate_ticket() for _ in range(NUM_TICKETS)]
    df = pd.DataFrame(data)

    df['id'] = range(1, len(df) + 1)
    df = df[['id', 'title', 'body', 'text', 'category', 'priority']]

    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"File salvato in: {OUTPUT_FILE}")
    print("\nCheck Bilanciamento Classi:")
    print(df['category'].value_counts())

if __name__ == "__main__":
    main()
