/**
 * ========================================
 * Costanti dell'Applicazione
 * ========================================
 *
 * File centralizzato per tutte le costanti usate nell'app:
 * - URL API backend
 * - Endpoint API
 * - Configurazioni UI (colori, icone)
 * - Dati di esempio
 */

// ========================================
// CONFIGURAZIONE API
// ========================================
// URL base del backend FastAPI
// Usa env var NEXT_PUBLIC_API_URL se disponibile, altrimenti localhost
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ----------------------------------------
// ENDPOINT API
// ----------------------------------------
// Tutti gli endpoint del backend FastAPI
export const API_ENDPOINTS = {
  CLASSIFY: `${API_BASE_URL}/api/v1/ticket/classify`,        // Classificazione singolo ticket
  BATCH: `${API_BASE_URL}/api/v1/ticket/batch`,              // Classificazione multipla
  UPLOAD_CSV: `${API_BASE_URL}/api/v1/ticket/upload-csv`,    // Upload file CSV
  METRICS_SUMMARY: `${API_BASE_URL}/api/v1/metrics/summary`, // Metriche numeriche
  METRICS_IMAGES: `${API_BASE_URL}/api/v1/metrics/images`,   // Percorsi immagini grafici
  HEALTH: `${API_BASE_URL}/api/v1/health`,                    // Health check backend
} as const;

// ========================================
// DATI DOMINIO
// ========================================
// Categorie possibili per i ticket
export const CATEGORIES = [
  "Hardware",
  "Software",
  "Reti",
  "Accesso",
  "Altro",
] as const;

// Livelli di priorità possibili
export const PRIORITIES = ["Bassa", "Media", "Alta"] as const;

// ----------------------------------------
// CONFIGURAZIONE PRIORITÀ
// ----------------------------------------
// Configurazione completa per ogni livello di priorità:
// - Classi Tailwind per styling
// - Icona emoji
// - Colore (nome)
export const PRIORITY_CONFIG = {
  Alta: {
    color: "red",
    bgClass: "bg-red-500/20",          // Sfondo rosso 20% opacità
    textClass: "text-red-400",         // Testo rosso
    borderClass: "border-red-500/30",  // Bordo rosso 30% opacità
    icon: "🔴",                        // Emoji cerchio rosso
  },
  Media: {
    color: "orange",
    bgClass: "bg-orange-500/20",
    textClass: "text-orange-400",
    borderClass: "border-orange-500/30",
    icon: "🟠",                        // Emoji cerchio arancione
  },
  Bassa: {
    color: "green",
    bgClass: "bg-green-500/20",
    textClass: "text-green-400",
    borderClass: "border-green-500/30",
    icon: "🟢",                        // Emoji cerchio verde
  },
} as const;

// ========================================
// DATI DI ESEMPIO
// ========================================
// Ticket di esempio per la pagina di classificazione manuale
// Usati come "quick fill" per testare l'applicazione
// Mix di categorie (Tecnico, Amministrativo, Commerciale) e priorità (Bassa, Media, Alta)
export const EXAMPLE_TICKETS = [
  "Il server è down e non funzionano i servizi, urgenza!", // Tecnico - Alta
  "Vorrei informazioni generiche sui vostri prodotti, senza urgenza.", // Commerciale - Bassa
  "Ho problemi con il sistema fatturazione per emettere le fatture.", // Amministrativo - Media
  "La connessione VPN è lenta e non riesco a lavorare da casa.", // Tecnico - Media
  "Cliente importante che vuole annullare il contratto, gestire subito!", // Commerciale - Alta
];
