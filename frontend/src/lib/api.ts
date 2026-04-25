/**
 * ========================================
 * API Client - Backend FastAPI
 * ========================================
 *
 * Modulo per la comunicazione con il backend FastAPI.
 * Utilizza Axios per le chiamate HTTP REST.
 *
 * Tutte le funzioni gestiscono automaticamente:
 * - Serializzazione/deserializzazione JSON
 * - Gestione errori
 * - Tipi TypeScript per type-safety
 */

import axios, { AxiosError } from "axios";

// ========================================
// CONFIGURAZIONE AXIOS
// ========================================
/**
 * Legge il base URL dalle variabili d'ambiente.
 * Le variabili NEXT_PUBLIC_* sono disponibili lato client in Next.js.
 */
function getApiBaseUrl(): string {
  // Try process.env (runtime)
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  // Fallback per sviluppo
  return "http://localhost:8000";
}

const API_BASE_URL = getApiBaseUrl();

/**
 * Istanza Axios configurata con:
 * - Base URL comune a tutti gli endpoint
 * - Header Content-Type: application/json
 */
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Endpoint API
const API_ENDPOINTS = {
  CLASSIFY: `${API_BASE_URL}/api/v1/ticket/classify`,
  BATCH: `${API_BASE_URL}/api/v1/ticket/batch`,
  UPLOAD_CSV: `${API_BASE_URL}/api/v1/ticket/upload-csv`,
  METRICS_SUMMARY: `${API_BASE_URL}/api/v1/metrics/summary`,
  METRICS_IMAGES: `${API_BASE_URL}/api/v1/metrics/images`,
  HEALTH: `${API_BASE_URL}/api/v1/health`,
};

// ========================================
// TIPI PER RISPOSTE API
// ========================================

/**
 * Risposta API per classificazione singola ticket
 */
export interface ClassificationResult {
  category: string;        // Categoria assegnata (es: "Hardware")
  priority: string;        // Priorità assegnata (es: "Alta")
  confidence: number;      // Confidenza globale (0-1)
  probabilities: Record<string, number>;  // Probabilità per classe
  triggers: string[];      // Parole chiave che hanno attivato la priorità
  lime_explanation: Array<{  // Spiegazione LIME parole importanti
    word: string;
    weight: number;
    type: "trigger" | "lime";
  }>;
  text_preview: string;    // Anteprima testo classificato
  success: boolean;        // True se classificazione riuscita
}

/**
 * Risposta API per classificazione batch
 */
export interface BatchResult {
  results: Array<{
    text: string;           // Testo del ticket
    category: string;       // Categoria assegnata
    priority: string;       // Priorità assegnata
    confidence: number;     // Confidenza (0-1)
    probabilities: Record<string, number>;  // Probabilità per classe
    triggers: string[];     // Parole chiave attivatrici
    success: boolean;       // True se classificazione riuscita
  }>;
  summary: {
    total: number;                   // Totale ticket processati
    high_priority: number;           // Ticket priorità Alta
    medium_priority: number;         // Ticket priorità Media
    low_priority: number;            // Ticket priorità Bassa
    high_priority_percentage: number;  // % Alta priorità
  };
  success: boolean;          // True se operazione riuscita
}

/**
 * Risposta API per metriche modello
 */
export interface MetricsSummary {
  category: {
    accuracy: number;   // Accuracy classificazione categoria
    precision: number;  // Precision media
    recall: number;     // Recall media
    f1: number;         // F1-score medio
  };
  priority: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  };
}

// ========================================
// FUNZIONI API
// ========================================

/**
 * Classifica un singolo ticket.
 *
 * @param text - Testo del ticket da classificare
 * @returns Risultato classificazione con spiegazioni
 * @throws Error se chiamata fallisce
 */
export async function classifyTicket(text: string): Promise<ClassificationResult> {
  try {
    const response = await api.post(API_ENDPOINTS.CLASSIFY, { text });
    return response.data;
  } catch (error) {
    const axiosError = error as AxiosError<any>;
    console.error("Error classifying ticket:", axiosError.response?.data || axiosError.message);
    throw new Error(axiosError.response?.data?.detail || "Errore durante la classificazione");
  }
}

/**
 * Classifica più ticket contemporaneamente.
 *
 * @param texts - Array di testi da classificare
 * @returns Risultati + riepilogo statistiche
 * @throws Error se chiamata fallisce
 */
export async function classifyBatch(texts: string[]): Promise<BatchResult> {
  try {
    const response = await api.post(API_ENDPOINTS.BATCH, { texts });
    return response.data;
  } catch (error) {
    const axiosError = error as AxiosError<any>;
    console.error("Error in batch classification:", axiosError.response?.data || axiosError.message);
    throw new Error(axiosError.response?.data?.detail || "Errore durante la classificazione batch");
  }
}

/**
 * Carica un file CSV per analisi batch.
 *
 * @param file - File CSV da analizzare
 * @returns Risultati + riepilogo
 * @throws Error se upload fallisce
 */
export async function uploadCSV(file: File): Promise<any> {
  try {
    // FormData per upload file multipart
    const formData = new FormData();
    formData.append("file", file);

    const response = await api.post(API_ENDPOINTS.UPLOAD_CSV, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  } catch (error) {
    const axiosError = error as AxiosError<any>;
    console.error("Error uploading CSV:", axiosError.response?.data || axiosError.message);
    throw new Error(axiosError.response?.data?.detail || "Errore durante l'upload del CSV");
  }
}

/**
 * Recupera le metriche del modello.
 *
 * @returns Oggetto con metriche accuracy, precision, recall, f1
 * @throws Error se recupero fallisce
 */
export async function getMetricsSummary(): Promise<MetricsSummary> {
  try {
    console.log("Fetching metrics from:", API_ENDPOINTS.METRICS_SUMMARY);
    const response = await api.get(API_ENDPOINTS.METRICS_SUMMARY);
    return response.data;
  } catch (error) {
    const axiosError = error as AxiosError<any>;
    console.error("Error fetching metrics:", axiosError.response?.data || axiosError.message);
    console.error("Request URL:", API_ENDPOINTS.METRICS_SUMMARY);
    console.error("Full error:", error);
    throw new Error(axiosError.response?.data?.detail || "Errore durante il recupero delle metriche");
  }
}

/**
 * Recupera i percorsi delle immagini delle metriche (confusion matrix).
 *
 * @returns Oggetto con paths delle immagini
 * @returns Oggetto vuoto se errore (non bloccante)
 */
export async function getMetricsImages(): Promise<Record<string, string>> {
  try {
    const response = await api.get(API_ENDPOINTS.METRICS_IMAGES);
    return response.data;
  } catch (error) {
    console.error("Error fetching metrics images:", error);
    return {};  // Non bloccante: ritorna oggetto vuoto
  }
}

/**
 * Verifica lo stato di salute del backend.
 *
 * @returns Oggetto con status e flag model_loaded
 * @returns {status: "unhealthy", model_loaded: false} se errore
 */
export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  try {
    const response = await api.get(API_ENDPOINTS.HEALTH);
    return response.data;
  } catch (error) {
    return { status: "unhealthy", model_loaded: false };
  }
}
