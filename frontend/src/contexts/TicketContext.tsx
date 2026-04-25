/**
 * ========================================
 * TicketContext - Context API per Stato Globale
 * ========================================
 *
 * Gestisce lo stato globale della classificazione ticket.
 * Fornisce un'interfaccia centralizzata per:
 * - Classificare ticket
 * - Gestire stati di caricamento/errore
 * - Accedere ai risultati
 *
 * Utilizza il pattern Context + Hook per evitare prop drilling.
 */

"use client"; // Necessario per Context API in Next.js App Router

import React, { createContext, useContext, useState, useCallback } from "react";
import { ClassificationResult, classifyTicket } from "@/lib/api";

// ========================================
// INTERFACCIA CONTEXT
// ========================================
interface TicketContextType {
  result: ClassificationResult | null;  // Risultato della classificazione
  isLoading: boolean;                    // True durante chiamata API
  error: string | null;                  // Messaggio di errore se fallito
  classify: (text: string) => Promise<void>;  // Funzione per classificare
  clear: () => void;                     // Funzione per resettare stato
}

// Creazione del Context
const TicketContext = createContext<TicketContextType | undefined>(undefined);

// ========================================
// PROVIDER
// ========================================
/**
 * Provider che avvolge i componenti che hanno bisogno dello stato ticket.
 *
 * @param children - Componenti figli da avvolgere
 */
export function TicketProvider({ children }: { children: React.ReactNode }) {
  // ----------------------------------------
  // STATI LOCALI
  // ----------------------------------------
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ----------------------------------------
  // FUNZIONE DI CLASSIFICAZIONE
  // ----------------------------------------
  /**
   * Classifica un ticket text usando l'API backend.
   * Gestisce automaticamente stati di caricamento ed errore.
   *
   * @param text - Testo del ticket da classificare
   */
  const classify = useCallback(async (text: string) => {
    // Validazione: testo vuoto
    if (!text.trim()) {
      setError("Inserisci un ticket da analizzare");
      return;
    }

    // Inizio caricamento
    setIsLoading(true);
    setError(null);

    try {
      // Chiamata API
      const response = await classifyTicket(text);
      setResult(response);
    } catch (err) {
      // Gestione errore
      const message = err instanceof Error ? err.message : "Errore sconosciuto";
      setError(message);
      setResult(null);
    } finally {
      // Fine caricamento (sempre eseguito)
      setIsLoading(false);
    }
  }, []);

  // ----------------------------------------
  // FUNZIONE DI RESET
  // ----------------------------------------
  /**
   * Ripristina lo stato iniziale (nessun risultato, nessun errore).
   */
  const clear = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  // ----------------------------------------
  // RENDER PROVIDER
  // ----------------------------------------
  return (
    <TicketContext.Provider value={{ result, isLoading, error, classify, clear }}>
      {children}
    </TicketContext.Provider>
  );
}

// ========================================
// HOOK PERSONALIZZATO
// ========================================
/**
 * Hook per accedere al TicketContext.
 *
 * @throws Error se usato fuori dal TicketProvider
 * @returns Il context con tutti i metodi e stati
 *
 * @example
 * function MyComponent() {
 *   const { result, isLoading, classify } = useTicket();
 *   // ...
 * }
 */
export function useTicket() {
  const context = useContext(TicketContext);

  // Controllo: useTicket deve essere usato dentro TicketProvider
  if (!context) {
    throw new Error("useTicket must be used within TicketProvider");
  }

  return context;
}
