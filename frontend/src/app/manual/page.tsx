/**
 * ========================================
 * ManualPage - Pagina Classificazione Manuale
 * ========================================
 *
 * Pagina per l'analisi singola di ticket.
 * Permette di:
 * - Inserire un ticket manualmente
 * - Ottenere classificazione in tempo reale
 * - Visualizzare spiegazioni LIME (parole chiave)
 * - Usare esempi predefiniti
 */

"use client"; // Necessario per interattività e Context

import React, { useState } from "react";
import { TicketProvider, useTicket } from "@/contexts/TicketContext";
import { Button } from "@/components/ui/Button";
import { TextArea } from "@/components/ui/TextArea";
import { PriorityBadge } from "@/components/PriorityBadge";
import { GlassCard } from "@/components/GlassCard";
import { NavigationButton } from "@/components/Navigation";
import { EXAMPLE_TICKETS } from "@/lib/constants";
import { AlertCircle, Sparkles, TrendingUp } from "lucide-react";

// ========================================
// COMPONENTE INTERNO (CON CONTEXT)
// ========================================
/**
 * Componente interno che usa il Context Ticket.
 * Separato dal wrapper per poter usare il Context.
 */
function ManualPageContent() {
  // ----------------------------------------
  // HOOK CONTEXT
  // ----------------------------------------
  // Estraggo stato e metodi dal TicketContext
  const { result, isLoading, error, classify, clear } = useTicket();

  // Stato locale per il testo del ticket
  const [text, setText] = useState("");

  // ----------------------------------------
  // HANDLERS
  // ----------------------------------------
  /**
   * Esegue la classificazione del ticket corrente.
   */
  const handleClassify = () => {
    classify(text);
  };

  /**
   * Carica un esempio di ticket nella textarea.
   */
  const handleExample = (example: string) => {
    setText(example);
  };

  // ========================================
  // RENDER
  // ========================================
  return (
    <div className="container-custom py-6">
      {/* Pulsante torna indietro */}
      <NavigationButton />

      {/* ----------------------------------------
          HERO SECTION
          ---------------------------------------- */}
      <div className="text-center mb-8 animate-fade-in-up">
        <h1 className="text-3xl md:text-4xl font-bold mb-4 text-white">
          Classificazione Ticket
        </h1>
        <p className="text-lg text-gray-400">
          Inserisci un ticket per ottenerne la categoria e la priorità automaticamente
        </p>
      </div>

      {/* ----------------------------------------
          MAIN CONTENT (GRID 2 COLONNE)
          ---------------------------------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-7xl mx-auto">

        {/* ========================================
            COLONNA SINISTRA: INPUT
            ======================================== */}
        <div className="space-y-6">
          <GlassCard className="animate-slide-in-left">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Sparkles className="w-6 h-6 text-blue-400" />
              Inserisci il Ticket
            </h2>

            {/* TextArea per input */}
            <TextArea
              label="Contenuto del ticket:"
              placeholder="Inserisci qui il testo del ticket..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              characterCount={text.length}
              maxLength={10000}
            />

            {/* Bottoni azione */}
            <div className="flex flex-wrap gap-3 mt-6">
              <Button
                onClick={handleClassify}
                isLoading={isLoading}
                disabled={!text.trim()}
                className="flex-1"
              >
                Analizza Ticket
              </Button>
              <Button
                variant="outline"
                onClick={clear}
                disabled={!result && !error}
              >
                Cancella
              </Button>
            </div>

            {/* ----------------------------------------
                ESEMPI DI TICKET
                ---------------------------------------- */}
            <div className="mt-6 pt-6 border-t border-white/10">
              <p className="text-sm text-gray-400 mb-3">Esempi di ticket:</p>
              <div className="flex flex-col gap-2">
                {EXAMPLE_TICKETS.map((example, index) => (
                  <button
                    key={index}
                    onClick={() => handleExample(example)}
                    className="text-left text-sm px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors text-gray-300"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </GlassCard>

          {/* ----------------------------------------
              DISPLAY ERROR
              ---------------------------------------- */}
          {error && (
            <GlassCard className="bg-red-500/10 border-red-500/30 animate-fade-in-up">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-semibold text-red-400 mb-1">Errore</h3>
                  <p className="text-gray-300">{error}</p>
                </div>
              </div>
            </GlassCard>
          )}
        </div>

        {/* ========================================
            COLONNA DESTRA: RISULTATI
            ======================================== */}
        <div className="space-y-6">
          {/* Stato vuoto iniziale */}
          {!result && !error && (
            <GlassCard className="text-center py-12 animate-fade-in-up">
              <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-white/5 flex items-center justify-center">
                <TrendingUp className="w-10 h-10 text-gray-500" />
              </div>
              <p className="text-gray-400">
                Inserisci un ticket e clicca su &quot;Analizza&quot; per vedere i risultati
              </p>
            </GlassCard>
          )}

          {/* ----------------------------------------
              RISULTATI CLASSIFICAZIONE
              ---------------------------------------- */}
          {result && (
            <>
              {/* Card principale risultati */}
              <GlassCard className="animate-slide-in-right">
                <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                  <Sparkles className="w-6 h-6 text-green-400" />
                  Risultato Analisi
                </h2>

                {/* Categoria e Priorità */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  {/* Categoria */}
                  <div className="text-center p-6 rounded-xl bg-white/5">
                    <p className="text-sm text-gray-400 mb-2">Categoria</p>
                    <p className="text-3xl font-bold text-white">{result.category}</p>
                  </div>

                  {/* Priorità */}
                  <div className="text-center p-6 rounded-xl bg-white/5">
                    <p className="text-sm text-gray-400 mb-2">Priorità</p>
                    <PriorityBadge priority={result.priority as "Alta" | "Media" | "Bassa"} size="lg" />
                  </div>
                </div>

                {/* Confidenza con progress bar */}
                <div className="mb-6">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-gray-400">Confidenza</span>
                    <span className="text-lg font-semibold text-green-400">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-green-500 transition-all duration-1000"
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>

                {/* Probabilità per classe di priorità */}
                <div className="space-y-3">
                  <p className="text-sm text-gray-400 mb-2">Probabilità Priorità:</p>
                  {Object.entries(result.probabilities).map(([priority, prob]) => (
                    <div key={priority} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-300">{priority}</span>
                        <span className="text-gray-400">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-1000 ${priority === "Alta"
                              ? "bg-red-500"
                              : priority === "Media"
                                ? "bg-orange-500"
                                : "bg-green-500"
                            }`}
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </GlassCard>

              {/* ----------------------------------------
                  SPIEGAZIONE LIME
                  ---------------------------------------- */}
              {result.lime_explanation && result.lime_explanation.length > 0 && (
                <GlassCard className="animate-slide-in-right">
                  <h3 className="text-xl font-bold mb-4">Perché questo risultato?</h3>
                  <p className="text-sm text-gray-400 mb-4">
                    Parole chiave che hanno influenzato la classificazione:
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {result.lime_explanation.map((item, index) => (
                      <span
                        key={index}
                        className={`px-4 py-2 rounded-lg font-medium ${item.type === "trigger"
                            ? "bg-red-500/20 text-red-400 border border-red-500/30"  // Trigger = rosso
                            : "bg-gray-500/20 text-gray-400 border border-gray-500/30" // LIME = grigio
                          }`}
                      >
                        {item.word}
                      </span>
                    ))}
                  </div>
                </GlassCard>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ========================================
// WRAPPER CON PROVIDER
// ========================================
/**
 * Componente wrapper che fornisce il TicketContext.
 * Deve essere radice dell'albero componenti che usano useTicket().
 */
export default function ManualPage() {
  return (
    <TicketProvider>
      <ManualPageContent />
    </TicketProvider>
  );
}
