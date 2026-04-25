/**
 * ========================================
 * BatchPage - Pagina Importazione CSV
 * ========================================
 *
 * Pagina per l'analisi batch di ticket tramite file CSV.
 * Permette di:
 * - Trascinare o selezionare un file CSV
 * - Visualizzare risultati dettagliati
 * - Vedere riepilogo statistiche
 *
 * Il CSV deve avere una colonna 'text', 'body', 'testo' o 'descrizione'.
 */

"use client"; // Necessario per drag & drop e interattività

import React, { useState, useCallback } from "react";
import { uploadCSV } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { GlassCard } from "@/components/GlassCard";
import { PriorityBadge } from "@/components/PriorityBadge";
import { NavigationButton } from "@/components/Navigation";
import { Upload, FileText, CheckCircle, AlertCircle } from "lucide-react";

// ========================================
// INTERFACCE DATI
// ========================================
/**
 * Risultato della classificazione di un singolo ticket.
 */
interface BatchResult {
  description: string;    // Testo del ticket
  category: string;       // Categoria assegnata
  priority: string;       // Priorità assegnata
  confidence: number;     // Confidenza (0-1)
}

/**
 * Riepilogo statistiche dell'intero batch.
 */
interface Summary {
  total_processed: number;  // Totale ticket processati
  high_priority: number;    // Ticket priorità Alta
  success: boolean;         // True se operazione riuscita
}

export default function BatchPage() {
  // ========================================
  // STATI LOCALI
  // ========================================
  const [file, setFile] = useState<File | null>(null);           // File selezionato
  const [isDragging, setIsDragging] = useState(false);          // Stato drag & drop
  const [isUploading, setIsUploading] = useState(false);        // Upload in corso
  const [results, setResults] = useState<BatchResult[]>([]);   // Risultati classificazione
  const [summary, setSummary] = useState<Summary | null>(null); // Riepilogo statistiche
  const [error, setError] = useState<string | null>(null);      // Messaggio errore

  // ========================================
  // HANDLERS DRAG & DROP
  // ========================================
  /**
   * Gestisce l'evento drag (prevenire default).
   */
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  /**
   * Gestisce l'entrata nell'area di drag.
   */
  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);  // Attiva stato dragging
    }
  }, []);

  /**
   * Gestisce l'uscita dall'area di drag.
   */
  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);  // Disattiva stato dragging
  }, []);

  /**
   * Gestisce il rilascio del file (drop).
   * Valida che sia un CSV prima di accettarlo.
   */
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);  // Reset stato dragging

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      // Validazione: deve essere CSV
      if (droppedFile.type === "text/csv" || droppedFile.name.endsWith(".csv")) {
        setFile(droppedFile);
        setError(null);
      } else {
        setError("Per favore carica solo file CSV");
      }
    }
  }, []);

  // ========================================
  // HANDLERS FILE SELECT
  // ========================================
  /**
   * Gestisce la selezione file tramite input.
   */
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  // ========================================
  // HANDLERS UPLOAD
  // ========================================
  /**
   * Esegue l'upload del CSV al backend.
   */
  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setError(null);

    try {
      const response = await uploadCSV(file);
      setResults(response.results || []);
      setSummary(response.summary);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Errore durante l'upload";
      setError(message);
      setResults([]);
      setSummary(null);
    } finally {
      setIsUploading(false);
    }
  };

  /**
   * Reset della pagina per nuovo upload.
   */
  const handleReset = () => {
    setFile(null);
    setResults([]);
    setSummary(null);
    setError(null);
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
        <h1 className="text-3xl md:text-4xl font-bold mb-4">
          <span className="text-gradient">Importazione CSV</span>
        </h1>
        <p className="text-lg text-gray-400">
          Carica un file CSV per analizzare gruppi di ticket in un&apos;unica operazione
        </p>
      </div>

      <div className="max-w-6xl mx-auto">
        {/* ========================================
            UPLOAD AREA
            ======================================== */}
        {!results.length && (
          <GlassCard className="animate-slide-in-left">
            {/* Area drag & drop */}
            <div
              onDragEnter={handleDragIn}
              onDragLeave={handleDragOut}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              // Styling dinamico base stato dragging
              className={`border-2 border-dashed rounded-xl p-12 text-center transition-all ${
                isDragging
                  ? "border-blue-500 bg-blue-500/10"          // Stato dragging
                  : "border-white/20 hover:border-white/40"  // Stato normale
              }`}
            >
              <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
              <h3 className="text-xl font-semibold mb-2">
                {isDragging ? "Rilascia il file CSV" : "Trascina qui il file CSV"}
              </h3>
              <p className="text-gray-400 mb-6">oppure</p>

              {/* Input file nascosto con label stilizzato */}
              <label className="inline-block cursor-pointer">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <span className="glass-button px-6 py-3 rounded-lg inline-block">
                  Sfoglia File
                </span>
              </label>

              <p className="text-sm text-gray-500 mt-4">
                Il file CSV deve avere una colonna denominata &apos;text&apos;, &apos;body&apos;, &apos;testo&apos; o &apos;descrizione&apos;
              </p>
            </div>

            {/* ----------------------------------------
                FILE SELEZIONATO
                ---------------------------------------- */}
            {file && (
              <div className="mt-6 p-4 rounded-lg bg-white/5 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <FileText className="w-8 h-8 text-blue-400" />
                  <div>
                    <p className="font-medium">{file.name}</p>
                    <p className="text-sm text-gray-400">
                      {(file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
                <Button onClick={handleUpload} isLoading={isUploading}>
                  Analizza
                </Button>
              </div>
            )}

            {/* ----------------------------------------
                DISPLAY ERROR
                ---------------------------------------- */}
            {error && (
              <div className="mt-6 p-4 rounded-lg bg-red-500/10 border border-red-500/30 flex items-start gap-3">
                <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-1" />
                <div>
                  <p className="font-semibold text-red-400">Errore</p>
                  <p className="text-gray-300">{error}</p>
                </div>
              </div>
            )}
          </GlassCard>
        )}

        {/* ========================================
            RESULTS DISPLAY
            ======================================== */}
        {results.length > 0 && summary && (
          <>
            {/* ----------------------------------------
                SUMMARY CARD
                ---------------------------------------- */}
            <GlassCard className="mb-8 animate-fade-in-up">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <CheckCircle className="w-6 h-6 text-green-400" />
                  Analisi Completata
                </h2>
                <Button variant="outline" onClick={handleReset}>
                  Carica Altro
                </Button>
              </div>

              {/* KPI Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {/* Ticket Totali */}
                <div className="text-center p-4 rounded-lg bg-white/5">
                  <p className="text-3xl font-bold text-blue-400">{summary.total_processed}</p>
                  <p className="text-sm text-gray-400 mt-1">Ticket Analizzati</p>
                </div>

                {/* Priorità Alta */}
                <div className="text-center p-4 rounded-lg bg-red-500/10 border border-red-500/30">
                  <p className="text-3xl font-bold text-red-400">{summary.high_priority}</p>
                  <p className="text-sm text-gray-400 mt-1">Priorità Alta</p>
                </div>

                {/* Priorità Media (calcolata) */}
                <div className="text-center p-4 rounded-lg bg-orange-500/10 border border-orange-500/30">
                  <p className="text-3xl font-bold text-orange-400">
                    {results.filter(r => r.priority === "Media").length}
                  </p>
                  <p className="text-sm text-gray-400 mt-1">Priorità Media</p>
                </div>

                {/* Priorità Bassa (calcolata) */}
                <div className="text-center p-4 rounded-lg bg-green-500/10 border border-green-500/30">
                  <p className="text-3xl font-bold text-green-400">
                    {results.filter(r => r.priority === "Bassa").length}
                  </p>
                  <p className="text-sm text-gray-400 mt-1">Priorità Bassa</p>
                </div>
              </div>
            </GlassCard>

            {/* ----------------------------------------
                RESULTS TABLE
                ---------------------------------------- */}
            <GlassCard className="animate-slide-in-up">
              <h3 className="text-xl font-bold mb-4">Risultati Dettagliati</h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">
                        Descrizione
                      </th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">
                        Categoria
                      </th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">
                        Priorità
                      </th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">
                        Confidenza
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((result, index) => (
                      <tr
                        key={index}
                        className="border-b border-white/5 hover:bg-white/5 transition-colors"
                      >
                        {/* Descrizione */}
                        <td className="py-3 px-4 text-sm">
                          {result.description}
                        </td>

                        {/* Categoria */}
                        <td className="py-3 px-4">
                          <span className="px-3 py-1 rounded-full bg-blue-500/20 text-blue-400 text-sm font-medium">
                            {result.category}
                          </span>
                        </td>

                        {/* Priorità */}
                        <td className="py-3 px-4">
                          <PriorityBadge
                            priority={result.priority as "Alta" | "Media" | "Bassa"}
                            size="sm"
                          />
                        </td>

                        {/* Confidenza con color coding */}
                        <td className="py-3 px-4 text-sm">
                          <span className={`font-medium ${
                            result.confidence > 0.8
                              ? "text-green-400"
                              : result.confidence > 0.6
                              ? "text-yellow-400"
                              : "text-red-400"
                          }`}>
                            {(result.confidence * 100).toFixed(1)}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Note truncation results */}
              {summary.total_processed > results.length && (
                <p className="text-sm text-gray-400 mt-4 text-center">
                  Mostrati i primi {results.length} risultati di {summary.total_processed} totali
                </p>
              )}
            </GlassCard>
          </>
        )}
      </div>
    </div>
  );
}
