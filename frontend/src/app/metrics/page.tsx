/**
 * ========================================
 * MetricsPage - Pagina Metriche Modello
 * ========================================
 *
 * Pagina che visualizza le performance del modello:
 * - KPI cards (accuracy, precision, recall, f1)
 * - Grafici interattivi Chart.js
 * - Confusion matrix immagini
 * - Spiegazione metriche
 */

"use client"; // Necessario per fetch dati e interattività

import React, { useState, useEffect } from "react";
import { getMetricsSummary, getMetricsImages } from "@/lib/api";
import { GlassCard } from "@/components/GlassCard";
import { MetricsChart } from "@/components/MetricsChart";
import { NavigationButton } from "@/components/Navigation";
import { Activity, Target, Award, TrendingUp } from "lucide-react";

// ========================================
// INTERFACCIA METRICHE
// ========================================
interface Metrics {
  category: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  };
  priority: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  };
}

export default function MetricsPage() {
  // ----------------------------------------
  // STATI LOCALI
  // ----------------------------------------
  const [metrics, setMetrics] = useState<Metrics | null>(null);      // Metriche numeriche
  const [images, setImages] = useState<Record<string, string>>({});  // Path immagini grafici
  const [isLoading, setIsLoading] = useState(true);                   // Loading state
  const [error, setError] = useState<string | null>(null);           // Error state

  // ----------------------------------------
  // FETCH DATI
  // ----------------------------------------
  useEffect(() => {
    async function loadMetrics() {
      try {
        setIsLoading(true);
        // Fetch parallelo metriche + immagini
        const [metricsData, imagesData] = await Promise.all([
          getMetricsSummary(),
          getMetricsImages(),
        ]);
        setMetrics(metricsData);
        setImages(imagesData);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Errore nel caricamento delle metriche";
        setError(message);
      } finally {
        setIsLoading(false);
      }
    }

    loadMetrics();
  }, []);

  // ========================================
  // LOADING STATE
  // ========================================
  if (isLoading) {
    return (
      <div className="container-custom py-6">
        <div className="text-center">
          {/* Spinner animato */}
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          <p className="mt-4 text-gray-400">Caricamento metriche...</p>
        </div>
      </div>
    );
  }

  // ========================================
  // ERROR STATE
  // ========================================
  if (error || !metrics) {
    return (
      <div className="container-custom py-6">
        <GlassCard className="max-w-2xl mx-auto text-center">
          <p className="text-red-400">{error || "Impossibile caricare le metriche"}</p>
        </GlassCard>
      </div>
    );
  }

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
          Metriche del Modello
        </h1>
        <p className="text-lg text-gray-400">
          Performance del modello di classificazione su test set
        </p>
      </div>

      {/* ----------------------------------------
          KPI CARDS
          ---------------------------------------- */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {/* Accuracy Categoria */}
        <GlassCard className="animate-slide-in-left">
          <div className="flex items-center justify-between mb-4">
            <Activity className="w-10 h-10 text-blue-400" />
            <span className="text-3xl font-bold text-blue-400">
              {(metrics.category.accuracy * 100).toFixed(0)}%
            </span>
          </div>
          <p className="text-sm text-gray-400">Accuracy Categoria</p>
        </GlassCard>

        {/* Precision Categoria */}
        <GlassCard className="animate-slide-in-left">
          <div className="flex items-center justify-between mb-4">
            <Target className="w-10 h-10 text-purple-400" />
            <span className="text-3xl font-bold text-purple-400">
              {(metrics.category.precision * 100).toFixed(0)}%
            </span>
          </div>
          <p className="text-sm text-gray-400">Precision Categoria</p>
        </GlassCard>

        {/* F1-Score Categoria */}
        <GlassCard className="animate-slide-in-right">
          <div className="flex items-center justify-between mb-4">
            <Award className="w-10 h-10 text-green-400" />
            <span className="text-3xl font-bold text-green-400">
              {(metrics.category.f1 * 100).toFixed(0)}%
            </span>
          </div>
          <p className="text-sm text-gray-400">F1-Score Categoria</p>
        </GlassCard>

        {/* Accuracy Priorità */}
        <GlassCard className="animate-slide-in-right">
          <div className="flex items-center justify-between mb-4">
            <TrendingUp className="w-10 h-10 text-orange-400" />
            <span className="text-3xl font-bold text-orange-400">
              {(metrics.priority.accuracy * 100).toFixed(0)}%
            </span>
          </div>
          <p className="text-sm text-gray-400">Accuracy Priorità</p>
        </GlassCard>
      </div>

      {/* ----------------------------------------
          CHARTS
          ---------------------------------------- */}
      <div className="mb-8">
        <MetricsChart metrics={metrics} />
      </div>

      {/* ----------------------------------------
          CONFUSION MATRIX IMAGES
          ---------------------------------------- */}
      {Object.keys(images).length > 0 && (
        <div className="animate-fade-in-up">
          <h2 className="text-2xl font-bold mb-6">Confusion Matrix</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Matrice confusione Categoria */}
            {images.confusion_matrix_category && (
              <GlassCard>
                <h3 className="text-lg font-semibold mb-4">Categoria</h3>
                <img
                  src={images.confusion_matrix_category.replace("../assets", "/assets")}
                  alt="Confusion Matrix Categoria"
                  className="w-full rounded-lg"
                />
              </GlassCard>
            )}
            {/* Matrice confusione Priorità */}
            {images.confusion_matrix_priority && (
              <GlassCard>
                <h3 className="text-lg font-semibold mb-4">Priorità</h3>
                <img
                  src={images.confusion_matrix_priority.replace("../assets", "/assets")}
                  alt="Confusion Matrix Priorità"
                  className="w-full rounded-lg"
                />
              </GlassCard>
            )}
          </div>
        </div>
      )}

      {/* ----------------------------------------
          SPIEGAZIONE METRICHE
          ---------------------------------------- */}
      <GlassCard className="mt-12">
        <h3 className="text-xl font-bold mb-4">Come leggere questi dati</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-gray-300">
          <div>
            <p className="font-semibold text-white mb-2">📊 Confusion Matrix</p>
            <p>Visualizza gli errori di smistamento. La diagonale rappresenta le previsioni corrette.</p>
          </div>
          <div>
            <p className="font-semibold text-white mb-2">🎯 Accuracy</p>
            <p>Percentuale complessiva di classificazioni esatte.</p>
          </div>
          <div>
            <p className="font-semibold text-white mb-2">🔍 Precision</p>
            <p>Affidabilità del modello quando assegna una determinata categoria.</p>
          </div>
          <div>
            <p className="font-semibold text-white mb-2">📈 Recall</p>
            <p>Capacità del modello di non perdere ticket di una certa categoria.</p>
          </div>
          <div>
            <p className="font-semibold text-white mb-2">⚖️ F1-Score</p>
            <p>Media bilanciata tra Precision e Recall. Metrica complessiva del modello.</p>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}
