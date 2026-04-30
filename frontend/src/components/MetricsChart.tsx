/**
 * ========================================
 * MetricsChart - Grafici Metriche Modello
 * ========================================
 *
 * Componente che visualizza le metriche del modello usando Chart.js:
 * - Bar chart: confronto metriche Categoria vs Priorità
 * - Doughnut chart: visualizzazione accuracy percentuale
 *
 * Usa react-chartjs-2 come wrapper per Chart.js
 */

"use client"; // Necessario per grafici interattivi

import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from "chart.js";
import { Bar, Doughnut } from "react-chartjs-2";
import { GlassCard } from "./GlassCard";

// ========================================
// REGISTRAZIONE COMPONENTI CHART.JS
// ========================================
// Registro i componenti necessari per i grafici
ChartJS.register(
  CategoryScale,   // Asse X categorico
  LinearScale,     // Asse Y lineare
  BarElement,      // Elementi barre
  Title,           // Titoli grafico
  Tooltip,         // Tooltip al hover
  Legend,          // Legenda
  ArcElement       // Elementi arco (per doughnut)
);

// ========================================
// INTERFACCIA PROPRIETÀ
// ========================================
interface MetricsChartProps {
  metrics: {
    category: { accuracy: number; precision: number; recall: number; f1: number };
    priority: { accuracy: number; precision: number; recall: number; f1: number };
  };
}

export function MetricsChart({ metrics }: MetricsChartProps) {
  // ========================================
  // DATI BAR CHART
  // ========================================
  const barData = {
    labels: ["Accuracy", "Precision", "Recall", "F1-Score"],
    datasets: [
      {
        label: "Categoria",
        data: [
          metrics.category.accuracy * 100,
          metrics.category.precision * 100,
          metrics.category.recall * 100,
          metrics.category.f1 * 100,
        ],
        backgroundColor: "rgba(59, 130, 246, 0.7)",    // Blu
        borderColor: "rgba(59, 130, 246, 1)",
        borderWidth: 2,
        borderRadius: 8,
      },
      {
        label: "Priorità",
        data: [
          metrics.priority.accuracy * 100,
          metrics.priority.precision * 100,
          metrics.priority.recall * 100,
          metrics.priority.f1 * 100,
        ],
        backgroundColor: "rgba(168, 85, 247, 0.7)",    // Viola
        borderColor: "rgba(168, 85, 247, 1)",
        borderWidth: 2,
        borderRadius: 8,
      },
    ],
  };

  // Opzioni bar chart
  const barOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top" as const,
        labels: {
          color: "rgba(255, 255, 255, 0.8)",
          font: { size: 14 },
        },
      },
      tooltip: {
        backgroundColor: "rgba(0, 0, 0, 0.9)",
        titleColor: "white",
        bodyColor: "white",
        borderColor: "rgba(255, 255, 255, 0.2)",
        borderWidth: 1,
        padding: 12,
        displayColors: true,
        callbacks: {
          label: (context: any) => `${context.dataset.label}: ${context.raw.toFixed(1)}%`,
        },
      },
    },
    scales: {
      x: {
        grid: { color: "rgba(255, 255, 255, 0.05)" },
        ticks: { color: "rgba(255, 255, 255, 0.7)" },
      },
      y: {
        min: 0,
        max: 100,  // Scala 0-100%
        grid: { color: "rgba(255, 255, 255, 0.05)" },
        ticks: {
          color: "rgba(255, 255, 255, 0.7)",
          callback: (value: any) => `${value}%`,
        },
      },
    },
  };

  // ========================================
  // DATI DOUGHNUT CHART CATEGORIA
  // ========================================
  const doughnutCategoryData = {
    labels: ["Corretti", "Errati"],
    datasets: [
      {
        data: [
          metrics.category.accuracy * 100,
          (1 - metrics.category.accuracy) * 100,
        ],
        backgroundColor: ["rgba(59, 130, 246, 0.8)", "rgba(168, 85, 247, 0.7)"],
        borderColor: ["rgba(59, 130, 246, 1)", "rgba(168, 85, 247, 1)"],
        borderWidth: 2,
      },
    ],
  };

  // ========================================
  // DATI DOUGHNUT CHART PRIORITÀ
  // ========================================
  const doughnutPriorityData = {
    labels: ["Corretti", "Errati"],
    datasets: [
      {
        data: [
          metrics.priority.accuracy * 100,
          (1 - metrics.priority.accuracy) * 100,
        ],
        backgroundColor: ["rgba(249, 115, 22, 0.8)", "rgba(168, 85, 247, 0.7)"],
        borderColor: ["rgba(249, 115, 22, 1)", "rgba(168, 85, 247, 1)"],
        borderWidth: 2,
      },
    ],
  };

  // Opzioni doughnut chart
  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "bottom" as const,
        labels: {
          color: "rgba(255, 255, 255, 0.8)",
          font: { size: 12 },
        },
      },
      tooltip: {
        backgroundColor: "rgba(0, 0, 0, 0.9)",
        callbacks: {
          label: (context: any) => `${context.label}: ${context.raw.toFixed(1)}%`,
        },
      },
    },
  };

  // ========================================
  // RENDER
  // ========================================
  return (
    <div className="space-y-6">
      {/* ========================================
          BAR CHART
          ======================================== */}
      <GlassCard>
        <h3 className="text-xl font-bold mb-4">Metriche Complete</h3>
        <div className="h-[350px]">
          <Bar data={barData} options={barOptions} />
        </div>
      </GlassCard>

      {/* ========================================
          DOUGHNUT CHARTS (2 affiancati)
          ======================================== */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Doughnut Categoria */}
        <GlassCard>
          <h3 className="text-lg font-bold mb-4">Accuracy Categoria</h3>
          <div className="h-[250px] flex items-center justify-center">
            <Doughnut data={doughnutCategoryData} options={doughnutOptions} />
          </div>
          <div className="text-center mt-4">
            <p className="text-4xl font-bold text-blue-400">
              {(metrics.category.accuracy * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-gray-400 mt-1">Accuracy Complessiva</p>
          </div>
        </GlassCard>

        {/* Doughnut Priorità */}
        <GlassCard>
          <h3 className="text-lg font-bold mb-4">Accuracy Priorità</h3>
          <div className="h-[250px] flex items-center justify-center">
            <Doughnut data={doughnutPriorityData} options={doughnutOptions} />
          </div>
          <div className="text-center mt-4">
            <p className="text-4xl font-bold text-orange-400">
              {(metrics.priority.accuracy * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-gray-400 mt-1">Accuracy Complessiva</p>
          </div>
        </GlassCard>
      </div>
    </div>
  );
}
