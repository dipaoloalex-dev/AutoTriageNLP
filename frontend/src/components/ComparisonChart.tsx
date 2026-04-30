/**
 * ========================================
 * ComparisonChart - Grafici Confronto Modelli
 * ========================================
 *
 * Componente che visualizza il confronto tra Modello A e Modello B usando Chart.js
 * Stile coerente con MetricsChart
 */

"use client";

import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import { GlassCard } from "./GlassCard";

// ========================================
// REGISTRAZIONE COMPONENTI CHART.JS
// ========================================
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// ========================================
// DATI DEL CONFRONTO
// ========================================
const comparisonData = {
  labels: ["Modello A", "Modello B"],
  datasets: [
    {
      label: "Accuracy",
      data: [50.9, 58.96],
      backgroundColor: [
        "rgba(239, 68, 68, 0.7)",   // Rosso per Modello A
        "rgba(34, 197, 94, 0.7)",   // Verde per Modello B
      ],
      borderColor: [
        "rgba(239, 68, 68, 1)",
        "rgba(34, 197, 94, 1)",
      ],
      borderWidth: 2,
      borderRadius: 8,
    },
  ],
};

// Opzioni grafico
const comparisonOptions = {
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
        label: (context: any) => `Accuracy: ${context.raw.toFixed(1)}%`,
      },
    },
  },
  scales: {
    x: {
      grid: { color: "rgba(255, 255, 255, 0.05)" },
      ticks: {
        color: "rgba(255, 255, 255, 0.7)",
        font: { size: 13 },
      },
    },
    y: {
      min: 0,
      max: 100,
      grid: { color: "rgba(255, 255, 255, 0.05)" },
      ticks: {
        color: "rgba(255, 255, 255, 0.7)",
        callback: (value: any) => `${value}%`,
      },
    },
  },
};

// ========================================
// COMPONENTE
// ========================================
export function ComparisonChart() {
  return (
    <GlassCard>
      <h3 className="text-xl font-bold mb-4">Confronto Capacità di Generalizzazione</h3>
      <div className="h-[300px]">
        <Bar data={comparisonData} options={comparisonOptions} />
      </div>
    </GlassCard>
  );
}
