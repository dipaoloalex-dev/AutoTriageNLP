/**
 * ========================================
 * HomePage - Pagina Principale
 * ========================================
 *
 * Landing page dell'applicazione con:
 * - Hero section con titolo
 * - Grid di cards per le funzionalità
 * - Tech stack showcase
 */

import Link from "next/link";
import { FileText, Upload, BarChart3, GitCompare } from "lucide-react";
import { GlassCard } from "@/components/GlassCard";

export default function HomePage() {
  // ----------------------------------------
    // DATI FUNZIONALITÀ
    // ----------------------------------------
    // Array delle feature principali dell'applicazione
    // Ogni feature ha: titolo, descrizione, icona, link, colore
  const features = [
    {
      title: "Inserimento Manuale",
      description: "Analizza singoli ticket con classificazione in tempo reale e spiegazioni LIME",
      icon: FileText,
      href: "/manual",
      color: "from-blue-500 to-cyan-500",
    },
    {
      title: "Importazione CSV",
      description: "Carica file CSV per analizzare gruppi di ticket in un'unica operazione",
      icon: Upload,
      href: "/batch",
      color: "from-purple-500 to-pink-500",
    },
    {
      title: "Metriche Modello",
      description: "Visualizza le performance del modello con grafici interattivi",
      icon: BarChart3,
      href: "/metrics",
      color: "from-green-500 to-emerald-500",
    },
    {
      title: "Confronto Dati",
      description: "Analisi comparativa tra dati sintetici e reali",
      icon: GitCompare,
      href: "/comparison",
      color: "from-orange-500 to-red-500",
    },
  ];

  return (
    <div className="container-custom py-12">
      {/* ========================================
          HERO SECTION
          ======================================== */}
      <div className="text-center mb-12 animate-fade-in-up">
        <h1 className="text-4xl md:text-6xl font-bold mb-6 text-white">
          AutoTriage NLP
        </h1>
        <p className="text-lg md:text-xl text-gray-300 max-w-3xl mx-auto">
          Sistema intelligente di classificazione e prioritarizzazione ticket
          per l&apos;assistenza aziendale
        </p>
      </div>

      {/* ========================================
          FEATURE CARDS GRID
          ======================================== */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-5xl mx-auto mb-12">
        {features.map((feature, index) => {
          // Estraggo il componente Icona dall'oggetto
          const Icon = feature.icon;
          return (
            <Link
              key={feature.href}
              href={feature.href}
              className="glass-card p-6 group cursor-pointer"
              // Animation delay in base all'index per effetto cascata
              style={{
                animationDelay: `${index * 0.1}s`,
                animation: 'fadeInUp 0.6s ease-out both',
              }}
            >
              {/* Icona sfumata gradient */}
              <div
                className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.color} p-3 mb-4 group-hover:scale-110 transition-transform`}
              >
                <Icon className="w-full h-full text-white" />
              </div>
              {/* Titolo con hover effect */}
              <h3 className="text-xl font-bold mb-2 group-hover:text-white transition-colors text-gray-200">
                {feature.title}
              </h3>
              <p className="text-gray-400 text-sm">
                {feature.description}
              </p>
            </Link>
          );
        })}
      </div>

      {/* ========================================
          TECH STACK SECTION
          ======================================== */}
      <GlassCard className="p-8 max-w-4xl mx-auto animate-fade-in-up">
        <h2 className="text-2xl font-bold mb-6 text-center">Stack Tecnologico</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
          {/* Next.js */}
          <div>
            <div className="text-4xl mb-2">⚛️</div>
            <div className="text-sm text-gray-400">Next.js 15</div>
            <div className="text-xs text-gray-500">React Framework</div>
          </div>
          {/* FastAPI */}
          <div>
            <div className="text-4xl mb-2">⚡</div>
            <div className="text-sm text-gray-400">FastAPI</div>
            <div className="text-xs text-gray-500">Python Backend</div>
          </div>
          {/* Scikit-Learn */}
          <div>
            <div className="text-4xl mb-2">🤖</div>
            <div className="text-sm text-gray-400">Scikit-Learn</div>
            <div className="text-xs text-gray-500">Machine Learning</div>
          </div>
          {/* LIME + Chart.js */}
          <div>
            <div className="text-4xl mb-2">📊</div>
            <div className="text-sm text-gray-400">LIME + Chart.js</div>
            <div className="text-xs text-gray-500">Explainability & UI</div>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}
