/**
 * ========================================
 * RootLayout - Layout Principale Applicazione
 * ========================================
 *
 * Layout radice che avvolge tutte le pagine dell'applicazione.
 * Definisce:
 * - Metadata SEO
 * - Font (Inter)
 * - Background effects
 * - Struttura HTML
 * - Footer comune
 */

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { DevIndicatorRemover } from "@/components/DevIndicatorRemover";

// ========================================
// CONFIGURAZIONE FONT
// ========================================
// Font Inter di Google Fonts (ottimizzato per UI)
const inter = Inter({ subsets: ["latin"] });

// ========================================
// METADATA SEO
// ========================================
export const metadata: Metadata = {
  title: "AutoTriage NLP - Advanced Ticket Classification",
  description: "Sistema intelligente di classificazione e prioritarizzazione ticket per l'assistenza aziendale",
};

// ========================================
// COMPONENTE LAYOUT
// ========================================
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="it" className="dark">
      <body className={inter.className + " min-h-screen flex flex-col"}>
        {/* ----------------------------------------
            COMPONENTE: Rimozione Indicatori Dev
            ----------------------------------------
            Rimuove aggressivamente tutti gli indicatori
            di sviluppo che Next.js inietta durante il dev mode.
          */}
        <DevIndicatorRemover />

        {/* ----------------------------------------
            SFONDO ANIMATO
            ----------------------------------------
            Layer con effetti visivi di sfondo:
            - Gradient overlay
            - Grid pattern
            - Glow effects
          */}
        <div className="fixed inset-0 -z-10 overflow-hidden">
          {/* Gradient overlay: sfumatura blu-viola */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 via-purple-900/20 to-transparent" />

          {/* Grid pattern: griglia sottile */}
          <div
            className="absolute inset-0 opacity-[0.03]"
            style={{
              backgroundImage: `linear-gradient(to right, #ffffff 1px, transparent 1px),
                                     linear-gradient(to bottom, #ffffff 1px, transparent 1px)`,
              backgroundSize: '50px 50px',
            }}
          />

          {/* Glow effects: cerchi sfumati decorativi */}
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
        </div>

        {/* ----------------------------------------
            CONTENUTO PRINCIPALE
            ---------------------------------------- */}
        <div className="relative flex-1 flex flex-col">
          <main className="pt-20 pb-8 flex-1 min-h-[calc(100vh-12rem)]">
            {/* children = pagina corrente (Next.js 13+ App Router) */}
            {children}
          </main>
        </div>

        {/* ----------------------------------------
            FOOTER
            ----------------------------------------
            Footer comune a tutte le pagine con info autore.
          */}
        <footer className="relative border-t border-white/10 py-6 flex-shrink-0">
          <div className="container-custom text-center text-sm text-gray-400">
            <p>© 2026 AutoTriage NLP - Project Work Universitario</p>
            <p className="mt-1">Informatica per le Aziende Digitali (L-31) - Alex Di Paolo</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
