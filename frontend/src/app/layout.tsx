/**
 * ========================================
 * RootLayout - Layout Principale Applicazione
 * ========================================
 *
 * Layout radice che avvolge tutte le pagine dell'applicazione.
 * Definisce:
 * - Metadata SEO
 * - Font (Inter)
 * - Background nero con griglia, linea e gradient
 * - Struttura HTML
 * - Footer comune
 */

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

// ========================================
// CONFIGURAZIONE FONT
// ========================================
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
            SFONDO NERO CON GRIGLIA, LINEA E GLOW
            ---------------------------------------- */}
        <div className="fixed inset-0 -z-10 overflow-hidden">
          {/* Grid pattern: griglia sottile */}
          <div
            className="absolute inset-0 opacity-[0.03]"
            style={{
              backgroundImage: `linear-gradient(to right, #ffffff 1px, transparent 1px),
                                     linear-gradient(to bottom, #ffffff 1px, transparent 1px)`,
              backgroundSize: '50px 50px',
            }}
          />

          {/* Vertical accent line centrale */}
          <div className="absolute left-1/2 top-0 -translate-x-1/2 h-full w-px bg-gradient-to-b from-zinc-400/30 via-zinc-500/5 to-transparent" />

          {/* Radial glow circles */}
          <div className="absolute top-[20%] left-[20%] w-[600px] h-[600px] bg-zinc-400/20 rounded-full blur-[100px]" />
          <div className="absolute bottom-[20%] right-[20%] w-[500px] h-[500px] bg-zinc-500/15 rounded-full blur-[80px]" />
          <div className="absolute top-[60%] left-[40%] w-[400px] h-[400px] bg-zinc-300/10 rounded-full blur-[60px]" />
        </div>

        {/* ----------------------------------------
            CONTENUTO PRINCIPALE
            ---------------------------------------- */}
        <div className="relative flex-1 flex flex-col">
          <main className="pt-20 pb-8 flex-1 min-h-[calc(100vh-12rem)]">
            {children}
          </main>
        </div>

        {/* ----------------------------------------
            FOOTER
            ---------------------------------------- */}
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
