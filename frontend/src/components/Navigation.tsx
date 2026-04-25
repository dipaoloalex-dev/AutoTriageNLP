/**
 * ========================================
 * NavigationButton - Pulsante "Torna Indietro"
 * ========================================
 *
 * Componente pulsante per navigare alla pagina precedente.
 * Utilizzato nelle pagine interne per tornare alla home.
 *
 * Nota: Il file si chiama Navigation.tsx ma esporta NavigationButton
 */

"use client"; // Necessario per useRouter

import { useRouter } from "next/navigation";
import { ArrowLeft } from "lucide-react";

export function NavigationButton() {
  // Hook di Next.js per la navigazione programmatica
  const router = useRouter();

  return (
    <button
      // Torna alla pagina precedente nella history del browser
      onClick={() => router.back()}
      // Styling glassmorphism + animazioni
      className="inline-flex items-center gap-2 glass-button px-3 py-2 rounded-lg hover:scale-105 transition-all duration-300 group mb-4"
      // Accessibilità: label per screen reader
      aria-label="Torna indietro"
      title="Torna indietro"
    >
      {/* Icona freccia sinistra con micro-animazione al hover */}
      <ArrowLeft size={18} className="group-hover:-translate-x-0.5 transition-transform" />
    </button>
  );
}
