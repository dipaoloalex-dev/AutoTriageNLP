/**
 * ========================================
 * GlassCard - Componente Card Glassmorphism
 * ========================================
 *
 * Componente contenitore base con effetto vetro (glassmorphism).
 * Utilizzato per tutti i pannelli UI dell'applicazione.
 *
 * Lo stile è definito in globals.css (.glass-card)
 */

import React from "react";
import { cn } from "@/lib/utils";

// ========================================
// INTERFACCIA PROPRIETÀ
// ========================================
interface GlassCardProps {
  children: React.ReactNode;  // Contenuto della card
  className?: string;         // Classi CSS aggiuntive
  hover?: boolean;            // Abilita effetto hover (scale)
  onClick?: () => void;       // Click handler (rende la card cliccabile)
}

export function GlassCard({ children, className, hover = true, onClick }: GlassCardProps) {
  // ----------------------------------------
  // RENDER
  // ----------------------------------------
  return (
    <div
      // Merge classi:
      // - glass-card p-6: stile base + padding
      // - hover:scale-[1.02]: ingrandimento al passaggio mouse
      // - cursor-pointer: se c'è onClick, mostra il pointer
      className={cn(
        "glass-card p-6",
        hover && "hover:scale-[1.02]",
        onClick && "cursor-pointer",
        className
      )}
      onClick={onClick}
    >
      {children}
    </div>
  );
}
