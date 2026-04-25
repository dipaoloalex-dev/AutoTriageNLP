/**
 * ========================================
 * PriorityBadge - Badge Priorità Ticket
 * ========================================
 *
 * Componente badge per visualizzare la priorità di un ticket.
 * Supporta 3 livelli: Alta (rossa), Media (arancione), Bassa (verde).
 *
 * Include animazione pulsante per la priorità Alta.
 */

import React from "react";
import { cn } from "@/lib/utils";
import { PRIORITY_CONFIG } from "@/lib/constants";

// ========================================
// INTERFACCIA PROPRIETÀ
// ========================================
interface PriorityBadgeProps {
  priority: "Alta" | "Media" | "Bassa";  // Livello priorità
  size?: "sm" | "md" | "lg";             // Dimensione badge
  showIcon?: boolean;                    // Mostra emoji icona
}

export function PriorityBadge({ priority, size = "md", showIcon = true }: PriorityBadgeProps) {
  // Recupero configurazione per la priorità specificata (colori, icona, ecc.)
  const config = PRIORITY_CONFIG[priority];

  // ----------------------------------------
  // CLASSI SIZE
  // ----------------------------------------
  const sizeClasses = {
    sm: "px-3 py-1 text-sm",      // Small: compatto
    md: "px-4 py-2 text-base",    // Medium: default
    lg: "px-6 py-3 text-lg",      // Large: prominente
  };

  // ----------------------------------------
  // RENDER
  // ----------------------------------------
  return (
    <span
      // Merge classi base + config colori + size + animazione Alta
      className={cn(
        "inline-flex items-center gap-2 rounded-full border font-semibold transition-all",
        config.bgClass,       // Sfondo (es: bg-red-500/20)
        config.textClass,     // Testo (es: text-red-400)
        config.borderClass,   // Bordo (es: border-red-500/30)
        sizeClasses[size],
        // Animazione pulsante solo per priorità Alta (attenzione!)
        priority === "Alta" && "animate-pulse-glow"
      )}
    >
      {/* Icona emoji se showIcon=true */}
      {showIcon && <span>{config.icon}</span>}
      {/* Testo della priorità */}
      {priority}
    </span>
  );
}
