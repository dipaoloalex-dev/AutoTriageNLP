/**
 * ========================================
 * TextArea - Componente Textarea Glassmorphism
 * ========================================
 *
 * Componente textarea riutilizzabile con:
 * - Etichetta opzionale
 * - Gestione errori
 * - Contatore caratteri
 * - Styling glassmorphism
 */

import React from "react";
import { cn } from "@/lib/utils";

// ========================================
// INTERFACCIA PROPRIETÀ
// ========================================
interface TextAreaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;           // Etichetta sopra la textarea
  error?: string;           // Messaggio di errore da mostrare
  characterCount?: number;  // Numero caratteri attuali
  maxLength?: number;       // Massimo numero di caratteri
}

export function TextArea({
  label,
  error,
  characterCount,
  maxLength,
  className,
  ...props
}: TextAreaProps) {
  // ----------------------------------------
  // RENDER
  // ----------------------------------------
  return (
    <div className="w-full">
      {/* ----------------------------------------
          ETICHETTA (opzionale)
          ---------------------------------------- */}
      {label && (
        <label className="block text-sm font-medium text-gray-300 mb-2">
          {label}
        </label>
      )}

      {/* ----------------------------------------
          TEXTAREA PRINCIPALE
          ---------------------------------------- */}
      <textarea
        // Merge classi: glass-input (stile) + condizionale errore + custom className
        className={cn(
          "glass-input w-full rounded-lg px-4 py-3 min-h-[120px] resize-none",
          error && "border-red-500/50",  // Bordo rosso se c'è errore
          className
        )}
        maxLength={maxLength}
        {...props}
      />

      {/* ----------------------------------------
          AREA INFORMATIVA (errore o contatore)
          ---------------------------------------- */}
      <div className="flex justify-between mt-2 text-sm">
        {/* Messaggio di errore (sinistra) */}
        {error && <span className="text-red-400">{error}</span>}

        {/* Contatore caratteri (destra) */}
        {characterCount !== undefined && maxLength && (
          <span className={cn(
            "text-gray-400",                                 // Grigio (normale)
            characterCount >= maxLength * 0.9 && "text-orange-400",  // Arancione (90%++)
            characterCount >= maxLength && "text-red-400"      // Rosso (al limite)
          )}>
            {characterCount}/{maxLength}
          </span>
        )}
      </div>
    </div>
  );
}
