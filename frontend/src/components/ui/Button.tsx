/**
 * ========================================
 * Button - Componente Bottone Glassmorphism
 * ========================================
 *
 * Componente bottone riutilizzabile con stile glassmorphism.
 * Supporta varianti, dimensioni e stato di caricamento.
 */

import React from "react";
import { cn } from "@/lib/utils";

// ========================================
// INTERFACCIA PROPRIETÀ
// ========================================
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline";  // Stile visivo del bottone
  size?: "sm" | "md" | "lg";                       // Dimensione del bottone
  isLoading?: boolean;                             // Mostra spinner di caricamento
}

export function Button({
  children,
  variant = "primary",
  size = "md",
  isLoading = false,
  className,
  disabled,
  ...props
}: ButtonProps) {
  // ----------------------------------------
  // CLASSI BASE
  // ----------------------------------------
  // Classi comuni a tutte le varianti e dimensioni
  const baseClasses = "rounded-lg font-medium transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed";

  // ----------------------------------------
  // CLASSI VARIANT
  // ----------------------------------------
  // Definisce lo stile visivo in base alla variante scelta
  const variantClasses = {
    primary: "glass-button",                    // Effetto vetro blu (definito in globals.css)
    secondary: "bg-white/10 hover:bg-white/20 border border-white/20",  // Sfondo semitrasparente
    outline: "bg-transparent border border-white/30 hover:bg-white/10",  // Solo bordo
  };

  // ----------------------------------------
  // CLASSI SIZE
  // ----------------------------------------
  // Definisce la dimensione (padding e testo) in base alla size scelta
  const sizeClasses = {
    sm: "px-4 py-2 text-sm",      // Small: compatto
    md: "px-6 py-3 text-base",    // Medium: default
    lg: "px-8 py-4 text-lg",      // Large: prominente
  };

  // ----------------------------------------
  // RENDER
  // ----------------------------------------
  return (
    <button
      // Merge intelligente delle classi con cn() (evita conflitti Tailwind)
      className={cn(baseClasses, variantClasses[variant], sizeClasses[size], className)}
      // Disabilita il bottone se isLoading o disabled esplicito
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        // ----------------------------------------
        // SPINNER DI CARICAMENTO
        // ----------------------------------------
        // SVG che ruota per indicare caricamento in corso
        <span className="flex items-center gap-2">
          <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
            {/* Cerchio esterno (opaco) */}
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
            />
            {/* Arco di caricamento (visibile) */}
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          Caricamento...
        </span>
      ) : (
        // children normale se non in caricamento
        children
      )}
    </button>
  );
}
