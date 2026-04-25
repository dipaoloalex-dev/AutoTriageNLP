/**
 * ========================================
 * Utilità Generali
 * ========================================
 *
 * Funzioni helper riutilizzabili in tutta l'applicazione.
 */

import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

// ========================================
// MERGE CLASSI TAILWIND
// ========================================
/**
 * Unisce classi Tailwind in modo intelligente.
 *
 * - clsx: unisce classi condizionalmente
 * - twMerge: risolve conflitti Tailwind (es: "px-4 px-6" → "px-6")
 *
 * @param inputs - Classi da unire (stringhe, oggetti, array)
 * @returns Stringa di classi merged
 *
 * @example
 * cn("px-4 py-2", isActive && "bg-blue-500", "hover:scale-105")
 * // → "px-4 py-2 bg-blue-500 hover:scale-105"
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// ========================================
// FORMATTAZIONE CONFIDENZA
// ========================================
/**
 * Formatta un valore di confidenza (0-1) come percentuale.
 *
 * @param confidence - Valore tra 0 e 1
 * @returns Stringa percentuale con 1 decimale
 *
 * @example
 * formatConfidence(0.856) // → "85.6%"
 */
export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}

// ========================================
// TRONCATURA TESTO
// ========================================
/**
 * Tronca il testo se supera la lunghezza massima specificata.
 *
 * @param text - Testo da troncare
 * @param maxLength - Lunghezza massima (default: 200)
 * @returns Testo troncato con "..." finale se necessario
 *
 * @example
 * truncateText("Questo è un testo molto lungo...", 10) // → "Questo è..."
 */
export function truncateText(text: string, maxLength: number = 200): string {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + "...";
}
