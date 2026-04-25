/**
 * ========================================
 * DevIndicatorRemover - Rimozione Indicatori Dev
 * ========================================
 *
 * Componente che rimuove AGGRESSIVAMENTE tutti gli indicatori
 * di sviluppo che Next.js inietta durante il development mode.
 *
 * Utilizza multipla strategia:
 * - Query selector multipli
 * - Polling a più intervalli
 * - MutationObserver per DOM changes
 *
 * Nota: Questo componente non renderizza nulla (return null)
 */

"use client"; // Necessario per useEffect e DOM manipulation

import { useEffect } from "react";

export function DevIndicatorRemover() {
  useEffect(() => {
    // ========================================
    // FUNZIONE DI RIMOZIONE
    // ========================================
    /**
     * Cerca e rimuove tutti gli elementi che sembrano
     * indicatori di sviluppo Next.js.
     */
    const removeDevIndicator = () => {
      // ----------------------------------------
      // LISTA SELETTORI
      // ----------------------------------------
      // Tutti i possibili selettori che potrebbero
      // corrispondere all'indicatore Next.js
      const selectors = [
        'div[style*="z-index: 2147483647"]',   // Z-index massimo
        'div[style*="z-index:2147483647"]',
        'div[style*="z-index: 9999"]',
        'div[style*="z-index:9999"]',
        'div[style*="z-index: 99999"]',
        'div[style*="z-index:99999"]',
        '[data-nextjs-route-indicator]',      // Attributo specifico
        '[data-nextjs-dev]',
        '[data-nextjs]',
        'div[class*="nextjs"]',               // Classe con "nextjs" nel nome
        'div[aria-label*="Next.js"]',
        'div[aria-label*="Route"]',
        'button[aria-label*="Next.js"]',
        'button[aria-label*="Route"]',
      ];

      // ----------------------------------------
      // RIMOZIONE CON SELETTORI
      // ----------------------------------------
      selectors.forEach((selector) => {
        try {
          document.querySelectorAll(selector).forEach((el) => {
            const style = window.getComputedStyle(el);
            const position = style.position;
            const zIndex = style.zIndex;
            const bottom = style.bottom;
            const right = style.right;

            // Condizioni: elemento in basso a destra con z-index elevato
            if (
              (position === "fixed" || position === "absolute") &&
              (parseInt(zIndex) > 1000 ||
                zIndex === "2147483647" ||
                zIndex === "9999" ||
                zIndex === "99999") &&
              (bottom === "12px" ||
                bottom === "16px" ||
                bottom === "20px" ||
                bottom === "8px" ||
                bottom === "0px") &&
              (right === "12px" ||
                right === "16px" ||
                right === "20px" ||
                right === "8px" ||
                right === "0px")
            ) {
              el.remove();  // Rimuovi elemento
            }
          });
        } catch (e) {
          // Ignora errori (elementi potrebbero non esistere)
        }
      });

      // ----------------------------------------
      // RIMOZIONE GENERICAMENTE TUTTI I DIV
      // ----------------------------------------
      // Controlla tutti i div nel documento
      document.querySelectorAll("div").forEach((div) => {
        try {
          const style = window.getComputedStyle(div);
          if (
            style.position === "fixed" &&
            style.zIndex === "2147483647" &&
            (style.bottom === "12px" || style.bottom === "16px") &&
            (style.right === "12px" || style.right === "16px")
          ) {
            div.remove();
          }
        } catch (e) {
          // Ignora errori
        }
      });

      // ----------------------------------------
      // RIMOZIONE TUTTI I BUTTON
      // ----------------------------------------
      // Controlla tutti i button nell'angolo basso-destra
      document.querySelectorAll("button").forEach((btn) => {
        try {
          const style = window.getComputedStyle(btn);
          if (
            style.position === "fixed" &&
            (parseInt(style.zIndex) > 1000 || style.zIndex === "2147483647")
          ) {
            const rect = btn.getBoundingClientRect();
            // Se è nell'angolo in basso a destra
            if (
              rect.bottom >= window.innerHeight - 100 &&
              rect.right >= window.innerWidth - 100
            ) {
              btn.remove();
            }
          }
        } catch (e) {
          // Ignora errori
        }
      });
    };

    // ========================================
    // ESECUZIONE IMMEDIATA
    // ========================================
    // Esegui subito per catturare elementi già presenti
    removeDevIndicator();

    // ========================================
    // POLLING MULTIPLA
    // ========================================
    // Eseguo a intervalli diversi per catturare elementi
    // che potrebbero essere iniettati in momenti diversi
    const intervals = [
      setInterval(removeDevIndicator, 10),    // 10ms: molto frequente
      setInterval(removeDevIndicator, 50),    // 50ms: frequente
      setInterval(removeDevIndicator, 100),   // 100ms: medio
      setInterval(removeDevIndicator, 500),   // 500ms: lento
    ];

    // ========================================
    // MUTATION OBSERVER
    // ========================================
    // Osserva le modifiche al DOM e rimuove indicatori
    // quando nuovi elementi vengono aggiunti
    const observer = new MutationObserver(() => {
      removeDevIndicator();
    });
    observer.observe(document.body, {
      childList: true,  // Osserva aggiunte/rimozioni figli
      subtree: true,     // Osserva anche i discendenti
    });

    // ========================================
    // CLEANUP
    // ========================================
    // Quando il componente viene smontato, pulisci tutto
    return () => {
      intervals.forEach(clearInterval);
      observer.disconnect();
    };
  }, []);

  // Non renderizza nulla
  return null;
}
