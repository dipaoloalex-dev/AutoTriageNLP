/**
 * Script di Rimozione Indicatori Dev Next.js
 * ===========================================
 *
 * Questo script rimuove aggressivamente tutti i possibili indicatori
 * di sviluppo che Next.js inietta nella pagina durante il sviluppo.
 *
 * Viene caricato dal file HTML principale per nascondere elementi come:
 * - Route indicator (bottone in basso a destra)
 * - Overlay di debug
 * - Altri elementi UI di sviluppo
 */

// ========================================
// IIFE (Immediately Invoked Function Expression)
// ========================================
// Esegue la funzione immediatamente dopo la definizione
// per non inquinare il namespace globale
(function() {
  /**
   * Funzione principale che cerca e rimuove gli indicatori
   */
  const hideIndicators = () => {
    // ----------------------------------------
    // SELETTORI CSS PER GLI INDICATORI
    // ----------------------------------------
    // Lista di selettori CSS che potrebbero corrispondere
    // agli indicatori di Next.js in basso a destra
    const selectors = [
      // Elementi con position: fixed (spesso usati per overlay)
      'div[style*="position: fixed"]',
      'div[style*="position:fixed"]',
      'button[style*="position: fixed"]',
      'button[style*="position:fixed"]',
      'svg[style*="position: fixed"]',
      'svg[style*="position:fixed"]',
      // Elementi con z-index molto elevato (tipico degli overlay)
      'div[style*="z-index: 9999"]',
      'div[style*="z-index:9999"]',
      'div[style*="z-index: 99999"]',
      'div[style*="z-index:99999"]',
      // Attributi specifici di Next.js
      '[data-nextjs-route-indicator]',
      '[aria-label*="Next.js"]',
      '[aria-label*="Route"]',
    ];

    // ----------------------------------------
    // RIMUOZIONE SELETTIVA
    // ----------------------------------------
    // Per ogni selettore, cerco gli elementi corrispondenti
    selectors.forEach(selector => {
      document.querySelectorAll(selector).forEach(el => {
        // Ottengo gli stili calcolati dell'elemento
        const style = window.getComputedStyle(el);
        const position = style.position;
        const zIndex = style.zIndex;
        const bottom = style.bottom;
        const right = style.right;

        // ----------------------------------------
        // CONDIZIONI DI RIMOZIONE
        // ----------------------------------------
        // Rimuovo solo gli elementi che soddisfano TUTTE queste condizioni:
        // 1. Position fixed o absolute
        // 2. Z-index elevato (> 1000 o valori tipici di overlay)
        // 3. Posizionato in basso a destra (dove appare solitamente l'indicator)
        if (
          (position === 'fixed' || position === 'absolute') &&
          (parseInt(zIndex) > 1000 || zIndex === '9999' || zIndex === '99999') &&
          (bottom === '12px' || bottom === '16px' || bottom === '20px' || bottom === '8px') &&
          (right === '12px' || right === '16px' || right === '20px' || right === '8px')
        ) {
          // Rimuovo l'elemento dal DOM
          el.remove();
        }
      });
    });
  };

  // ========================================
    // ESECUZIONE IMMEDIATA
    // ========================================
    // Esegui subito per catturare elementi già presenti
    hideIndicators();

    // ----------------------------------------
    // POLLING PERIODICO
    // ----------------------------------------
    // Next.js potrebbe iniettare elementi dinamici dopo il caricamento
    // Eseguo la funzione ogni 100ms per catturarli
    setInterval(hideIndicators, 100);

    // ----------------------------------------
    // ESECUZIONE AL CARICAMENTO DOM
    // ----------------------------------------
    // Se il DOM non è ancora pronto, attendo l'evento DOMContentLoaded
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', hideIndicators);
    }
  })();
  // Chiusura IIFE
})();
