/**
 * Configurazione PostCSS
 * ======================
 *
 * PostCSS è un tool per trasformare il CSS con plugin.
 * Next.js lo usa automaticamente durante la build.
 *
 * Questo file specifica quali plugin eseguire in ordine:
 * 1. Tailwind CSS (genera le utility classes)
 * 2. Autoprefixer (aggiunge vendor prefixes per compatibilità browser)
 */

module.exports = {
  plugins: {
    // ----------------------------------------
    // TAILWIND CSS
    // ----------------------------------------
    // Processa le direttive @tailwind e genera le utility classes
    // Legge la configurazione da tailwind.config.ts
    tailwindcss: {},

    // ----------------------------------------
    // AUTOPREFIXER
    // ----------------------------------------
    // Aggiunge automaticamente i vendor prefixes (-webkit-, -moz-, -ms-)
    // per garantire compatibilità con browser più vecchi
    // Esempio: transform → -webkit-transform, -ms-transform
    autoprefixer: {},
  },
};
