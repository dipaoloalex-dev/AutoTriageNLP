/**
 * Configurazione Next.js
 * =====================
 *
 * Questo file configura il comportamento di Next.js durante
 * lo sviluppo e la build di produzione.
 *
 * Documentazione: https://nextjs.org/docs/app/api-reference/next-config-js
 */

/** @type {import('next').NextConfig} */
const nextConfig = {
  // ----------------------------------------
  // OUTPUT STANDALONE
  // ----------------------------------------
  // Abilita l'output standalone per la build in Docker
  // Genera la cartella .next/standalone con tutto il necessario per l'esecuzione
  // Riduce la dimensione dell'immagine Docker includendo solo le dipendenze necessarie
  output: 'standalone',

  // ----------------------------------------
  // REACT STRICT MODE
  // ----------------------------------------
  // Abilita controlli aggiuntivi per identificare problemi potenziali
  // (es: lifecycle methods non sicuri, uso di API legacy)
  // In sviluppo, esegue ogni componente due volte per rilevare side effects
  reactStrictMode: true,

  // ----------------------------------------
  // IMAGES
  // ----------------------------------------
  // Configurazione per il componente Image di Next.js
  images: {
    // Domini consentiti per le immagini esterne
    // In questo caso, solo localhost per immagini locali
    domains: ['localhost'],
  },

  // ----------------------------------------
  // WEBPACK CUSTOMIZATION
  // ----------------------------------------
  // Per abilitare l'import di asset dalla root del progetto
  // e risolvere problemi con symlink (es: monorepo, pnpm)
  webpack: (config) => {
    // Disabilita la risoluzione dei symlink
    // Questo evita problemi quando si usano link simbolici per i moduli
    config.resolve.symlinks = false;
    return config;
  },
}

module.exports = nextConfig
