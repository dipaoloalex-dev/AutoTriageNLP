/**
 * Configurazione Tailwind CSS
 * ============================
 *
 * Tailwind è un framework CSS utility-first che permette di
 * costruire interfacce rapidamente usando classi utility.
 *
 * Questo file estende la configurazione di default con:
 * - Colori semantici basati su CSS variables
 * - Animazioni custom
 * - Tema per priorità dei ticket
 */

import type { Config } from "tailwindcss";

const config: Config = {
  // ----------------------------------------
  // DARK MODE
  // ----------------------------------------
  // "class" = attiva dark mode quando la classe 'dark' è presente
  // su un elemento genitore (per toggle manuale)
  darkMode: ["class"],

  // ----------------------------------------
  // CONTENT PATHS
  // ----------------------------------------
  // Percorsi dove Tailwind cerca le classi da generare
  // Include tutti i file .js, .ts, .jsx, .tsx, .mdx nelle cartelle specificate
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",   // Pagine Next.js
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}", // Componenti riutilizzabili
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",     // App directory (Next.js 13+)
  ],

  theme: {
    extend: {
      // ========================================
      // COLORI SEMANTICI
      // ========================================
      // I colori sono definiti come HSL che puntano a CSS variables
      // Questo permette di cambiare tema facilmente modificando solo le CSS variables
      colors: {
        // Sfondo e testo principale
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",

        // Colore primario (call-to-action, elementi principali)
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },

        // Colore secondario (elementi meno prominenti)
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },

        // Colore accento (evidenziamenti, hover states)
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },

        // Testi/elementi smorzati (disabilitati, secondari)
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },

        // Sfondo delle card
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },

        // ----------------------------------------
        // COLORI PRIORITÀ TICKET
        // ----------------------------------------
        // Colori specifici per visualizzare le priorità
        // Alta = rosso, Media = arancione, Bassa = verde
        priority: {
          alta: "#ffcdd2",   // Rosso chiaro (errore, urgente)
          media: "#ffe0b2",  // Arancione chiaro (warning)
          bassa: "#c8e6c9",  // Verde chiaro (success)
        },
      },

      // ========================================
      // BORDER RADIUS
      // ========================================
      // Raggio bordi basato su CSS variable per consistenza
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },

      // ========================================
      // ANIMAZIONI CUSTOM
      // ========================================
      // Definisco le animazioni che posso usare con le classi:
      // animate-fade-in-up, animate-pulse-glow, animate-slide-in-right, animate-slide-in-left
      animation: {
        "fade-in-up": "fadeInUp 0.6s ease-out",       // Dissolvenza dal basso
        "pulse-glow": "pulseGlow 2s infinite",        // Bagliore pulsante
        "slide-in-right": "slideInRight 0.4s ease-out", // Scorrimento da destra
        "slide-in-left": "slideInLeft 0.4s ease-out",   // Scorrimento da sinistra
      },

      // ----------------------------------------
      // KEYFRAMES
      // ----------------------------------------
      // Definizione dei keyframes per le animazioni sopra
      keyframes: {
        fadeInUp: {
          "0%": {
            opacity: "0",
            transform: "translateY(20px)", // Parte 20px più in basso
          },
          "100%": {
            opacity: "1",
            transform: "translateY(0)",    // Finisce in posizione
          },
        },
        pulseGlow: {
          "0%, 100%": {
            boxShadow: "0 0 5px rgba(255, 77, 77, 0.5)", // Bagliore debole
          },
          "50%": {
            boxShadow: "0 0 20px rgba(255, 77, 77, 0.8)", // Bagliore forte a metà
          },
        },
        slideInRight: {
          "0%": {
            opacity: "0",
            transform: "translateX(-30px)", // Parte 30px a sinistra
          },
          "100%": {
            opacity: "1",
            transform: "translateX(0)",      // Finisce in posizione
          },
        },
        slideInLeft: {
          "0%": {
            opacity: "0",
            transform: "translateX(30px)",  // Parte 30px a destra
          },
          "100%": {
            opacity: "1",
            transform: "translateX(0)",      // Finisce in posizione
          },
        },
      },
    },
  },

  // ----------------------------------------
  // PLUGINS
  // ----------------------------------------
  // Plugin Tailwind aggiuntivi (vuoto per ora)
  plugins: [],
};

export default config;
