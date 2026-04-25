# ========================================
# Frontend Dockerfile per Next.js
# ========================================
#
# Multi-stage build per ottimizzare la dimensione dell'immagine
#
# Stage: base → deps → builder → runner
# ========================================

# ----------------------------------------
# BASE IMAGE
# ----------------------------------------
# Immagine base leggera (Alpine Linux) con Node.js 20
FROM node:20-alpine AS base

# ========================================
# STAGE 1: INSTALLAZIONE DIPENDENZE
# ========================================
FROM base AS deps

# Imposto la directory di lavoro
WORKDIR /app

# Copio solo i file package.json e package-lock.json
# (non copio tutto il progetto per sfruttare la cache Docker)
COPY package*.json ./

# Installo le dipendenze con npm ci (più sicuro e veloce di npm install)
RUN npm ci

# ========================================
# STAGE 2: BUILD
# ========================================
FROM base AS builder

WORKDIR /app

# Copio le dipendenze dallo stage deps
COPY --from=deps /app/node_modules ./node_modules

# Copio tutto il resto del progetto
COPY . .

# Eseguo la build di produzione
# Questo genera la cartella .next con l'app ottimizzata
RUN npm run build

# ========================================
# STAGE 3: PRODUCTION RUNNER
# ========================================
# Immagine finale minimale che contiene solo l'eseguibile
FROM base AS runner

WORKDIR /app

# ----------------------------------------
# CONFIGURAZIONE AMBIENTE
# ----------------------------------------
# Imposto Node.js in modalità produzione
ENV NODE_ENV production

# ----------------------------------------
# UTENTE NON-ROOT
# # ----------------------------------------
# Creo un utente non-root per sicurezza (best practice)
# Se l'app viene compromessa, l'attaccante non ha privilegi di root
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# ----------------------------------------
# COPIA ARTIFACTS DI BUILD
# ----------------------------------------
# Copio solo i file necessari per l'esecuzione

# File statici (immagini, favicon, ecc.)
COPY --from=builder /app/public ./public

# Build standalone di Next.js (configurato in next.config.js)
# Contiene il server e le pagine ottimizzate
COPY --from=builder /app/.next/standalone ./

# Asset statici generati da Next.js (CSS, JS ottimizzato)
COPY --from=builder /app/.next/static ./.next/static

# ----------------------------------------
# CONFIGURAZIONE UTENTE E PORTA
# ----------------------------------------
# Passo all'utente non-root
USER nextjs

# Espongo la porta 3000 (standard Next.js)
EXPOSE 3000

# Variabili ambiente per la configurazione del server
ENV PORT 3000
ENV HOSTNAME "0.0.0.0"  # Ascolto su tutte le interfacce

# ----------------------------------------
# COMANDO DI AVVIO
# ----------------------------------------
# Avvio il server Next.js in modalità produzione
# server.js è generato automaticamente dalla build standalone
CMD ["node", "server.js"]
