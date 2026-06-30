# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Build
# Use the official Node LTS Alpine image to build the Vite/React app
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY package*.json ./
RUN npm config set registry https://registry.npmmirror.com && \
    npm install
    
# Copy the rest of the source
COPY . .

# VITE_API_URL is passed in at build time by GitHub Actions.
# In K8s, this becomes "/api" because nginx on the same pod will
# reverse-proxy /api/* → the internal backend service.
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Serve
# Ultra-lightweight nginx:alpine image — only the dist/ folder goes in here.
# ─────────────────────────────────────────────────────────────────────────────
FROM nginx:alpine

# Replace the default nginx config with our custom one
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the compiled frontend from the build stage
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
