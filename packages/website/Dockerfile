# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 node:26-bookworm-slim AS base

ENV PNPM_HOME="/pnpm" \
    PATH="$PNPM_HOME:$PATH" \
    NODE_ENV=production \
    PORT=8080 \
    CI=true

RUN npm install -g pnpm@11.27.0

WORKDIR /app

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./

RUN pnpm fetch --reporter=silent

COPY . .

RUN pnpm install --frozen-lockfile --prefer-offline

RUN pnpm build

EXPOSE 8080

CMD ["node", "/app/packages/website/.output/server/index.mjs"]
