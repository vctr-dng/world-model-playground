#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
ENV_FILE=".env"

if [[ ! -f "$ENV_FILE" && -f ".env.example" ]]; then
  cp .env.example "$ENV_FILE"
  echo "[INFO] Created $ENV_FILE from .env.example"
fi

if [[ -f "$ENV_FILE" ]]; then
  # Update existing keys or append if missing.
  if grep -q '^HOST_UID=' "$ENV_FILE"; then
    sed -i "s/^HOST_UID=.*/HOST_UID=${HOST_UID}/" "$ENV_FILE"
  else
    printf '\nHOST_UID=%s\n' "$HOST_UID" >>"$ENV_FILE"
  fi
  if grep -q '^HOST_GID=' "$ENV_FILE"; then
    sed -i "s/^HOST_GID=.*/HOST_GID=${HOST_GID}/" "$ENV_FILE"
  else
    printf 'HOST_GID=%s\n' "$HOST_GID" >>"$ENV_FILE"
  fi
else
  echo "[WARNING] .env and .env.example not found, please create .env manually."
fi

mkdir -p mlflow-data
chown -R "${HOST_UID}:${HOST_GID}" mlflow-data
docker compose up -d
