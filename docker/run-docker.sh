#!/bin/bash
cd "$(dirname "$0")"

if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    echo "🚀 NVIDIA GPU detected. Enabling GPU support..."
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.gpu.yml"
else
    echo "💻 No NVIDIA GPU detected or driver not responding. Running in CPU mode..."
    COMPOSE_FILES="-f docker-compose.yml"
fi

# デフォルトではbash、引数があればそれを実行
docker compose $COMPOSE_FILES run --rm ml-env "${@:-bash}"
