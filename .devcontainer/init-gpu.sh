#!/bin/bash

# このスクリプトは devcontainer の initializeCommand で実行されます。
# 目的: GPU が利用可能な場合は docker-compose.gpu.yml を有効化するための
# オーバーライドファイルを生成します。

GENERATED_FILE=".devcontainer/docker-compose.generated.yml"
GPU_COMPOSE_FILE="docker/docker-compose.gpu.yml"

echo "🔍 Checking for NVIDIA GPU..."

if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    echo "🚀 NVIDIA GPU supporting found. Generating GPU configuration..."
    # GPU設定をコピー
    # initializeCommand は workspace root で実行されることが多いが、
    # devcontainer.json の場所に基づいて相対パスが変わる可能性があるため注意。
    # 通常 initializeCommand はローカルのワークスペースルートで実行される。

    if [ -f "$GPU_COMPOSE_FILE" ]; then
        cp "$GPU_COMPOSE_FILE" "$GENERATED_FILE"
        echo "✅ Created $GENERATED_FILE with GPU settings."
    else
        echo "⚠️  Warning: $GPU_COMPOSE_FILE not found. GPU support will be disabled."
        echo "services: {}" > "$GENERATED_FILE"
    fi
else
    echo "💻 No NVIDIA GPU detected. Generating empty override configuration..."
    # GPUがない場合は空のオーバーライドファイルを生成 (これがないと docker-compose がエラーになる可能性があるため)
    echo "services: {}" > "$GENERATED_FILE"
    echo "✅ Created empty $GENERATED_FILE."
fi
