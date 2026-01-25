#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# ==== 設定（必要なら変更）====
USER_NAME="${USER_NAME:-kujira}"
GROUP_NAME="${GROUP_NAME:-$USER_NAME}"

CONTEXT=".."   # レポジトリルート
REPO_NAME="$(basename "$(cd "${CONTEXT}" && pwd)")"

# bash の組込み $UID をそのまま使う（上書きしない）
USER_UID="${UID}"
USER_GID="$(id -g)"
USER_HOME="/home/${USER_NAME}"
IMAGE_NAME=${REPO_NAME}
DOCKERFILE="./Dockerfile"

# ==== 実行 ====
echo "Build args:"
echo "  REPO_NAME=${REPO_NAME}"
echo "  USER_NAME=${USER_NAME}"
echo "  GROUP_NAME=${GROUP_NAME}"
echo "  USER_UID=${USER_UID}"
echo "  USER_GID=${USER_GID}"
echo "  HOME=${USER_HOME}"

docker build \
  -t "${IMAGE_NAME}" \
  -f "${DOCKERFILE}" \
  --build-arg REPO_NAME="${REPO_NAME}" \
  --build-arg USER_NAME="${USER_NAME}" \
  --build-arg GROUP_NAME="${GROUP_NAME}" \
  --build-arg USER_UID="${USER_UID}" \
  --build-arg USER_GID="${USER_GID}" \
  --build-arg HOME="${USER_HOME}" \
  "${CONTEXT}"