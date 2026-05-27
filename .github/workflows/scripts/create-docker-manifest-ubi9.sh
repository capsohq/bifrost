#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "Usage: $0 <version>" >&2
  exit 1
fi

VERSION="$1"
REGISTRY="${REGISTRY:-ghcr.io}"
ACCOUNT="${ACCOUNT:-capsohq}"
IMAGE_NAME="bifrost"
IMAGE="${REGISTRY}/${ACCOUNT}/${IMAGE_NAME}"

echo "Creating UBI9 multi-arch manifest for ${IMAGE}:v${VERSION}-ubi9"
docker buildx imagetools create \
    --tag "${IMAGE}:v${VERSION}-ubi9" \
    "${IMAGE}:v${VERSION}-ubi9-amd64" \
    "${IMAGE}:v${VERSION}-ubi9-arm64"

if [[ "$VERSION" != *-* ]]; then
    echo "Creating stable latest UBI9 manifest for ${IMAGE}:latest-ubi9"
    docker buildx imagetools create \
        --tag "${IMAGE}:latest-ubi9" \
        "${IMAGE}:v${VERSION}-ubi9-amd64" \
        "${IMAGE}:v${VERSION}-ubi9-arm64"
fi
