#!/usr/bin/env bash
set -euo pipefail

# Validate input argument
if [ "${1:-}" = "" ]; then
  echo "Usage: $0 <version>" >&2
  exit 1
fi

VERSION="$1"
REGISTRY="${REGISTRY:-ghcr.io}"
ACCOUNT="${ACCOUNT:-capsohq}"
IMAGE_NAME="bifrost"
IMAGE="${REGISTRY}/${ACCOUNT}/${IMAGE_NAME}"

echo "Creating multi-arch manifest for ${IMAGE}:v${VERSION}"
docker buildx imagetools create \
    --tag "${IMAGE}:v${VERSION}" \
    "${IMAGE}:v${VERSION}-amd64" \
    "${IMAGE}:v${VERSION}-arm64"

# Create latest manifest only for stable versions
if [[ "$VERSION" != *-* ]]; then
    echo "Creating stable latest manifest for ${IMAGE}:latest"
    docker buildx imagetools create \
        --tag "${IMAGE}:latest" \
        "${IMAGE}:v${VERSION}-amd64" \
        "${IMAGE}:v${VERSION}-arm64"
fi
