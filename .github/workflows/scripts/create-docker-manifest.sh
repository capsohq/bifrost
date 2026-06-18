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

# Filter by platform.architecture rather than relying on positional [0]:
# docker/build-push-action with default provenance creates an OCI image index
# containing platform image manifests and provenance attestations.
AMD64_DIGEST=$(docker manifest inspect "${IMAGE}:v${VERSION}-amd64" | jq -er '.manifests[] | select(.platform.architecture == "amd64") | .digest')
ARM64_DIGEST=$(docker manifest inspect "${IMAGE}:v${VERSION}-arm64" | jq -er '.manifests[] | select(.platform.architecture == "arm64") | .digest')

echo "AMD64 digest: ${AMD64_DIGEST}"
echo "ARM64 digest: ${ARM64_DIGEST}"

# Create manifest for versioned tag using digests.
docker manifest create \
    "${IMAGE}:v${VERSION}" \
    "${IMAGE}@${AMD64_DIGEST}" \
    "${IMAGE}@${ARM64_DIGEST}"

docker manifest push "${IMAGE}:v${VERSION}"

# Create latest manifest only for stable versions.
if [[ "$VERSION" != *-* ]]; then
    docker manifest create \
        "${IMAGE}:latest" \
        "${IMAGE}@${AMD64_DIGEST}" \
        "${IMAGE}@${ARM64_DIGEST}"

    docker manifest push "${IMAGE}:latest"
fi
