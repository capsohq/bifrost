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

# Filter by platform.architecture rather than positional [0]: buildx may include
# provenance attestations in the OCI index and ordering is not stable.
AMD64_DIGEST=$(docker manifest inspect "${IMAGE}:v${VERSION}-ubi9-amd64" | jq -er '.manifests[] | select(.platform.architecture == "amd64") | .digest')
ARM64_DIGEST=$(docker manifest inspect "${IMAGE}:v${VERSION}-ubi9-arm64" | jq -er '.manifests[] | select(.platform.architecture == "arm64") | .digest')

echo "UBI9 AMD64 digest: ${AMD64_DIGEST}"
echo "UBI9 ARM64 digest: ${ARM64_DIGEST}"

docker manifest create \
    "${IMAGE}:v${VERSION}-ubi9" \
    "${IMAGE}@${AMD64_DIGEST}" \
    "${IMAGE}@${ARM64_DIGEST}"

docker manifest push "${IMAGE}:v${VERSION}-ubi9"

if [[ "$VERSION" != *-* ]]; then
    docker manifest create \
        "${IMAGE}:latest-ubi9" \
        "${IMAGE}@${AMD64_DIGEST}" \
        "${IMAGE}@${ARM64_DIGEST}"

    docker manifest push "${IMAGE}:latest-ubi9"
fi
