#!/usr/bin/env bash
set -euo pipefail

# Configure AWS CLI for R2 uploads.
#
# Forks may build and publish Docker images without also publishing binary
# downloads to R2. Set R2_REQUIRED=true to make missing R2 credentials fatal.
# Usage: ./configure-r2.sh

echo "⚙️ Configuring AWS CLI for R2..."

# Clean and trim environment variables (removing any whitespace)
R2_ENDPOINT="$(echo "${R2_ENDPOINT:-}" | tr -d '[:space:]')"
R2_ACCESS_KEY_ID="$(echo "${R2_ACCESS_KEY_ID:-}" | tr -d '[:space:]')"
R2_SECRET_ACCESS_KEY="$(echo "${R2_SECRET_ACCESS_KEY:-}" | tr -d '[:space:]')"

# Validate environment variables
if [ -z "$R2_ENDPOINT" ] || [ -z "$R2_ACCESS_KEY_ID" ] || [ -z "$R2_SECRET_ACCESS_KEY" ]; then
  if [ "${R2_REQUIRED:-false}" = "true" ]; then
    echo "❌ Missing required R2 credentials"
    exit 1
  fi
  echo "⚠️ Missing R2 credentials; skipping R2 configuration"
  exit 0
fi

pip install awscli

# Configure AWS CLI for R2 using dedicated profile
aws configure set --profile R2 aws_access_key_id "$R2_ACCESS_KEY_ID"
aws configure set --profile R2 aws_secret_access_key "$R2_SECRET_ACCESS_KEY"
aws configure set --profile R2 region us-east-1
aws configure set --profile R2 s3.signature_version s3v4

# Test connection
echo "🔍 Testing R2 connection..."
aws s3 ls s3://prod-downloads/ --endpoint-url "$R2_ENDPOINT" --profile R2 >/dev/null
echo "✅ R2 connection successful"
