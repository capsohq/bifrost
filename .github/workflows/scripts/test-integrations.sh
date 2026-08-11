#!/usr/bin/env bash
set -euo pipefail

# Test integration tests by building bifrost-http from source, starting it,
# and running Python and TypeScript SDK integration tests
# Usage: ./test-integrations.sh

# Get the absolute path of the script directory
if command -v readlink >/dev/null 2>&1 && readlink -f "$0" >/dev/null 2>&1; then
  SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
fi

# Repository root (3 levels up from .github/workflows/scripts)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"

# Setup Go workspace for CI (go.work is gitignored, must be regenerated)
source "$SCRIPT_DIR/setup-go-workspace.sh"

echo "🧪 Running Integration Tests"
echo "   Repository root: $REPO_ROOT"

# Configuration
TEST_PORT="${PORT:-8080}"
TEST_HOST="${HOST:-localhost}"
BIFROST_PID=""
TEST_FAILED=0
LOG_FILE="$(mktemp /tmp/bifrost-integrations.XXXXXX.log)"

# Some integration runners still check GOOGLE_API_KEY, while workflow secrets
# provide GEMINI_API_KEY. Mirror it here so both code paths work.
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-${GEMINI_API_KEY:-}}"

has_any_provider_credentials() {
  local vars=(
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    GEMINI_API_KEY
    GOOGLE_API_KEY
    VERTEX_PROJECT_ID
    VERTEX_CREDENTIALS
    MISTRAL_API_KEY
    COHERE_API_KEY
    GROQ_API_KEY
    PERPLEXITY_API_KEY
    CEREBRAS_API_KEY
    OPENROUTER_API_KEY
    PARASAIL_API_KEY
    ELEVENLABS_API_KEY
    XAI_API_KEY
    HUGGING_FACE_API_KEY
    AZURE_API_KEY
    AWS_ACCESS_KEY_ID
  )

  local var_name
  for var_name in "${vars[@]}"; do
    if [ -n "${!var_name:-}" ]; then
      return 0
    fi
  done
  return 1
}

# Cleanup function
cleanup() {
  local exit_code=$?
  echo ""
  echo "🧹 Cleaning up..."
  
  # Kill Bifrost server if running
  if [ -n "${BIFROST_PID:-}" ]; then
    echo "   Stopping Bifrost server (PID: $BIFROST_PID)..."
    kill "$BIFROST_PID" 2>/dev/null || true
    wait "$BIFROST_PID" 2>/dev/null || true
  fi

  rm -f "${LOG_FILE:-}" 2>/dev/null || true

  exit $exit_code
}
trap cleanup EXIT

# Step 1: Build bifrost-http from source
echo ""
echo "🔨 Building bifrost-http from source..."
cd "$REPO_ROOT"

# Build the UI first, then the binary
make build-ui
make build LOCAL=1

if [ ! -f "$REPO_ROOT/tmp/bifrost-http" ]; then
  echo "❌ Error: bifrost-http binary not found at $REPO_ROOT/tmp/bifrost-http"
  exit 1
fi

echo "✅ Build complete: $REPO_ROOT/tmp/bifrost-http"

# Step 2: Start Bifrost server with Python integration test config
echo ""
echo "🚀 Starting Bifrost server..."
echo "   Config: tests/integrations/python/config.json"
echo "   Host: $TEST_HOST"
echo "   Port: $TEST_PORT"

# Start server in background with Python config directory
"$REPO_ROOT/tmp/bifrost-http" \
  -host "$TEST_HOST" \
  -port "$TEST_PORT" \
  -log-style json \
  -log-level info \
  -app-dir "$REPO_ROOT/tests/integrations/python" \
  > "$LOG_FILE" 2>&1 &

BIFROST_PID=$!
echo "   Started with PID: $BIFROST_PID"


# Wait for server to be ready. Startup now includes model catalog sync,
# optional MCP client retries, and provider/model discovery, so 30s is too low.
echo "⏳ Waiting for Bifrost to be ready..."
MAX_WAIT="${BIFROST_STARTUP_MAX_WAIT:-90}"
ELAPSED=0
SERVER_READY=false

while [ $ELAPSED -lt $MAX_WAIT ]; do
  if curl --connect-timeout 10 --max-time 20 -sf "http://$TEST_HOST:$TEST_PORT/health" > /dev/null 2>&1; then
    SERVER_READY=true
    echo "✅ Bifrost is ready (took ${ELAPSED}s)"
    break
  fi
  
  # Check if server process is still running
  if ! kill -0 "$BIFROST_PID" 2>/dev/null; then
    echo "❌ Bifrost process died unexpectedly"
    echo "📄 Last 200 lines of server log:"
    tail -n 200 "$LOG_FILE" || true
    exit 1
  fi
  
  sleep 1
  ELAPSED=$((ELAPSED + 1))
done

if [ "$SERVER_READY" = false ]; then
  echo "❌ Bifrost failed to start within ${MAX_WAIT}s"
  echo "📄 Last 200 lines of server log:"
  tail -n 200 "$LOG_FILE" || true
  exit 1
fi

# Set environment variable for tests
export BIFROST_BASE_URL="http://$TEST_HOST:$TEST_PORT"
echo "   BIFROST_BASE_URL=$BIFROST_BASE_URL"

if ! has_any_provider_credentials; then
  echo ""
  echo "⏭️  No provider credentials available in CI."
  echo "   Skipping SDK integration suites after successful Bifrost startup smoke test."
  exit 0
fi

# Step 3: Run Python integration tests
echo ""
echo "🐍 Running Python integration tests..."
echo "="

cd "$REPO_ROOT/tests/integrations/python"

# Check if uv is available
if command -v uv >/dev/null 2>&1; then
  echo "📦 Installing Python dependencies with uv..."
  uv sync --frozen --quiet
  
  echo ""
  echo "🏃 Running Python tests..."
  if ! uv run python run_all_tests.py; then
    echo "⚠️  Python tests failed"
    TEST_FAILED=1
  fi
else
  echo "⚠️  uv not found, trying pip..."
  
  # Create virtual environment if needed
  if [ ! -d ".venv" ]; then
    python3 -m venv .venv
  fi
  
  source .venv/bin/activate
  pip install -q -e .
  
  echo ""
  echo "🏃 Running Python tests..."
  if ! python run_all_tests.py; then
    echo "⚠️  Python tests failed"
    TEST_FAILED=1
  fi
fi

# Step 4: Run TypeScript integration tests
echo ""
echo "📘 Running TypeScript integration tests..."
echo "="

cd "$REPO_ROOT/tests/integrations/typescript"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
  echo "📦 Installing TypeScript dependencies with npm..."
  npm ci
fi

echo ""
echo "🏃 Running TypeScript tests..."
if ! npm test; then
  echo "⚠️  TypeScript tests failed"
  TEST_FAILED=1
fi

# Summary
echo ""
echo "="
if [ $TEST_FAILED -eq 1 ]; then
  echo "❌ Some integration tests failed"
  exit 1
else
  echo "✅ All integration tests passed!"
fi
