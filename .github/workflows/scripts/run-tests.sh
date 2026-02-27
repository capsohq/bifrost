#!/usr/bin/env bash
set -euo pipefail

# Comprehensive test runner for Bifrost PR validation
# This script runs all test suites to validate changes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "🧪 Starting Bifrost Test Suite..."
echo "=================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to report test result
report_result() {
  local test_name=$1
  local result=$2
  
  if [ "$result" -eq 0 ]; then
    echo -e "${GREEN}✅ $test_name passed${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
  else
    echo -e "${RED}❌ $test_name failed${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
  fi
}

# 1. Core Build Validation
echo ""
echo "📦 1/5 - Validating Core Build..."
echo "-----------------------------------"
cd core
if go mod download && go build ./...; then
  report_result "Core Build" 0
else
  report_result "Core Build" 1
fi
cd ..

# 2. Build MCP Test Servers
echo ""
echo "🔌 2/5 - Building MCP Test Servers..."
echo "-----------------------------------"
MCP_BUILD_FAILED=0
for mcp_dir in examples/mcps/*/; do
  if [ -d "$mcp_dir" ]; then
    mcp_name=$(basename "$mcp_dir")
    if [ -f "$mcp_dir/go.mod" ]; then
      echo "  Building $mcp_name (Go)..."
      mkdir -p "$mcp_dir/bin"
      if cd "$mcp_dir" && GOWORK=off go build -o "bin/$mcp_name" . && cd - > /dev/null; then
        echo -e "  ${GREEN}✓ $mcp_name${NC}"
      else
        echo -e "  ${RED}✗ $mcp_name${NC}"
        MCP_BUILD_FAILED=1
        cd - > /dev/null 2>&1 || true
      fi
    elif [ -f "$mcp_dir/package.json" ]; then
      echo "  Building $mcp_name (TypeScript)..."
      if cd "$mcp_dir" && npm install --silent && npm run build && cd - > /dev/null; then
        echo -e "  ${GREEN}✓ $mcp_name${NC}"
      else
        echo -e "  ${RED}✗ $mcp_name${NC}"
        MCP_BUILD_FAILED=1
        cd - > /dev/null 2>&1 || true
      fi
    fi
  fi
done
report_result "MCP Test Servers Build" $MCP_BUILD_FAILED

# 3. Core Provider Tests
echo ""
echo "🔧 3/5 - Running Core Provider Tests..."
echo "-----------------------------------"
cd core
if go test -v -run . ./...; then
  report_result "Core Provider Tests" 0
else
  report_result "Core Provider Tests" 1
fi
cd ..

# 4. Governance Tests
echo ""
echo "🛡️  4/5 - Running Governance Tests..."
echo "-----------------------------------"
if [ -d "tests/governance" ] && [ -x "$SCRIPT_DIR/run-governance-e2e-tests.sh" ]; then
  if "$SCRIPT_DIR/run-governance-e2e-tests.sh"; then
    report_result "Governance Tests" 0
  else
    report_result "Governance Tests" 1
  fi
else
  echo -e "${YELLOW}⚠️  Governance E2E script or directory not found, skipping...${NC}"
fi

# 5. Integration Tests
echo ""
echo "🔗 5/5 - Running Integration Tests..."
echo "-----------------------------------"
if [ -d "tests/integrations/python" ]; then
  echo "Building bifrost-http binary for integration tests..."
  mkdir -p tmp
  INTEGRATION_BUILD_FAILED=0
  if ! (cd transports/bifrost-http && CGO_ENABLED=1 go build -tags "sqlite_static" -o ../../tmp/bifrost-http .); then
    INTEGRATION_BUILD_FAILED=1
  fi

  if [ "$INTEGRATION_BUILD_FAILED" -ne 0 ]; then
    report_result "Integration Tests" 1
    echo -e "${RED}❌ Failed to build tmp/bifrost-http${NC}"
  else
    INTEGRATION_APP_DIR="$(mktemp -d)"
    if ! python3 - "$INTEGRATION_APP_DIR/config.json" "$INTEGRATION_APP_DIR" <<'PY'
import json
import os
import sys

src = "tests/integrations/python/config.json"
dst = sys.argv[1]
app_dir = sys.argv[2]

with open(src, "r", encoding="utf-8") as input_file:
    config = json.load(input_file)

config["config_store"] = {
    "enabled": True,
    "type": "sqlite",
    "config": {"path": os.path.join(app_dir, "config.db")},
}
config["logs_store"] = {
    "enabled": True,
    "type": "sqlite",
    "config": {"path": os.path.join(app_dir, "logs.db")},
}

for key in config.get("providers", {}).get("gemini", {}).get("keys", []):
    key.setdefault("name", "Gemini API Key")

for virtual_key in config.get("governance", {}).get("virtual_keys", []):
    virtual_key.setdefault("name", virtual_key.get("id", "integration-vk"))

if config.get("client", {}).get("allowed_origins") == ["*"]:
    config["client"]["allowed_origins"] = ["https://integration-tests.local"]

with open(dst, "w", encoding="utf-8") as output_file:
    json.dump(config, output_file, indent=2)
PY
    then
      report_result "Integration Tests" 1
      echo -e "${RED}❌ Failed to prepare integration test config${NC}"
      rm -rf "$INTEGRATION_APP_DIR"
      INTEGRATION_APP_DIR=""
    fi

    if [ -n "${INTEGRATION_APP_DIR:-}" ]; then
      INTEGRATION_PORT="${BIFROST_INTEGRATION_PORT:-18080}"
      INTEGRATION_URL="http://localhost:${INTEGRATION_PORT}"
      INTEGRATION_LOG_FILE="$(mktemp)"
      INTEGRATION_PID=""

      cleanup_integration_server() {
        if [ -n "${INTEGRATION_PID:-}" ] && kill -0 "$INTEGRATION_PID" 2>/dev/null; then
          kill "$INTEGRATION_PID" 2>/dev/null || true
          wait "$INTEGRATION_PID" 2>/dev/null || true
        fi
        if [ -n "${INTEGRATION_APP_DIR:-}" ] && [ -d "$INTEGRATION_APP_DIR" ]; then
          rm -rf "$INTEGRATION_APP_DIR"
        fi
      }

      echo "Starting Bifrost for integration tests on ${INTEGRATION_URL}..."
      ./tmp/bifrost-http --app-dir "$INTEGRATION_APP_DIR" --port "$INTEGRATION_PORT" --host localhost > "$INTEGRATION_LOG_FILE" 2>&1 &
      INTEGRATION_PID=$!

      SERVER_READY=0
      for _ in $(seq 1 60); do
        if curl -sf "${INTEGRATION_URL}/health" > /dev/null 2>&1; then
          SERVER_READY=1
          break
        fi
        if ! kill -0 "$INTEGRATION_PID" 2>/dev/null; then
          break
        fi
        sleep 1
      done

      if [ "$SERVER_READY" -ne 1 ]; then
        echo -e "${RED}❌ Integration test server failed to start${NC}"
        tail -n 100 "$INTEGRATION_LOG_FILE" || true
        cleanup_integration_server
        rm -f "$INTEGRATION_LOG_FILE"
        report_result "Integration Tests" 1
      else
        cd tests/integrations/python

        if ! command -v uv >/dev/null 2>&1; then
          echo "Installing uv..."
          python3 -m pip install --quiet uv
        fi

        echo "Installing Python dependencies with uv..."
        if uv sync --quiet && BIFROST_BASE_URL="$INTEGRATION_URL" uv run python run_all_tests.py; then
          report_result "Integration Tests" 0
        else
          report_result "Integration Tests" 1
        fi

        cd "$REPO_ROOT"
        cleanup_integration_server
        rm -f "$INTEGRATION_LOG_FILE"
      fi
    fi
  fi
elif [ -d "tests/integrations" ]; then
  echo -e "${YELLOW}⚠️  tests/integrations/python not found, running legacy tests/integrations path...${NC}"
  cd tests/integrations
  if python run_all_tests.py; then
    report_result "Integration Tests" 0
  else
    report_result "Integration Tests" 1
  fi
  cd "$REPO_ROOT"
else
  echo -e "${YELLOW}⚠️  Integration tests directory not found, skipping...${NC}"
fi

# Final Summary
echo ""
echo "=================================="
echo "🏁 Test Suite Complete!"
echo "=================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ "$TESTS_FAILED" -gt 0 ]; then
  echo -e "${RED}❌ Some tests failed. Please review the output above.${NC}"
  exit 1
else
  echo -e "${GREEN}✅ All tests passed successfully!${NC}"
  exit 0
fi
