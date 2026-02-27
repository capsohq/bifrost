package moonshot_test

import (
	"os"
	"strings"
	"testing"

	"github.com/capsohq/bifrost/core/internal/llmtests"
	"github.com/capsohq/bifrost/core/schemas"
)

func envOrDefault(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func TestMoonshot(t *testing.T) {
	t.Parallel()

	if strings.TrimSpace(os.Getenv("MOONSHOT_API_KEY")) == "" {
		t.Skip("Skipping Moonshot tests because MOONSHOT_API_KEY is not set")
	}

	client, ctx, cancel, err := llmtests.SetupTest()
	if err != nil {
		t.Fatalf("Error initializing test setup: %v", err)
	}
	defer cancel()

	testConfig := llmtests.ComprehensiveTestConfig{
		Provider:  schemas.Moonshot,
		ChatModel: envOrDefault("MOONSHOT_CHAT_MODEL", "kimi-k2.5"),
		TextModel: envOrDefault("MOONSHOT_TEXT_MODEL", "kimi-k2.5"),
		Scenarios: llmtests.TestScenarios{
			TextCompletion:        true,
			TextCompletionStream:  true,
			SimpleChat:            true,
			CompletionStream:      true,
			MultiTurnConversation: true,
			ToolCalls:             true,
			MultipleToolCalls:     true,
			End2EndToolCalling:    true,
			AutomaticFunctionCall: true,
			ListModels:            true,
		},
	}

	t.Run("MoonshotTests", func(t *testing.T) {
		llmtests.RunAllComprehensiveTests(t, client, ctx, testConfig)
	})
	client.Shutdown()
}
