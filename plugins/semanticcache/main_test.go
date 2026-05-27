package semanticcache

import (
	"context"
	"os"
	"testing"
	"time"

	bifrost "github.com/capsohq/bifrost/core"
	"github.com/capsohq/bifrost/core/schemas"
	"github.com/capsohq/bifrost/framework/vectorstore"
)

// TestMain drops the shared test namespace BEFORE the run starts (in case a
// previous run was interrupted and left stale entries) AND once after — both
// matter: tests share one namespace + one cache_key prefix per t.Name(),
// so stale writes from a prior interrupted run would surface as spurious
// cache hits on the first request of the next run.
func TestMain(m *testing.M) {
	dropSharedTestNamespace() // pre-run sweep
	code := m.Run()
	dropSharedTestNamespace() // post-run sweep
	os.Exit(code)
}

func dropSharedTestNamespace() {
	logger := bifrost.NewDefaultLogger(schemas.LogLevelError)
	stores := []struct {
		storeType vectorstore.VectorStoreType
		config    interface{}
	}{
		{vectorstore.VectorStoreTypeWeaviate, getWeaviateConfigFromEnv()},
		{vectorstore.VectorStoreTypeRedis, getRedisConfigFromEnv()},
		{vectorstore.VectorStoreTypeQdrant, getQdrantConfigFromEnv()},
	}
	for _, candidate := range stores {
		store, err := vectorstore.NewVectorStore(context.Background(), &vectorstore.Config{
			Type:    candidate.storeType,
			Config:  candidate.config,
			Enabled: true,
		}, logger)
		if err != nil {
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		_ = store.DeleteNamespace(ctx, SharedTestNamespace)
		cancel()
	}
}
