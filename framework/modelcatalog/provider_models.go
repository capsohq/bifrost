package modelcatalog

import (
	"context"
	"slices"

	"github.com/capsohq/bifrost/core/schemas"
)

type providerModelStore interface {
	GetAllProviderModelNames(ctx context.Context) (map[schemas.ModelProvider][]string, error)
	ReplaceProviderModelNames(ctx context.Context, provider schemas.ModelProvider, models []string) error
}

func (mc *ModelCatalog) getProviderModelStore() (providerModelStore, bool) {
	if mc.configStore == nil {
		return nil, false
	}
	store, ok := mc.configStore.(providerModelStore)
	return store, ok
}

func (mc *ModelCatalog) loadProviderModelSnapshots(ctx context.Context) {
	store, ok := mc.getProviderModelStore()
	if !ok {
		return
	}

	snapshots, err := store.GetAllProviderModelNames(ctx)
	if err != nil {
		mc.logger.Warn("failed to load provider model snapshots: %v", err)
		return
	}

	mc.mu.Lock()
	defer mc.mu.Unlock()

	for provider, models := range snapshots {
		if len(models) == 0 {
			continue
		}
		modelsClone := slices.Clone(models)
		mc.providerModelSnapshots[provider] = modelsClone
		mc.modelPool[provider] = slices.Clone(modelsClone)
		mc.unfilteredModelPool[provider] = slices.Clone(modelsClone)
		mc.providerModelSources[provider] = ProviderModelSourcePersistedSnapshot
		mc.unfilteredProviderModelSources[provider] = ProviderModelSourcePersistedSnapshot
	}
}

func (mc *ModelCatalog) persistProviderModelSnapshot(provider schemas.ModelProvider, models []string) {
	if len(models) == 0 {
		return
	}

	store, ok := mc.getProviderModelStore()
	if !ok {
		return
	}

	if err := store.ReplaceProviderModelNames(context.Background(), provider, models); err != nil {
		mc.logger.Warn("failed to persist provider model snapshot for %s: %v", provider, err)
	}
}
