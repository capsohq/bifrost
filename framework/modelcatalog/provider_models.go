package modelcatalog

import (
	"context"
	"slices"
	"time"

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

func (mc *ModelCatalog) ensureProviderModelStateLocked() {
	if mc.providerModelSnapshots == nil {
		mc.providerModelSnapshots = make(map[schemas.ModelProvider][]string)
	}
	if mc.providerModelSources == nil {
		mc.providerModelSources = make(map[schemas.ModelProvider]ProviderModelSource)
	}
	if mc.unfilteredProviderModelSources == nil {
		mc.unfilteredProviderModelSources = make(map[schemas.ModelProvider]ProviderModelSource)
	}
	if mc.providerModelHealth == nil {
		mc.providerModelHealth = make(map[schemas.ModelProvider]providerModelHealthState)
	}
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
	mc.ensureProviderModelStateLocked()

	for provider, models := range snapshots {
		if len(models) == 0 {
			continue
		}
		modelsClone := slices.Clone(models)
		mc.providerModelSnapshots[provider] = modelsClone
		mc.live.Upsert(provider, "", false, modelsClone)
		mc.live.Upsert(provider, "", true, modelsClone)
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

// DeleteModelDataForProvider preserves the pre-refactor fork API by clearing
// the new live model cache for the provider.
func (mc *ModelCatalog) DeleteModelDataForProvider(provider schemas.ModelProvider) {
	mc.live.InvalidateProvider(provider)
}

// UpsertModelDataForProvider preserves the pre-refactor fork API while routing
// data into upstream's live.Store.
func (mc *ModelCatalog) UpsertModelDataForProvider(provider schemas.ModelProvider, modelData *schemas.BifrostListModelsResponse, allowedModels []schemas.Model) {
	if modelData == nil {
		return
	}

	datasheetModels := mc.datasheet.DatasheetModelsForProvider(provider)
	var finalModelList []string
	switch {
	case len(modelData.Data) == 0 && len(allowedModels) == 0:
		finalModelList = slices.Clone(datasheetModels)
	case len(modelData.Data) == 0:
		finalModelList = extractModelIDs(&schemas.BifrostListModelsResponse{Data: allowedModels}, provider)
	default:
		finalModelList = extractModelIDs(modelData, provider)
		if len(allowedModels) == 0 {
			finalModelList = appendUniqueModels(finalModelList, datasheetModels)
		}
	}

	mc.live.Upsert(provider, "", false, finalModelList)

	mc.mu.Lock()
	mc.ensureProviderModelStateLocked()
	mc.providerModelSnapshots[provider] = slices.Clone(finalModelList)
	mc.providerModelSources[provider] = ProviderModelSourceLiveDiscovery
	mc.updateProviderModelHealthSnapshotUpdatedAtLocked(provider, time.Now().UTC())
	mc.mu.Unlock()
	mc.persistProviderModelSnapshot(provider, finalModelList)
	mc.persistProviderModelHealthState()
}

// UpsertUnfilteredModelDataForProvider preserves the pre-refactor fork API
// while routing data into upstream's live.Store.
func (mc *ModelCatalog) UpsertUnfilteredModelDataForProvider(provider schemas.ModelProvider, modelData *schemas.BifrostListModelsResponse) {
	if modelData == nil {
		return
	}

	models := appendUniqueModels(mc.datasheet.DatasheetModelsForProvider(provider), extractModelIDs(modelData, provider))
	mc.live.Upsert(provider, "", true, models)

	mc.mu.Lock()
	mc.ensureProviderModelStateLocked()
	mc.unfilteredProviderModelSources[provider] = ProviderModelSourceLiveDiscovery
	mc.updateProviderModelHealthSnapshotUpdatedAtLocked(provider, time.Now().UTC())
	mc.mu.Unlock()
	mc.persistProviderModelHealthState()
}
