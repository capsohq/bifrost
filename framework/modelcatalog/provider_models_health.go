package modelcatalog

import (
	"context"
	"errors"
	"sort"
	"time"

	"github.com/bytedance/sonic"
	"github.com/capsohq/bifrost/core/schemas"
	"github.com/capsohq/bifrost/framework/configstore"
	configstoreTables "github.com/capsohq/bifrost/framework/configstore/tables"
	"gorm.io/gorm"
)

const DefaultProviderModelSnapshotStaleAfter = 24 * time.Hour
const DefaultProviderModelHealthPersistDebounce = 500 * time.Millisecond

type ProviderModelSource string

const (
	ProviderModelSourceUnknown           ProviderModelSource = "unknown"
	ProviderModelSourcePricingCatalog    ProviderModelSource = "pricing_catalog"
	ProviderModelSourcePersistedSnapshot ProviderModelSource = "persisted_snapshot"
	ProviderModelSourceDefaultSeed       ProviderModelSource = "default_seed"
	ProviderModelSourceAllowedModels     ProviderModelSource = "allowed_models"
	ProviderModelSourceLiveDiscovery     ProviderModelSource = "live_discovery"
)

type ProviderModelHealthStatus string

const (
	ProviderModelHealthUnknown  ProviderModelHealthStatus = "unknown"
	ProviderModelHealthHealthy  ProviderModelHealthStatus = "healthy"
	ProviderModelHealthStale    ProviderModelHealthStatus = "stale"
	ProviderModelHealthError    ProviderModelHealthStatus = "error"
	ProviderModelHealthDegraded ProviderModelHealthStatus = "degraded"
)

type providerDiscoveryState struct {
	LastAttemptAt   time.Time
	LastSuccessAt   time.Time
	LastErrorAt     time.Time
	LastError       string
	LastModelsCount int
}

type providerModelHealthState struct {
	Filtered            providerDiscoveryState
	Unfiltered          providerDiscoveryState
	LastSnapshotUpdated time.Time
}

type ProviderModelDiscoveryHealth struct {
	Status          ProviderModelHealthStatus `json:"status"`
	LastAttemptAt   *time.Time                `json:"last_attempt_at,omitempty"`
	LastSuccessAt   *time.Time                `json:"last_success_at,omitempty"`
	LastErrorAt     *time.Time                `json:"last_error_at,omitempty"`
	LastError       string                    `json:"last_error,omitempty"`
	LastModelsCount int                       `json:"last_models_count"`
}

type ProviderModelSnapshotHealth struct {
	Provider             schemas.ModelProvider        `json:"provider"`
	Status               ProviderModelHealthStatus    `json:"status"`
	SnapshotModelCount   int                          `json:"snapshot_model_count"`
	FilteredModelCount   int                          `json:"filtered_model_count"`
	UnfilteredModelCount int                          `json:"unfiltered_model_count"`
	FilteredSource       ProviderModelSource          `json:"filtered_source"`
	UnfilteredSource     ProviderModelSource          `json:"unfiltered_source"`
	LastSnapshotUpdated  *time.Time                   `json:"last_snapshot_updated,omitempty"`
	FilteredDiscovery    ProviderModelDiscoveryHealth `json:"filtered_discovery"`
	UnfilteredDiscovery  ProviderModelDiscoveryHealth `json:"unfiltered_discovery"`
}

type ProviderModelSnapshotHealthSummary struct {
	TotalProviders    int `json:"total_providers"`
	HealthyProviders  int `json:"healthy_providers"`
	StaleProviders    int `json:"stale_providers"`
	ErrorProviders    int `json:"error_providers"`
	DegradedProviders int `json:"degraded_providers"`
	UnknownProviders  int `json:"unknown_providers"`
}

type ProviderModelSnapshotHealthReport struct {
	Status            ProviderModelHealthStatus          `json:"status"`
	GeneratedAt       time.Time                          `json:"generated_at"`
	StaleAfterSeconds int64                              `json:"stale_after_seconds"`
	Summary           ProviderModelSnapshotHealthSummary `json:"summary"`
	Providers         []ProviderModelSnapshotHealth      `json:"providers"`
}

type providerModelHealthStore interface {
	GetConfig(ctx context.Context, key string) (*configstoreTables.TableGovernanceConfig, error)
	UpdateConfig(ctx context.Context, config *configstoreTables.TableGovernanceConfig, tx ...*gorm.DB) error
}

type persistedProviderModelHealthState struct {
	Filtered            providerDiscoveryState `json:"filtered"`
	Unfiltered          providerDiscoveryState `json:"unfiltered"`
	LastSnapshotUpdated time.Time              `json:"last_snapshot_updated,omitempty"`
	FilteredSource      ProviderModelSource    `json:"filtered_source,omitempty"`
	UnfilteredSource    ProviderModelSource    `json:"unfiltered_source,omitempty"`
}

// RecordProviderModelDiscoveryResult records one provider model listing attempt (filtered or unfiltered).
func (mc *ModelCatalog) RecordProviderModelDiscoveryResult(
	provider schemas.ModelProvider,
	unfiltered bool,
	modelData *schemas.BifrostListModelsResponse,
	discoveryErr *schemas.BifrostError,
) {
	mc.mu.Lock()
	now := time.Now().UTC()
	state := mc.providerModelHealth[provider]
	target := &state.Filtered
	if unfiltered {
		target = &state.Unfiltered
	}

	target.LastAttemptAt = now
	if discoveryErr != nil {
		target.LastErrorAt = now
		target.LastError = extractDiscoveryErrorMessage(discoveryErr)
	} else {
		target.LastSuccessAt = now
		target.LastErrorAt = time.Time{}
		target.LastError = ""
		target.LastModelsCount = countProviderModelsInResponse(provider, modelData)
	}

	mc.providerModelHealth[provider] = state
	mc.mu.Unlock()

	mc.persistProviderModelHealthState()
}

func (mc *ModelCatalog) updateProviderModelHealthSnapshotUpdatedAtLocked(provider schemas.ModelProvider, updatedAt time.Time) {
	state := mc.providerModelHealth[provider]
	state.LastSnapshotUpdated = updatedAt
	mc.providerModelHealth[provider] = state
}

func countProviderModelsInResponse(provider schemas.ModelProvider, modelData *schemas.BifrostListModelsResponse) int {
	if modelData == nil || len(modelData.Data) == 0 {
		return 0
	}

	count := 0
	seenModels := make(map[string]struct{}, len(modelData.Data))
	for _, model := range modelData.Data {
		parsedProvider, parsedModel := schemas.ParseModelString(model.ID, provider)
		if parsedProvider != provider {
			continue
		}
		if _, exists := seenModels[parsedModel]; exists {
			continue
		}
		seenModels[parsedModel] = struct{}{}
		count++
	}
	return count
}

func extractDiscoveryErrorMessage(discoveryErr *schemas.BifrostError) string {
	if discoveryErr == nil {
		return ""
	}
	if discoveryErr.Error != nil {
		if discoveryErr.Error.Message != "" {
			return discoveryErr.Error.Message
		}
		if discoveryErr.Error.Error != nil {
			return discoveryErr.Error.Error.Error()
		}
	}
	return "model discovery failed"
}

func (mc *ModelCatalog) GetProviderModelSnapshotHealthReport() ProviderModelSnapshotHealthReport {
	now := time.Now().UTC()

	mc.mu.RLock()
	providerSet := make(map[schemas.ModelProvider]struct{})
	for provider := range mc.modelPool {
		providerSet[provider] = struct{}{}
	}
	for provider := range mc.unfilteredModelPool {
		providerSet[provider] = struct{}{}
	}
	for provider := range mc.providerModelSnapshots {
		providerSet[provider] = struct{}{}
	}
	for provider := range mc.providerModelHealth {
		providerSet[provider] = struct{}{}
	}
	for provider := range mc.providerModelSources {
		providerSet[provider] = struct{}{}
	}
	for provider := range mc.unfilteredProviderModelSources {
		providerSet[provider] = struct{}{}
	}

	providers := make([]schemas.ModelProvider, 0, len(providerSet))
	for provider := range providerSet {
		providers = append(providers, provider)
	}
	sort.Slice(providers, func(i, j int) bool {
		return providers[i] < providers[j]
	})

	items := make([]ProviderModelSnapshotHealth, 0, len(providers))
	summary := ProviderModelSnapshotHealthSummary{}
	for _, provider := range providers {
		filteredSource := mc.providerModelSources[provider]
		if filteredSource == "" {
			filteredSource = ProviderModelSourceUnknown
		}
		unfilteredSource := mc.unfilteredProviderModelSources[provider]
		if unfilteredSource == "" {
			unfilteredSource = ProviderModelSourceUnknown
		}

		state := mc.providerModelHealth[provider]
		filteredDiscovery := toProviderModelDiscoveryHealth(state.Filtered, now)
		unfilteredDiscovery := toProviderModelDiscoveryHealth(state.Unfiltered, now)
		status := mergeProviderHealthStatus(filteredDiscovery.Status, unfilteredDiscovery.Status)

		item := ProviderModelSnapshotHealth{
			Provider:             provider,
			Status:               status,
			SnapshotModelCount:   len(mc.providerModelSnapshots[provider]),
			FilteredModelCount:   len(mc.modelPool[provider]),
			UnfilteredModelCount: len(mc.unfilteredModelPool[provider]),
			FilteredSource:       filteredSource,
			UnfilteredSource:     unfilteredSource,
			FilteredDiscovery:    filteredDiscovery,
			UnfilteredDiscovery:  unfilteredDiscovery,
		}
		if !state.LastSnapshotUpdated.IsZero() {
			lastSnapshotUpdated := state.LastSnapshotUpdated
			item.LastSnapshotUpdated = &lastSnapshotUpdated
		}
		items = append(items, item)

		summary.TotalProviders++
		switch status {
		case ProviderModelHealthHealthy:
			summary.HealthyProviders++
		case ProviderModelHealthStale:
			summary.StaleProviders++
		case ProviderModelHealthError:
			summary.ErrorProviders++
		case ProviderModelHealthDegraded:
			summary.DegradedProviders++
		default:
			summary.UnknownProviders++
		}
	}
	mc.mu.RUnlock()

	reportStatus := ProviderModelHealthUnknown
	switch {
	case summary.TotalProviders == 0:
		reportStatus = ProviderModelHealthUnknown
	case summary.ErrorProviders > 0:
		reportStatus = ProviderModelHealthError
	case summary.StaleProviders > 0 || summary.DegradedProviders > 0:
		reportStatus = ProviderModelHealthDegraded
	case summary.HealthyProviders > 0:
		reportStatus = ProviderModelHealthHealthy
	}

	return ProviderModelSnapshotHealthReport{
		Status:            reportStatus,
		GeneratedAt:       now,
		StaleAfterSeconds: int64(DefaultProviderModelSnapshotStaleAfter.Seconds()),
		Summary:           summary,
		Providers:         items,
	}
}

func mergeProviderHealthStatus(filtered ProviderModelHealthStatus, unfiltered ProviderModelHealthStatus) ProviderModelHealthStatus {
	if filtered == ProviderModelHealthError || unfiltered == ProviderModelHealthError {
		return ProviderModelHealthError
	}
	if filtered == ProviderModelHealthStale || unfiltered == ProviderModelHealthStale {
		return ProviderModelHealthStale
	}
	if filtered == ProviderModelHealthUnknown && unfiltered == ProviderModelHealthUnknown {
		return ProviderModelHealthUnknown
	}
	if filtered == ProviderModelHealthUnknown || unfiltered == ProviderModelHealthUnknown {
		return ProviderModelHealthDegraded
	}
	return ProviderModelHealthHealthy
}

func toProviderModelDiscoveryHealth(state providerDiscoveryState, now time.Time) ProviderModelDiscoveryHealth {
	status := ProviderModelHealthUnknown
	switch {
	case state.LastAttemptAt.IsZero():
		status = ProviderModelHealthUnknown
	case !state.LastErrorAt.IsZero() && (state.LastSuccessAt.IsZero() || !state.LastErrorAt.Before(state.LastSuccessAt)):
		status = ProviderModelHealthError
	case state.LastSuccessAt.IsZero():
		status = ProviderModelHealthUnknown
	case now.Sub(state.LastSuccessAt) > DefaultProviderModelSnapshotStaleAfter:
		status = ProviderModelHealthStale
	default:
		status = ProviderModelHealthHealthy
	}

	discoveryHealth := ProviderModelDiscoveryHealth{
		Status:          status,
		LastModelsCount: state.LastModelsCount,
		LastError:       state.LastError,
	}
	if !state.LastAttemptAt.IsZero() {
		lastAttemptAt := state.LastAttemptAt
		discoveryHealth.LastAttemptAt = &lastAttemptAt
	}
	if !state.LastSuccessAt.IsZero() {
		lastSuccessAt := state.LastSuccessAt
		discoveryHealth.LastSuccessAt = &lastSuccessAt
	}
	if !state.LastErrorAt.IsZero() {
		lastErrorAt := state.LastErrorAt
		discoveryHealth.LastErrorAt = &lastErrorAt
	}
	return discoveryHealth
}

func (mc *ModelCatalog) getProviderModelHealthStore() (providerModelHealthStore, bool) {
	if mc.configStore == nil {
		return nil, false
	}
	store, ok := mc.configStore.(providerModelHealthStore)
	return store, ok
}

func (mc *ModelCatalog) loadProviderModelHealthState(ctx context.Context) {
	store, ok := mc.getProviderModelHealthStore()
	if !ok {
		return
	}

	config, err := store.GetConfig(ctx, ConfigProviderModelHealthStateKey)
	if err != nil {
		if errors.Is(err, configstore.ErrNotFound) {
			return
		}
		mc.logger.Warn("failed to load provider model health state: %v", err)
		return
	}
	if config == nil || config.Value == "" {
		return
	}

	var persistedState map[string]persistedProviderModelHealthState
	if err := sonic.Unmarshal([]byte(config.Value), &persistedState); err != nil {
		mc.logger.Warn("failed to parse provider model health state: %v", err)
		return
	}

	mc.mu.Lock()
	defer mc.mu.Unlock()
	for providerName, entry := range persistedState {
		provider := schemas.ModelProvider(providerName)
		mc.providerModelHealth[provider] = providerModelHealthState{
			Filtered:            entry.Filtered,
			Unfiltered:          entry.Unfiltered,
			LastSnapshotUpdated: entry.LastSnapshotUpdated,
		}
		if entry.FilteredSource != "" {
			mc.providerModelSources[provider] = entry.FilteredSource
		}
		if entry.UnfilteredSource != "" {
			mc.unfilteredProviderModelSources[provider] = entry.UnfilteredSource
		}
	}
}

func (mc *ModelCatalog) persistProviderModelHealthState() {
	if !mc.shouldPersistProviderModelHealthState() {
		return
	}

	if mc.providerModelHealthPersistSignal == nil {
		mc.persistProviderModelHealthStateNow()
		return
	}

	select {
	case mc.providerModelHealthPersistSignal <- struct{}{}:
	default:
	}
}

func (mc *ModelCatalog) shouldPersistProviderModelHealthState() bool {
	if mc.providerModelHealthPersistCallback != nil {
		return true
	}
	_, ok := mc.getProviderModelHealthStore()
	return ok
}

func (mc *ModelCatalog) persistProviderModelHealthStateNow() {
	if mc.providerModelHealthPersistCallback != nil {
		mc.providerModelHealthPersistCallback()
		return
	}

	store, ok := mc.getProviderModelHealthStore()
	if !ok {
		return
	}

	payload := mc.getPersistedProviderModelHealthState()
	payloadJSON, err := sonic.Marshal(payload)
	if err != nil {
		mc.logger.Warn("failed to marshal provider model health state: %v", err)
		return
	}

	if err := store.UpdateConfig(context.Background(), &configstoreTables.TableGovernanceConfig{
		Key:   ConfigProviderModelHealthStateKey,
		Value: string(payloadJSON),
	}); err != nil {
		mc.logger.Warn("failed to persist provider model health state: %v", err)
	}
}

func (mc *ModelCatalog) startProviderModelHealthPersistWorker(ctx context.Context) {
	mc.providerModelHealthPersistSignal = make(chan struct{}, 1)
	mc.wg.Add(1)
	go mc.providerModelHealthPersistWorker(ctx)
}

func (mc *ModelCatalog) providerModelHealthPersistWorker(ctx context.Context) {
	defer mc.wg.Done()

	var timer *time.Timer
	var timerC <-chan time.Time
	pending := false

	stopTimer := func() {
		if timer == nil {
			return
		}
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
	}
	drainSignals := func() {
		for {
			select {
			case <-mc.providerModelHealthPersistSignal:
				pending = true
			default:
				return
			}
		}
	}
	flushPending := func() {
		if !pending {
			return
		}
		mc.persistProviderModelHealthStateNow()
		pending = false
	}

	for {
		select {
		case <-ctx.Done():
			drainSignals()
			flushPending()
			stopTimer()
			return
		case <-mc.done:
			drainSignals()
			flushPending()
			stopTimer()
			return
		case <-mc.providerModelHealthPersistSignal:
			pending = true
			debounce := mc.getProviderModelHealthPersistDebounce()
			if timer == nil {
				timer = time.NewTimer(debounce)
			} else {
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
				timer.Reset(debounce)
			}
			timerC = timer.C
		case <-timerC:
			flushPending()
			timerC = nil
		}
	}
}

func (mc *ModelCatalog) getPersistedProviderModelHealthState() map[string]persistedProviderModelHealthState {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	providers := make(map[schemas.ModelProvider]struct{})
	for provider := range mc.providerModelHealth {
		providers[provider] = struct{}{}
	}
	for provider := range mc.providerModelSources {
		providers[provider] = struct{}{}
	}
	for provider := range mc.unfilteredProviderModelSources {
		providers[provider] = struct{}{}
	}

	persistedState := make(map[string]persistedProviderModelHealthState, len(providers))
	for provider := range providers {
		entry := persistedProviderModelHealthState{
			Filtered:            mc.providerModelHealth[provider].Filtered,
			Unfiltered:          mc.providerModelHealth[provider].Unfiltered,
			LastSnapshotUpdated: mc.providerModelHealth[provider].LastSnapshotUpdated,
			FilteredSource:      mc.providerModelSources[provider],
			UnfilteredSource:    mc.unfilteredProviderModelSources[provider],
		}
		if isPersistedProviderModelHealthStateEmpty(entry) {
			continue
		}
		persistedState[string(provider)] = entry
	}

	return persistedState
}

func isPersistedProviderModelHealthStateEmpty(entry persistedProviderModelHealthState) bool {
	if !entry.Filtered.LastAttemptAt.IsZero() ||
		!entry.Filtered.LastSuccessAt.IsZero() ||
		!entry.Filtered.LastErrorAt.IsZero() ||
		entry.Filtered.LastError != "" ||
		entry.Filtered.LastModelsCount != 0 {
		return false
	}
	if !entry.Unfiltered.LastAttemptAt.IsZero() ||
		!entry.Unfiltered.LastSuccessAt.IsZero() ||
		!entry.Unfiltered.LastErrorAt.IsZero() ||
		entry.Unfiltered.LastError != "" ||
		entry.Unfiltered.LastModelsCount != 0 {
		return false
	}
	if !entry.LastSnapshotUpdated.IsZero() {
		return false
	}
	if entry.FilteredSource != "" || entry.UnfilteredSource != "" {
		return false
	}
	return true
}
