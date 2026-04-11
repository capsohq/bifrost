package modelcatalog

import (
	"time"
)

const (
	DefaultPricingSyncInterval        = 24 * time.Hour
	MinimumPricingSyncIntervalSec     = int64(3600)
	ConfigLastPricingSyncKey          = "LastModelPricingSync"
	ConfigLastParamsSyncKey           = "LastModelParametersSync"
	ConfigProviderModelHealthStateKey = "ProviderModelHealthStateV1"
	DefaultPricingURL                 = "https://getbifrost.ai/datasheet"
	DefaultModelParametersURL         = "https://getbifrost.ai/datasheet/model-parameters"
	DefaultPricingTimeout             = 45 * time.Second
	DefaultModelParametersTimeout     = 45 * time.Second

	// syncWorkerTickerPeriod is the fixed interval at which the background sync worker
	// wakes up to check whether a sync is due. This is independent of pricingSyncInterval.
	// The ticker defines the check granularity, not the sync frequency.
	syncWorkerTickerPeriod = 1 * time.Hour
)

// Config is the model pricing configuration.
type Config struct {
	PricingURL                         *string        `json:"pricing_url,omitempty"`
	PricingSyncInterval                *int64         `json:"pricing_sync_interval,omitempty"` // seconds
	ProviderModelHealthPersistDebounce *time.Duration `json:"provider_model_health_persist_debounce_ms,omitempty"`
}
