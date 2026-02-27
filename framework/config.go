package framework

import "github.com/capsohq/bifrost/framework/modelcatalog"

// FrameworkConfig represents the configuration for the framework.
type FrameworkConfig struct {
	Pricing *modelcatalog.Config `json:"pricing,omitempty"`
}
