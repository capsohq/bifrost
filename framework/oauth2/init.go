package oauth2

import "github.com/capsohq/bifrost/core/schemas"

var logger schemas.Logger

func SetLogger(l schemas.Logger) {
	logger = l
}
