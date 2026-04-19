package mistral

import (
	"fmt"
	"strings"

	providerUtils "github.com/capsohq/bifrost/core/providers/utils"
	"github.com/capsohq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// MistralErrorResponse captures both Mistral's top-level error shape and nested OpenAI-style errors.
type MistralErrorResponse struct {
	Object  string              `json:"object,omitempty"`
	Message string              `json:"message,omitempty"`
	Type    string              `json:"type,omitempty"`
	Code    string              `json:"code,omitempty"`
	Error   *schemas.ErrorField `json:"error,omitempty"`
}

// ParseMistralError parses Mistral-specific error responses.
func ParseMistralError(resp *fasthttp.Response) *schemas.BifrostError {
	return ParseMistralErrorWithMetadata(resp, schemas.Mistral, "", "")
}

// ParseMistralErrorWithMetadata parses a Mistral error response and stamps
// provider/request metadata when the caller has that context available.
func ParseMistralErrorWithMetadata(
	resp *fasthttp.Response,
	providerName schemas.ModelProvider,
	requestType schemas.RequestType,
	requestedModel string,
) *schemas.BifrostError {
	var errorResp MistralErrorResponse
	bifrostErr := providerUtils.HandleProviderAPIError(resp, &errorResp)
	if bifrostErr == nil {
		return nil
	}

	if bifrostErr.Error == nil {
		bifrostErr.Error = &schemas.ErrorField{}
	}

	if errorResp.Error != nil {
		if strings.TrimSpace(errorResp.Error.Message) != "" {
			bifrostErr.Error.Message = errorResp.Error.Message
		}
		if errorResp.Error.Type != nil && strings.TrimSpace(*errorResp.Error.Type) != "" {
			bifrostErr.Error.Type = errorResp.Error.Type
			bifrostErr.Type = errorResp.Error.Type
		}
		if errorResp.Error.Code != nil && strings.TrimSpace(*errorResp.Error.Code) != "" {
			bifrostErr.Error.Code = errorResp.Error.Code
		}
		bifrostErr.Error.Param = errorResp.Error.Param
		if errorResp.Error.EventID != nil {
			bifrostErr.Error.EventID = errorResp.Error.EventID
		}
	}

	if strings.TrimSpace(errorResp.Message) != "" {
		bifrostErr.Error.Message = errorResp.Message
	}
	if strings.TrimSpace(errorResp.Type) != "" {
		errorType := schemas.Ptr(errorResp.Type)
		bifrostErr.Error.Type = errorType
		bifrostErr.Type = errorType
	}
	if strings.TrimSpace(errorResp.Code) != "" {
		bifrostErr.Error.Code = schemas.Ptr(errorResp.Code)
	}

	if strings.TrimSpace(bifrostErr.Error.Message) == "" {
		if bifrostErr.StatusCode != nil {
			bifrostErr.Error.Message = fmt.Sprintf("provider API error (status %d)", *bifrostErr.StatusCode)
		} else {
			bifrostErr.Error.Message = "provider API error"
		}
	}

	if providerName != "" {
		bifrostErr.ExtraFields.Provider = providerName
	}
	if requestType != "" {
		bifrostErr.ExtraFields.RequestType = requestType
	}
	if strings.TrimSpace(requestedModel) != "" {
		bifrostErr.ExtraFields.OriginalModelRequested = requestedModel
	}

	return bifrostErr
}
