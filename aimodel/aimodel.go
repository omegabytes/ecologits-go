package aimodel

import (
	"errors"
	"fmt"
	"log/slog"
	"os"

	"github.com/omegabytes/ecologits-go/common"
	"github.com/valyala/fastjson"
)

type AIModelIface interface {
	Provider() Provider
	Name() string
	Architecture() Architecture
	Sources() []string
	Warnings() []Warning
}

var _ AIModelIface = &AIModel{}

// AIModel represents an AI model with its properties.
type AIModel struct {
	name             string
	provider         Provider
	architecture     Architecture
	warnings         []Warning
	sources          []string
	quantizationBits float64
}

// Provider is the name of the provider, eg "openai", "anthropic".
type Provider string

const (
	Anthropic      Provider = "anthropic"
	Mistralai      Provider = "mistralai"
	OpenAI         Provider = "openai"
	HuggingfaceHub Provider = "huggingface_hub"
	Cohere         Provider = "cohere"
	Google         Provider = "google"
)

// Warning represents a warning message associated with the model.
type Warning struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// Alias is a model alias, used to map different names to the same model.
type Alias struct {
	Provider string `json:"provider"`
	Name     string `json:"name"`
	Alias    string `json:"alias"`
}

// ModelData is the top-level structure for the model used in the data file.
type ModelData struct {
	Aliases []Alias   `json:"aliases,omitempty"`
	Models  []AIModel `json:"models"`
}

// ArchitectureType is the type of architecture used in the model.
type ArchitectureType string

const (
	DENSE ArchitectureType = "dense"
	MOE   ArchitectureType = "moe"
)

// Architecture represents the model architecture.
type Architecture struct {
	Type       ArchitectureType `json:"type"`
	Parameters Parameters
}

// Parameters represents the parameters of the model.
type Parameters struct {
	// Active: Number of active parameters of the model (in billions).
	Active common.RangeValue `json:"active"`
	// Total: Number of parameters of the model (in billions).
	Total common.RangeValue `json:"total"`
}

// NewAIModel creates a new AIModel instance based on the provided name and provider.
func NewAIModel(
	name string,
	provider string,
) (*AIModel, error) {
	switch {
	case name == "":
		return nil, errors.New("name cannot be empty")
	case provider == "":
		return nil, errors.New("provider cannot be empty")
	}

	// todo: fetch model data from API
	// source := "https://raw.githubusercontent.com/genai-impact/ecologits/main/data/models.json"
	source := "aimodel/data/aimodels.json"

	// todo: provider name is unused. Can a model be used with multiple providers?
	models, err := FetchAIModels(source)
	if err != nil {
		slog.Error("failed to fetch AI models", "err", err)
		return nil, err
	}
	modelsMap, err := CreateModelsMap(models)
	if err != nil {
		slog.Error("failed to map AI models", "error", err)
		return nil, err
	}
	model := modelsMap[name]
	if model.quantizationBits == 0 {
		model.quantizationBits = 8
	}
	return &model, nil
}

func (a *AIModel) Provider() Provider {
	return a.provider
}

func (a *AIModel) Name() string {
	return a.name
}

func (a *AIModel) Architecture() Architecture {
	return a.architecture
}

func (a *AIModel) Sources() []string {
	return a.sources
}

func (a *AIModel) Warnings() []Warning {
	return a.warnings
}

// ModelRequiredMemory returns the required memory to load the model on a GPUModel.
func (a *AIModel) ModelRequiredMemory() float64 {
	return 1.2 * a.architecture.Parameters.Total.Max * a.quantizationBits / 8
}

// FetchAIModels parses and normalizes unstructured json into a list of AIModel objects.
func FetchAIModels(source string) (*ModelData, error) {
	data, err := os.ReadFile(source)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	var p fastjson.Parser
	v, err := p.ParseBytes(data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	models := &ModelData{}
	aliases := v.GetArray("aliases")
	for _, alias := range aliases {
		models.Aliases = append(
			models.Aliases, Alias{
				Provider: string(alias.GetStringBytes("provider")),
				Name:     string(alias.GetStringBytes("name")),
				Alias:    string(alias.GetStringBytes("alias")),
			},
		)
	}

	modelsArray := v.GetArray("models")
	for _, model := range modelsArray {
		aiModel := AIModel{
			name:     string(model.GetStringBytes("name")),
			provider: Provider(model.GetStringBytes("provider")),
			sources:  parseStringArray(model.GetArray("sources")),
			warnings: parseWarnings(model.GetArray("warnings")),
		}

		architecture := model.Get("architecture")
		if architecture != nil {
			aiModel.architecture.Type = ArchitectureType(architecture.GetStringBytes("type"))
			parameters := architecture.Get("parameters")

			parsedParams := Parameters{}
			if parameters.Exists("total") {
				totalVal, err := parseRangeValue(parameters.Get("total"))
				if err != nil {
					slog.Error("failed to parse total value", "error", err, "model-name", aiModel.name)
					return nil, fmt.Errorf("failed to parse min value for total: %w", err)
				}
				parsedParams.Total = totalVal
			}
			if parameters.Exists("active") {
				activeVal, err := parseRangeValue(parameters.Get("active"))
				if err != nil {
					slog.Error("failed to parse active value", "error", err, "model-name", aiModel.name)
					return nil, fmt.Errorf("failed to parse max value for active: %w", err)
				}
				parsedParams.Active = activeVal
			}
			if parameters.Exists("min") || parameters.Exists("max") {
				totalVal, err := parseRangeValue(parameters)
				if err != nil {
					slog.Error("failed to parse", "error", err, "model-name", aiModel.name)
					return nil, fmt.Errorf("failed to parse max value for active: %w", err)
				}
				parsedParams.Total = totalVal
			}
			aiModel.architecture.Parameters = parsedParams
		}
		models.Models = append(models.Models, aiModel)
	}

	return models, nil
}

func parseStringArray(arr []*fastjson.Value) []string {
	result := make([]string, 0)
	for _, v := range arr {
		result = append(result, string(v.GetStringBytes()))
	}
	return result
}

func parseWarnings(arr []*fastjson.Value) []Warning {
	warnings := make([]Warning, 0)
	for _, v := range arr {
		warnings = append(
			warnings, Warning{
				Code:    string(v.GetStringBytes("code")),
				Message: string(v.GetStringBytes("message")),
			},
		)
	}
	return warnings
}

func parseRangeValue(value *fastjson.Value) (common.RangeValue, error) {
	if value == nil {
		return common.RangeValue{}, nil
	}

	switch value.Type() {
	case fastjson.TypeNumber:
		val := value.GetFloat64()
		return common.RangeValue{Min: val, Max: val}, nil
	default:
		return common.RangeValue{}, fmt.Errorf("unexpected type: %s", value.Type())

	case fastjson.TypeObject:
		var parsedRange common.RangeValue
		if !value.Exists("min") && !value.Exists("max") {
			return common.RangeValue{}, fmt.Errorf("min and max values are missing")
		}
		switch {
		case value.Exists("min") && !value.Exists("max"):
			minVal, err := value.Get("min").Float64()
			if err != nil {
				return common.RangeValue{}, fmt.Errorf("failed to parse min value: %w", err)
			}
			parsedRange = common.RangeValue{Min: minVal, Max: minVal}
		case !value.Exists("min") && value.Exists("max"):
			maxVal, err := value.Get("max").Float64()
			if err != nil {
				return common.RangeValue{}, fmt.Errorf("failed to parse max value: %w", err)
			}
			parsedRange = common.RangeValue{Min: maxVal, Max: maxVal}
		default:
			minVal, err := value.Get("min").Float64()
			if err != nil {
				return common.RangeValue{}, fmt.Errorf("failed to parse min value: %w", err)
			}
			maxVal, err := value.Get("max").Float64()
			if err != nil {
				return common.RangeValue{}, fmt.Errorf("failed to parse max value: %w", err)
			}
			return common.RangeValue{Min: minVal, Max: maxVal}, nil
		}
		return parsedRange, nil
	}
}

func CreateModelsMap(models *ModelData) (map[string]AIModel, error) {
	// todo: review this you were tired
	// todo: refactor aliases
	modelsMap := make(map[string]AIModel)
	for _, model := range models.Models {
		modelsMap[model.name] = model
	}
	return modelsMap, nil
}