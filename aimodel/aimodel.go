package ecologits_go

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"github.com/omegabytes/ecologits-go/impact"
)

type AIModelIface interface {
	Provider() Provider
	Name() string
	Architecture() Architecture
	Sources() []string
	Warnings() []Warning
	ComputeImpacts(float64, float64, string) impact.Impacts
}

var _ AIModelIface = &AIModel{}

type AIModel struct {
	name         string
	provider     Provider
	architecture Architecture
	warnings     []Warning
	sources      []string

	// activeParamCounts: Number of active parameters of the model (in billions).
	activeParamCounts impact.RangeValue

	// totalParamCounts: Number of parameters of the model (in billions).
	totalParamCounts impact.RangeValue

	quantizationBits float64
}

type Provider string

const (
	Anthropic      Provider = "anthropic"
	Mistralai      Provider = "mistralai"
	OpenAI         Provider = "openai"
	HuggingfaceHub Provider = "huggingface_hub"
	Cohere         Provider = "cohere"
	Google         Provider = "google"
)

type Warning struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type Alias struct {
	Provider string `json:"provider"`
	Name     string `json:"name"`
	Alias    string `json:"alias"`
}

type ModelsData struct {
	Aliases []Alias   `json:"aliases,omitempty"`
	Models  []AIModel `json:"models"`
}

type ArchitectureType string

const (
	DENSE ArchitectureType = "dense"
	MOE   ArchitectureType = "moe"
)

type ParametersMoE struct {
	Total  impact.RangeValue `json:"total"`
	Active impact.RangeValue `json:"active"`
}

type Architecture struct {
	Type       ArchitectureType `json:"type"`
	Parameters any              `json:"parameters"`
}

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
	source := "aimodel/aimodels.json"

	// todo: provider name is unused. Can a model be used with multiple providers?
	models, err := FetchAIModels(source)
	if err != nil {
		// slog.Error("failed to fetch AI models", "err", err.Error())
		fmt.Println("failed to fetch AI models")
	}
	modelsMap, err := CreateModelsMap(models)
	if err != nil {
		// slog.Error("failed to map AI models", err.Error())
		fmt.Println("failed to create models map")
		return nil, err
	}
	model := modelsMap[name]
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

func (a *AIModel) ComputeImpacts(outputTokenCount float64, requestLatency float64, geo string) impact.Impacts {
	modelRequiredMemory := a.modelRequiredMemory()
	electricityMix := impact.GetElectricityMix(geo)
	return impact.ComputeImpacts(a.activeParamCounts, modelRequiredMemory, outputTokenCount, requestLatency, electricityMix)
}

// ModelRequiredMemory computes the required memory to load the model on GPU.
//
// Args:
//
//	modelTotalParameterCount: Number of parameters of the model (in billion).
//	modelQuantizationBits: Number of bits used to represent the model weights.
//
// Returns:
//
//	The amount of required GPU memory to load the model.
func (a *AIModel) modelRequiredMemory() float64 {
	// todo: totalParamCounts was a range but I can't find where it gets flattened
	return 1.2 * a.totalParamCounts.Max * a.quantizationBits / 8
}

func FetchAIModels(source string) (*ModelsData, error) {
	// parsedURL, err := url2.Parse(source)
	// resp, err := http.Get(url)
	// if err != nil {
	//	panic(err)
	// }
	// defer resp.Body.Close()
	//
	// body, err := io.ReadAll(resp.Body)
	// if err != nil {
	//	panic(err)
	// }

	data, err := os.ReadFile(source)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	var models ModelsData
	if err := json.Unmarshal(data, &models); err != nil {
		panic(err)
	}

	slog.Info("Loaded AI model data", "count", len(models.Models))
	return &models, nil
}

func CreateModelsMap(models *ModelsData) (map[string]AIModel, error) {
	// todo: review this you were tired
	// todo: refactor aliases
	modelsMap := make(map[string]AIModel)
	for _, model := range models.Models {
		modelsMap[model.name] = model
	}
	return modelsMap, nil
}
