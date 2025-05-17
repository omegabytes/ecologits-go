package aimodel

import (
	"os"
	"testing"

	"github.com/omegabytes/ecologits-go/common"
	"github.com/stretchr/testify/assert"
	"github.com/valyala/fastjson"
)

func TestParseRangeValue(t *testing.T) {
	tests := []struct {
		name        string
		jsonInput   string
		expected    common.RangeValue
		expectError bool
	}{
		{
			name:      "Valid range object",
			jsonInput: `{"min": 10.5, "max": 20.5}`,
			expected:  common.RangeValue{Min: 10.5, Max: 20.5},
		},
		{
			name:      "Valid single float value",
			jsonInput: `15.0`,
			expected:  common.RangeValue{Min: 15.0, Max: 15.0},
		},
		{
			name:        "Invalid type (string)",
			jsonInput:   `"invalid"`,
			expectError: true,
		},
		{
			name:      "Invalid range object (missing min)",
			jsonInput: `{"max": 20.5}`,
			expected:  common.RangeValue{Min: 20.5, Max: 20.5},
		},
		{
			name:      "Invalid range object (missing max)",
			jsonInput: `{"min": 10.5}`,
			expected:  common.RangeValue{Min: 10.5, Max: 10.5},
		},
	}

	for _, tt := range tests {
		t.Run(
			tt.name, func(t *testing.T) {
				var p fastjson.Parser
				v, err := p.Parse(tt.jsonInput)
				assert.NoError(t, err)

				result, err := parseRangeValue(v)
				if tt.expectError {
					assert.Error(t, err)
				} else {
					assert.NoError(t, err)
					assert.Equal(t, tt.expected, result)
				}
			},
		)
	}
}

func TestParseWarnings(t *testing.T) {
	tests := []struct {
		name      string
		jsonInput string
		expected  []Warning
	}{
		{
			name: "Valid warnings array",
			jsonInput: `[
				{"code": "warning-1", "message": "This is a warning"},
				{"code": "warning-2", "message": "Another warning"}
			]`,
			expected: []Warning{
				{Code: "warning-1", Message: "This is a warning"},
				{Code: "warning-2", Message: "Another warning"},
			},
		},
		{
			name:      "Empty warnings array",
			jsonInput: `[]`,
			expected:  []Warning{},
		},
		{
			name:      "Null warnings array",
			jsonInput: `null`,
			expected:  []Warning{},
		},
	}

	for _, tt := range tests {
		t.Run(
			tt.name, func(t *testing.T) {
				var p fastjson.Parser
				v, err := p.Parse(tt.jsonInput)
				assert.NoError(t, err)

				result := parseWarnings(v.GetArray())
				assert.Equal(t, tt.expected, result)
			},
		)
	}
}

func TestParseStringArray(t *testing.T) {
	tests := []struct {
		name      string
		jsonInput string
		expected  []string
	}{
		{
			name:      "Valid string array",
			jsonInput: `["string1", "string2", "string3"]`,
			expected:  []string{"string1", "string2", "string3"},
		},
		{
			name:      "Empty string array",
			jsonInput: `[]`,
			expected:  []string{},
		},
		{
			name:      "Null string array",
			jsonInput: `null`,
			expected:  []string{},
		},
	}

	for _, tt := range tests {
		t.Run(
			tt.name, func(t *testing.T) {
				var p fastjson.Parser
				v, err := p.Parse(tt.jsonInput)
				assert.NoError(t, err)

				result := parseStringArray(v.GetArray())
				assert.Equal(t, tt.expected, result)
			},
		)
	}
}

func TestFetchAIModels(t *testing.T) {
	tests := []struct {
		name          string
		jsonContent   string
		expectError   bool
		expectedCount int
		expectedModel *AIModel
	}{
		{
			name: "returns parsed ai model for valid json input",
			jsonContent: `{
				"models": [
					{
						"type": "model",
						"provider": "openai",
						"name": "gpt-4",
						"architecture": {
							"type": "moe",
							"parameters": {
								"total": 1760.8,
								"active": {
									"min": 220.000007,
									"max": 880.534
								}
							}
						},
						"warnings": null,
						"sources": [
							"https://example.com"
						]
					}
				]
			}`,
			expectError:   false,
			expectedCount: 1,
			expectedModel: &AIModel{
				name:     "gpt-4",
				provider: OpenAI,
				architecture: Architecture{
					Type: MOE,
					Parameters: Parameters{
						Total:  common.RangeValue{Min: 1760.8, Max: 1760.8},
						Active: common.RangeValue{Min: 220.000007, Max: 880.534},
					},
				},
			},
		},
		{
			name:          "returns error for empty input",
			jsonContent:   `{}`,
			expectError:   false,
			expectedCount: 0,
		},
		{
			name: "returns error for malformed JSON",
			jsonContent: `{
				"models": [
					{
						"name": "gpt-4",
						"provider": "openai",
						"architecture": {
							"type": "moe",
							"parameters": {
								"total": 1760.8,
								"active": {
									"min": 220.000007,
									"max": 880.534
								}
							}
						}
					}
				`, // missing closing braces
			expectError:   true,
			expectedCount: 0,
		},
		{
			name:          "Empty file",
			jsonContent:   "",
			expectError:   true,
			expectedCount: 0,
		},
		{
			name: "returns unexpected type error when parameter.total value is unsupported",
			jsonContent: `{
				"models": [
					{
						"type": "model",
						"provider": "openai",
						"name": "gpt-4",
						"architecture": {
							"type": "moe",
							"parameters": {
								"total": "not-a-number-or-a-range",
								"active": {
									"min": 220.000007,
									"max": 880.534
								}
							}
						},
						"warnings": null,
						"sources": [
							"https://example.com"
						]
					}
				]
			}`,
			expectError: true,
		},
		{
			name: "returns unexpected type error when parameter.active value is unsupported",
			jsonContent: `{
				"models": [
					{
						"type": "model",
						"provider": "openai",
						"name": "gpt-4",
						"architecture": {
							"type": "moe",
							"parameters": {
								"total": 220.000007,
								"active": "not-a-number-or-a-range",
							}
						},
						"warnings": null,
						"sources": [
							"https://example.com"
						]
					}
				]
			}`,
			expectError: true,
		},
		{
			name: "returns unexpected type error when range min value is unsupported",
			jsonContent: `{
				"models": [
					{
						"type": "model",
						"provider": "openai",
						"name": "gpt-4",
						"architecture": {
							"type": "moe",
							"parameters": {
								"total": 220.000007,
								"active": {
									"min": "not-a-float",
									"max": 880.534
								},
							}
						},
						"warnings": null,
						"sources": [
							"https://example.com"
						]
					}
				]
			}`,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tempFile, err := os.CreateTemp("", "aimodels_test_*.json")
			assert.NoError(t, err)
			defer os.Remove(tempFile.Name())

			if tt.jsonContent != "" {
				_, err = tempFile.WriteString(tt.jsonContent)
				assert.NoError(t, err)
			}

			modelsData, err := FetchAIModels(tempFile.Name())
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, modelsData)
				assert.Len(t, modelsData.Models, tt.expectedCount)

				if tt.expectedModel != nil {
					model := modelsData.Models[0]
					assert.Equal(t, tt.expectedModel.Name(), model.Name())
					assert.Equal(t, tt.expectedModel.Provider(), model.Provider())
					assert.Equal(t, tt.expectedModel.Architecture().Type, model.Architecture().Type)
					assert.Equal(
						t, tt.expectedModel.Architecture().Parameters.Total, model.Architecture().Parameters.Total,
					)
					assert.Equal(
						t, tt.expectedModel.Architecture().Parameters.Active,
						model.Architecture().Parameters.Active,
					)
				}
			}
		})
	}
}
