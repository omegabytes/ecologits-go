package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"time"

	ecogo "github.com/omegabytes/ecologits-go"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/responses"
)

func main() {
	client := openai.NewClient()
	ctx := context.Background()

	question := "Write me a haiku about computers"

	start := time.Now()
	resp, err := client.Responses.New(
		ctx, responses.ResponseNewParams{
			Input: responses.ResponseNewParamsInputUnion{OfString: openai.String(question)},
			Model: openai.ChatModelGPT4,
		},
	)
	if err != nil {
		slog.Error("client request failed", "error", err)
		os.Exit(1)
	}
	reqLatencyMs := time.Since(start).Milliseconds()

	req, err := ecogo.NewRequest(resp.Usage.OutputTokens, float64(reqLatencyMs), "USA")
	if err != nil {
		slog.Error("failed to create new request model", "error", err)
		return
	}

	llm, err := ecogo.NewLLM(openai.ChatModelGPT4, "openai")
	if err != nil {
		slog.Error("failed to create new llm model", "error", err)
		return
	}
	slog.Info("successfully created llm", "llm", llm)

	server, err := ecogo.NewGPUServer()
	if err != nil {
		slog.Error("failed to create new server model", "error", err)
		return
	}

	impacts, err := ecogo.ComputeImpacts(llm, req, server)
	if err != nil {
		slog.Error("failed to compute impacts", "error", err)
		return
	}

	jsonData, err := json.MarshalIndent(impacts, "", "  ")
	if err != nil {
		slog.Error("failed to marshal json", "error", err)
		return
	}
	slog.Info("impacts", "req latency", reqLatencyMs)
	fmt.Println("impacts: ", string(jsonData))
}