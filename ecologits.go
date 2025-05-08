package ecologits_go

import (
	"github.com/omegabytes/ecologits-go/aimodel"
	"github.com/omegabytes/ecologits-go/impact"
	"github.com/omegabytes/ecologits-go/request"
	"github.com/omegabytes/ecologits-go/server"
)

func NewLLM(modelName, provider string) (*aimodel.AIModel, error) {
	return aimodel.NewAIModel(modelName, provider)
}

func NewServer() (*server.ServerInfra, error) {
	return server.GenericServerInfra()
}

func NewRequest(outputTokenCount int64, latency float64, geo string) (request.Request, error) {
	return request.Request{OutputTokenCount: float64(outputTokenCount), Latency: latency, Geo: geo}, nil
}

func ComputeImpacts(
	aiModel *aimodel.AIModel,
	request request.Request,
	server *server.ServerInfra,
) (impact.Impacts, error) {
	return impact.ComputeImpacts(aiModel, server, request)
}