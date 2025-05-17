package ecologits_go

import (
	"github.com/omegabytes/ecologits-go/aimodel"
	"github.com/omegabytes/ecologits-go/gpuserver"
	"github.com/omegabytes/ecologits-go/impact"
	"github.com/omegabytes/ecologits-go/request"
)

type RangeValue struct {
	Min float64
	Max float64
}

func NewLLM(modelName string) (*aimodel.AIModel, error) {
	return aimodel.NewAIModel(modelName)
}

func NewGPUServer() (*gpuserver.GPUServer, error) {
	return gpuserver.GenericGPUServer()
}

func NewRequest(outputTokenCount int64, latency float64, geo string) (request.Request, error) {
	return request.Request{OutputTokenCount: float64(outputTokenCount), Latency: latency, Geo: geo}, nil
}

func ComputeImpacts(
	aiModel *aimodel.AIModel,
	request request.Request,
	server *gpuserver.GPUServer,
) (impact.Impacts, error) {
	return impact.ComputeImpacts(aiModel, server, request)
}
