/*
Package impact provides utilities for calculating the environmental and energy impact of using generative AI models.
*/
package impact

import (
	"fmt"

	"github.com/omegabytes/ecologits-go/aimodel"
	"github.com/omegabytes/ecologits-go/common"
	"github.com/omegabytes/ecologits-go/request"
	"github.com/omegabytes/ecologits-go/server"
)

type Usage struct {
	Energy common.RangeValue
	GWP    common.RangeValue
	ADPe   common.RangeValue
	PE     common.RangeValue
}

type Embodied struct {
	GWP  common.RangeValue
	ADPe common.RangeValue
	PE   common.RangeValue
}

type Total struct {
	GWP  common.RangeValue
	ADPe common.RangeValue
	PE   common.RangeValue
}

type ImpactIface interface {
	CalculateRequestUsage(requestEnergy common.RangeValue, electricityMix float64)
	CalculateRequestEmbodied(hardwareLifespan float64, generationLatency common.RangeValue)
	CalculateServerGpuEmbodied(server *server.ServerInfra, gpuRequiredCount int)
	CalculateTotal()
}

type Impacts struct {
	Energy common.RangeValue
	ADPe   ADPe
	GWP    GWP
	PE     PE
}

// ComputeImpacts computes the environmental and energy impact of the generative AI model.
func ComputeImpacts(aiModel *aimodel.AIModel, server *server.ServerInfra, req request.Request) (Impacts, error) {
	modelRequiredMemory := aiModel.ModelRequiredMemory()
	electricityMix := req.GetElectricityMix()

	gpuRequiredCount, err := server.GpuRequiredCount(modelRequiredMemory)
	if err != nil {
		return Impacts{}, fmt.Errorf("failed to get GPU required count: %w", err)
	}

	paramsActiveMax := aiModel.Architecture().Parameters.Active.Max
	generationLatency, err := server.GenerationLatency(paramsActiveMax, req.OutputTokenCount, req.Latency)
	if err != nil {
		return Impacts{}, fmt.Errorf("failed to get generation latency: %w", err)
	}

	gpuEnergy, err := server.GpuEnergy(paramsActiveMax, req.OutputTokenCount)
	if err != nil {
		return Impacts{}, fmt.Errorf("failed to get GPU energy: %w", err)
	}

	serverEnergy, err := server.ServerEnergyBaseline(generationLatency.Max, gpuRequiredCount)
	if err != nil {
		return Impacts{}, fmt.Errorf("failed to get server energy: %w", err)
	}

	requestEnergy, err := server.RequestEnergy(serverEnergy, gpuRequiredCount, gpuEnergy)
	if err != nil {
		return Impacts{}, fmt.Errorf("failed to get request energy: %w", err)
	}

	adpeImpact := &ADPe{}
	adpeImpact.CalculateRequestUsage(requestEnergy, electricityMix.ADPe)
	adpeImpact.CalculateServerGpuEmbodied(server, gpuRequiredCount)
	adpeImpact.CalculateRequestEmbodied(float64(server.HardwareLifespan), generationLatency)
	adpeImpact.CalculateTotal()

	gwpImpact := &GWP{}
	gwpImpact.CalculateRequestUsage(requestEnergy, electricityMix.GWP)
	gwpImpact.CalculateServerGpuEmbodied(server, gpuRequiredCount)
	gwpImpact.CalculateRequestEmbodied(float64(server.HardwareLifespan), generationLatency)
	gwpImpact.CalculateTotal()

	peImpact := &PE{}
	peImpact.CalculateRequestUsage(requestEnergy, electricityMix.PE)
	peImpact.CalculateServerGpuEmbodied(server, gpuRequiredCount)
	peImpact.CalculateRequestEmbodied(float64(server.HardwareLifespan), generationLatency)
	peImpact.CalculateTotal()

	return Impacts{
		Energy: requestEnergy,
		ADPe:   *adpeImpact,
		GWP:    *gwpImpact,
		PE:     *peImpact,
	}, nil
}

func requestUsage(requestEnergy common.RangeValue, electricityMix float64) common.RangeValue {
	return common.RangeValue{
		Min: requestEnergy.Min * electricityMix,
		Max: requestEnergy.Max * electricityMix,
	}
}

func requestEmbodied(
	serverGpuEmbodiedImpact float64,
	hardwareLifespan float64,
	generationLatency common.RangeValue,
) common.RangeValue {
	return common.RangeValue{
		Min: (generationLatency.Min / hardwareLifespan) * serverGpuEmbodiedImpact,
		Max: (generationLatency.Max / hardwareLifespan) * serverGpuEmbodiedImpact,
	}
}

func serverGpuEmbodied(
	serverEmbodiedImpact float64,
	gpuCount float64,
	gpuEmbodiedImpact float64,
	gpuRequiredCount int,
) float64 {
	return (float64(gpuRequiredCount)/gpuCount)*serverEmbodiedImpact + float64(gpuRequiredCount)*gpuEmbodiedImpact
}

func totalImpact(requestImpact, embodiedImpact common.RangeValue) common.RangeValue {
	return common.RangeValue{
		Min: requestImpact.Min + embodiedImpact.Min,
		Max: requestImpact.Max + embodiedImpact.Max,
	}
}