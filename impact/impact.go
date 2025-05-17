/*
Package impact provides utilities for calculating the environmental and energy impact of using generative AI models.
*/
package impact

import (
	"fmt"

	"github.com/omegabytes/ecologits-go/aimodel"
	"github.com/omegabytes/ecologits-go/common"
	"github.com/omegabytes/ecologits-go/gpuserver"
	"github.com/omegabytes/ecologits-go/request"
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
	CalculateServerGPUEmbodied(server *gpuserver.GPUServer, gpuRequiredCount int)
	CalculateTotal()
}

type Impacts struct {
	Energy common.RangeValue
	ADPe   ADPe
	GWP    GWP
	PE     PE
}

// ComputeImpacts computes the environmental and energy impact of the generative AI model.
func ComputeImpacts(aiModel *aimodel.AIModel, server *gpuserver.GPUServer, req request.Request) (Impacts, error) {
	modelRequiredMemory := aiModel.ModelRequiredMemory()
	electricityMix := req.GetElectricityMix()

	gpuRequiredCount, err := server.GPURequiredCount(modelRequiredMemory)
	if err != nil {
		return Impacts{}, fmt.Errorf("failed to get GPU required count: %w", err)
	}

	paramsActiveMax := aiModel.Architecture().Parameters.Active.Max
	generationLatency, err := server.GenerationLatency(paramsActiveMax, req.OutputTokenCount, req.Latency)
	if err != nil {
		return Impacts{}, fmt.Errorf("failed to get generation latency: %w", err)
	}

	gpuEnergyKWH, err := server.GPUEnergyKWH(paramsActiveMax, req.OutputTokenCount)
	if err != nil {
		return Impacts{}, fmt.Errorf("failed to get GPU energy: %w", err)
	}

	serverEnergyKWH, err := server.ServerEnergyBaseline(generationLatency.Max, gpuRequiredCount)
	if err != nil {
		return Impacts{}, fmt.Errorf("failed to get server energy: %w", err)
	}

	requestEnergy, err := server.RequestEnergy(serverEnergyKWH, gpuRequiredCount, gpuEnergyKWH)
	if err != nil {
		return Impacts{}, fmt.Errorf("failed to get request energy: %w", err)
	}

	adpeImpact := &ADPe{}
	adpeImpact.CalculateRequestUsage(requestEnergy, electricityMix.ADPe)
	adpeImpact.CalculateServerGPUEmbodied(server, gpuRequiredCount)
	adpeImpact.CalculateRequestEmbodied(float64(server.HardwareLifespan), generationLatency)
	adpeImpact.CalculateTotal()

	gwpImpact := &GWP{}
	gwpImpact.CalculateRequestUsage(requestEnergy, electricityMix.GWP)
	gwpImpact.CalculateServerGPUEmbodied(server, gpuRequiredCount)
	gwpImpact.CalculateRequestEmbodied(float64(server.HardwareLifespan), generationLatency)
	gwpImpact.CalculateTotal()

	peImpact := &PE{}
	peImpact.CalculateRequestUsage(requestEnergy, electricityMix.PE)
	peImpact.CalculateServerGPUEmbodied(server, gpuRequiredCount)
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
	serverGPUEmbodiedImpact float64,
	hardwareLifespan float64,
	generationLatency common.RangeValue,
) common.RangeValue {
	return common.RangeValue{
		Min: (generationLatency.Min / hardwareLifespan) * serverGPUEmbodiedImpact,
		Max: (generationLatency.Max / hardwareLifespan) * serverGPUEmbodiedImpact,
	}
}

func serverGPUEmbodied(
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
