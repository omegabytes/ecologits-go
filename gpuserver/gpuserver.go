package gpuserver

import (
	"fmt"
	"math"

	"github.com/omegabytes/ecologits-go/common"
)

// GPUServer represents server overall infrastructure used to train LLMs or execute user requestg.
// Some climate impact is attributed to training or serving requests and some is attributed to server
// construction and operation. The latter is called embodied impact.
type GPUServer struct {
	AvailableGPUCount  int
	PowerConsumptionKW float64
	EmbodiedImpactADPe float64
	EmbodiedImpactGWP  float64
	EmbodiedImpactPE   float64
	HardwareLifespan   int64
	GPUModel           GPU
	DatacenterPUE      float64
}

// GPU represents a GPU contained in a server that is used train LLMs or execute user requestg.
// Some climate impact is attributed to training or serving requests and some is attributed to GPU
// manufacturing, operation, and disposal. The latter is called embodied impact.
type GPU struct {
	EnergyAlpha        float64
	EnergyBeta         float64
	EnergyStdev        float64
	LatencyAlpha       float64
	LatencyBeta        float64
	LatencyStdev       float64
	AvailMemoryGB      float64
	EmbodiedImpactADPe float64
	EmbodiedImpactGWP  float64
	EmbodiedImpactPE   float64
}

// GenericGPUServer returns a gpu server with default values for energy and latency parameterg.
func GenericGPUServer() (*GPUServer, error) {
	const (
		serverGPUCount           = 100
		serverPowerKw            = 1
		serverEmbodiedImpactGWP  = 3000
		serverEmbodiedImpactADPe = 0.24
		serverEmbodiedImpactPE   = 38000
		hardwareLifespan         = 5 * 365 * 24 * 60 * 60
		datacenterPUE            = 1.2
	)

	return &GPUServer{
		AvailableGPUCount:  serverGPUCount,
		PowerConsumptionKW: serverPowerKw,
		EmbodiedImpactADPe: serverEmbodiedImpactADPe,
		EmbodiedImpactGWP:  serverEmbodiedImpactGWP,
		EmbodiedImpactPE:   serverEmbodiedImpactPE,
		HardwareLifespan:   hardwareLifespan,
		GPUModel:           GenericGPU(),
		DatacenterPUE:      datacenterPUE,
	}, nil
}

// GenericGPU returns a GPU with default values for energy and latency parameterg.
func GenericGPU() GPU {
	const (
		gpuEnergyAlpha        = 8.91e-8
		gpuEnergyBeta         = 1.43e-6
		gpuEnergyStdev        = 5.19e-7
		gpuLatencyAlpha       = 8.02e-4
		gpuLatencyBeta        = 2.23e-2
		gpuLatencyStdev       = 7.00e-6
		gpuMemoryGB           = 80
		gpuEmbodiedImpactGWP  = 143
		gpuEmbodiedImpactADPe = 5.1e-3
		gpuEmbodiedImpactPE   = 1828
	)

	return GPU{
		EnergyAlpha:        gpuEnergyAlpha,
		EnergyBeta:         gpuEnergyBeta,
		EnergyStdev:        gpuEnergyStdev,
		LatencyAlpha:       gpuLatencyAlpha,
		LatencyBeta:        gpuLatencyBeta,
		LatencyStdev:       gpuLatencyStdev,
		AvailMemoryGB:      gpuMemoryGB,
		EmbodiedImpactADPe: gpuEmbodiedImpactADPe,
		EmbodiedImpactGWP:  gpuEmbodiedImpactGWP,
		EmbodiedImpactPE:   gpuEmbodiedImpactPE,
	}
}

// GPURequiredCount returns the number of GPUs required to load the model, rounding up.
func (g *GPUServer) GPURequiredCount(modelRequiredMemory float64) (int, error) {
	if modelRequiredMemory <= 0 {
		return 0, fmt.Errorf("model required memory must be greater than 0")
	}
	if g.GPUModel.AvailMemoryGB <= 0 {
		return 0, fmt.Errorf("available GPU count must be greater than 0")
	}
	return int(math.Ceil(modelRequiredMemory / g.GPUModel.AvailMemoryGB)), nil
}

// ServerEnergyBaseline returns the energy consumption of the server in kWh. Does not include GPU power consumption.
func (g *GPUServer) ServerEnergyBaseline(tokenGenLatencySeconds float64, gpuRequiredCount int) (float64, error) {
	if tokenGenLatencySeconds <= 0 {
		return 0, fmt.Errorf("token generation latency must be greater than 0")
	}
	if gpuRequiredCount <= 0 || gpuRequiredCount > g.AvailableGPUCount {
		return 0, fmt.Errorf("gpuRequiredCount must be between 1 and the number of available GPUs")
	}
	if g.PowerConsumptionKW <= 0 {
		return 0, fmt.Errorf("power consumption must be greater than 0")
	}
	return (tokenGenLatencySeconds / 3600) * g.PowerConsumptionKW * (float64(gpuRequiredCount) / float64(g.AvailableGPUCount)), nil
}

// GPUEnergyKWH returns the 95% confidence interval of the energy consumption of a single GPU in kWh.
func (g *GPUServer) GPUEnergyKWH(
	modelActiveParamCount float64,
	outputTokenCount float64,
) (common.RangeValue, error) {
	if modelActiveParamCount <= 0 {
		return common.RangeValue{}, fmt.Errorf("modelActiveParamCount must be greater than 0")
	}
	if outputTokenCount <= 0 {
		return common.RangeValue{}, fmt.Errorf("outputTokenCount must be greater than 0")
	}
	if g.GPUModel.EnergyAlpha <= 0 || g.GPUModel.EnergyBeta <= 0 || g.GPUModel.EnergyStdev <= 0 {
		return common.RangeValue{}, fmt.Errorf("GPU energy parameters must be greater than 0")
	}
	gpuEnergyPerTokenMean := g.GPUModel.EnergyAlpha*modelActiveParamCount + g.GPUModel.EnergyBeta
	gpuEnergyMin := outputTokenCount * (gpuEnergyPerTokenMean - 1.96*g.GPUModel.EnergyStdev)
	gpuEnergyMax := outputTokenCount * (gpuEnergyPerTokenMean + 1.96*g.GPUModel.EnergyStdev)
	return common.RangeValue{
		Min: math.Max(0, gpuEnergyMin),
		Max: gpuEnergyMax,
	}, nil
}

// GenerationLatency returns the token generation latency in secondg.
func (g *GPUServer) GenerationLatency(
	modelActiveParamCount float64,
	outputTokenCount float64,
	requestLatencySecs float64,
) (common.RangeValue, error) {
	if modelActiveParamCount <= 0 {
		return common.RangeValue{}, fmt.Errorf("modelActiveParamCount must be greater than 0")
	}
	if outputTokenCount <= 0 {
		return common.RangeValue{}, fmt.Errorf("outputTokenCount must be greater than 0")
	}
	if requestLatencySecs <= 0 {
		return common.RangeValue{}, fmt.Errorf("requestLatencySecs must be greater than 0")
	}
	if g.GPUModel.LatencyAlpha <= 0 || g.GPUModel.LatencyBeta <= 0 || g.GPUModel.LatencyStdev <= 0 {
		return common.RangeValue{}, fmt.Errorf("GPU latency parameters must be greater than 0")
	}
	if g.AvailableGPUCount <= 0 {
		return common.RangeValue{}, fmt.Errorf("AvailableGPUCount must be greater than 0")
	}
	if g.PowerConsumptionKW <= 0 {
		return common.RangeValue{}, fmt.Errorf("PowerConsumptionKW must be greater than 0")
	}
	gpuLatencyPerTokenMean := g.GPUModel.LatencyAlpha*modelActiveParamCount + g.GPUModel.LatencyBeta
	gpuLatencyMin := outputTokenCount * (gpuLatencyPerTokenMean - 1.96*g.GPUModel.LatencyStdev)
	gpuLatencyMax := outputTokenCount * (gpuLatencyPerTokenMean + 1.96*g.GPUModel.LatencyStdev)
	gpuLatencyInterval := common.RangeValue{
		Min: math.Max(0, gpuLatencyMin),
		Max: gpuLatencyMax,
	}
	if gpuLatencyInterval.Max < requestLatencySecs {
		return gpuLatencyInterval, nil
	}
	return common.RangeValue{
		Min: requestLatencySecs,
		Max: requestLatencySecs,
	}, nil
}

// RequestEnergy returns the energy consumption of the request in kWh.
func (g *GPUServer) RequestEnergy(
	serverEnergyKWH float64,
	gpuRequiredCount int,
	gpuEnergyKWH common.RangeValue,
) (common.RangeValue, error) {
	if serverEnergyKWH <= 0 {
		return common.RangeValue{}, fmt.Errorf("serverEnergyKWH must be greater than 0")
	}
	if gpuRequiredCount <= 0 || gpuRequiredCount > g.AvailableGPUCount {
		return common.RangeValue{}, fmt.Errorf("gpuRequiredCount must be between 1 and the number of available GPUs")
	}
	if gpuEnergyKWH.Min < 0 || gpuEnergyKWH.Max < 0 {
		return common.RangeValue{}, fmt.Errorf("gpuEnergyKWH values must be non-negative")
	}
	return common.RangeValue{
		Min: g.DatacenterPUE * (serverEnergyKWH + float64(gpuRequiredCount)*gpuEnergyKWH.Min),
		Max: g.DatacenterPUE * (serverEnergyKWH + float64(gpuRequiredCount)*gpuEnergyKWH.Max),
	}, nil
}