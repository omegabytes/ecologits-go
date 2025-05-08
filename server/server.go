package server

import (
	"fmt"
	"math"

	"github.com/omegabytes/ecologits-go/common"
)

// ServerInfra represents server overall infrastructure used to train LLMs or execute user requests.
// Some climate impact is attributed to training or serving requests and some is attributed to server
// construction and operation. The latter is called embodied impact.
type ServerInfra struct {
	AvailableGpuCount  int
	PowerConsumptionKW float64
	EmbodiedImpactADPe float64
	EmbodiedImpactGWP  float64
	EmbodiedImpactPE   float64
	HardwareLifespan   int64
	Gpu                GPU
	DatacenterPue      float64
}

// GPU represents a GPU contained in a server that is used train LLMs or execute user requests.
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

// GenericServerInfra returns a server with default values for energy and latency parameters.
func GenericServerInfra() (*ServerInfra, error) {
	const (
		serverGpuCount           = 100
		serverPowerKw            = 1
		serverEmbodiedImpactGwp  = 3000
		serverEmbodiedImpactAdpe = 0.24
		serverEmbodiedImpactPe   = 38000
		hardwareLifespan         = 5 * 365 * 24 * 60 * 60
		datacenterPue            = 1.2
	)

	return &ServerInfra{
		AvailableGpuCount:  serverGpuCount,
		PowerConsumptionKW: serverPowerKw,
		EmbodiedImpactADPe: serverEmbodiedImpactAdpe,
		EmbodiedImpactGWP:  serverEmbodiedImpactGwp,
		EmbodiedImpactPE:   serverEmbodiedImpactPe,
		HardwareLifespan:   hardwareLifespan,
		Gpu:                GenericGPU(),
		DatacenterPue:      datacenterPue,
	}, nil
}

// GenericGPU returns a GPU with default values for energy and latency parameters.
func GenericGPU() GPU {
	const (
		gpuEnergyAlpha        = 8.91e-8
		gpuEnergyBeta         = 1.43e-6
		gpuEnergyStdev        = 5.19e-7
		gpuLatencyAlpha       = 8.02e-4
		gpuLatencyBeta        = 2.23e-2
		gpuLatencyStdev       = 7.00e-6
		gpuMemoryGB           = 80
		gpuEmbodiedImpactGwp  = 143
		gpuEmbodiedImpactAdpe = 5.1e-3
		gpuEmbodiedImpactPe   = 1828
	)

	return GPU{
		EnergyAlpha:        gpuEnergyAlpha,
		EnergyBeta:         gpuEnergyBeta,
		EnergyStdev:        gpuEnergyStdev,
		LatencyAlpha:       gpuLatencyAlpha,
		LatencyBeta:        gpuLatencyBeta,
		LatencyStdev:       gpuLatencyStdev,
		AvailMemoryGB:      gpuMemoryGB,
		EmbodiedImpactADPe: gpuEmbodiedImpactAdpe,
		EmbodiedImpactGWP:  gpuEmbodiedImpactGwp,
		EmbodiedImpactPE:   gpuEmbodiedImpactPe,
	}
}

// GpuRequiredCount returns the number of GPUs required to load the model, rounding up.
func (s *ServerInfra) GpuRequiredCount(modelRequiredMemory float64) (int, error) {
	if modelRequiredMemory <= 0 {
		return 0, fmt.Errorf("model required memory must be greater than 0")
	}
	if s.Gpu.AvailMemoryGB <= 0 {
		return 0, fmt.Errorf("available GPU count must be greater than 0")
	}
	return int(math.Ceil(modelRequiredMemory / s.Gpu.AvailMemoryGB)), nil
}

// ServerEnergyBaseline returns the energy consumption of the server in kWh. Does not include GPU power consumption.
func (s *ServerInfra) ServerEnergyBaseline(tokenGenLatencySeconds float64, gpuRequiredCount int) (float64, error) {
	if tokenGenLatencySeconds <= 0 {
		return 0, fmt.Errorf("token generation latency must be greater than 0")
	}
	if gpuRequiredCount <= 0 || gpuRequiredCount > s.AvailableGpuCount {
		return 0, fmt.Errorf("gpuRequiredCount must be between 1 and the number of available GPUs")
	}
	if s.PowerConsumptionKW <= 0 {
		return 0, fmt.Errorf("power consumption must be greater than 0")
	}
	return (tokenGenLatencySeconds / 3600) * s.PowerConsumptionKW * (float64(gpuRequiredCount) / float64(s.AvailableGpuCount)), nil
}

// GpuEnergyKWH returns the 95% confidence interval of the energy consumption of a single GPU in kWh.
func (s *ServerInfra) GpuEnergyKWH(
	modelActiveParamCount float64,
	outputTokenCount float64,
) (common.RangeValue, error) {
	if modelActiveParamCount <= 0 {
		return common.RangeValue{}, fmt.Errorf("modelActiveParamCount must be greater than 0")
	}
	if outputTokenCount <= 0 {
		return common.RangeValue{}, fmt.Errorf("outputTokenCount must be greater than 0")
	}
	if s.Gpu.EnergyAlpha <= 0 || s.Gpu.EnergyBeta <= 0 || s.Gpu.EnergyStdev <= 0 {
		return common.RangeValue{}, fmt.Errorf("GPU energy parameters must be greater than 0")
	}
	gpuEnergyPerTokenMean := s.Gpu.EnergyAlpha*modelActiveParamCount + s.Gpu.EnergyBeta
	gpuEnergyMin := outputTokenCount * (gpuEnergyPerTokenMean - 1.96*s.Gpu.EnergyStdev)
	gpuEnergyMax := outputTokenCount * (gpuEnergyPerTokenMean + 1.96*s.Gpu.EnergyStdev)
	return common.RangeValue{
		Min: math.Max(0, gpuEnergyMin),
		Max: gpuEnergyMax,
	}, nil
}

// GenerationLatency returns the token generation latency in seconds.
func (s *ServerInfra) GenerationLatency(
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
	if s.Gpu.LatencyAlpha <= 0 || s.Gpu.LatencyBeta <= 0 || s.Gpu.LatencyStdev <= 0 {
		return common.RangeValue{}, fmt.Errorf("GPU latency parameters must be greater than 0")
	}
	if s.AvailableGpuCount <= 0 {
		return common.RangeValue{}, fmt.Errorf("AvailableGpuCount must be greater than 0")
	}
	if s.PowerConsumptionKW <= 0 {
		return common.RangeValue{}, fmt.Errorf("PowerConsumptionKW must be greater than 0")
	}
	gpuLatencyPerTokenMean := s.Gpu.LatencyAlpha*modelActiveParamCount + s.Gpu.LatencyBeta
	gpuLatencyMin := outputTokenCount * (gpuLatencyPerTokenMean - 1.96*s.Gpu.LatencyStdev)
	gpuLatencyMax := outputTokenCount * (gpuLatencyPerTokenMean + 1.96*s.Gpu.LatencyStdev)
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
func (s *ServerInfra) RequestEnergy(
	serverEnergyKWH float64,
	gpuRequiredCount int,
	gpuEnergyKWH common.RangeValue,
) (common.RangeValue, error) {
	if serverEnergyKWH <= 0 {
		return common.RangeValue{}, fmt.Errorf("serverEnergyKWH must be greater than 0")
	}
	if gpuRequiredCount <= 0 || gpuRequiredCount > s.AvailableGpuCount {
		return common.RangeValue{}, fmt.Errorf("gpuRequiredCount must be between 1 and the number of available GPUs")
	}
	if gpuEnergyKWH.Min < 0 || gpuEnergyKWH.Max < 0 {
		return common.RangeValue{}, fmt.Errorf("gpuEnergyKWH values must be non-negative")
	}
	return common.RangeValue{
		Min: s.DatacenterPue * (serverEnergyKWH + float64(gpuRequiredCount)*gpuEnergyKWH.Min),
		Max: s.DatacenterPue * (serverEnergyKWH + float64(gpuRequiredCount)*gpuEnergyKWH.Max),
	}, nil
}