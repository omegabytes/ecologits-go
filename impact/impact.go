/*
Package impact provides utilities for calculating the environmental and energy impact of using generative AI models.

The following parameters are used throughout the package:

- modelActiveParameterCount: Number of active parameters of the model (in billions).
- modelTotalParameterCount: Number of parameters of the model (in billions).
- outputTokenCount: Number of generated tokens.
- requestLatency: Measured request latency in seconds.
- ifElectricityMixADPe: ADPe impact factor of electricity consumption in kgSbeq / kWh (Antimony).
- ifElectricityMixPE: PE impact factor of electricity consumption in MJ / kWh.
- ifElectricityMixGWP: GWP impact factor of electricity consumption in kgCO2eq / kWh.
- modelQuantizationBits: Number of bits used to represent the model weights.
- gpuEnergyAlpha: Alpha parameter of the GPU linear power consumption profile.
- gpuEnergyBeta: Beta parameter of the GPU linear power consumption profile.
- gpuEnergyStdev: Standard deviation of the GPU linear power consumption profile.
- gpuLatencyAlpha: Alpha parameter of the GPU linear latency profile.
- gpuLatencyBeta: Beta parameter of the GPU linear latency profile.
- gpuLatencyStdev: Standard deviation of the GPU linear latency profile.
- gpuMemory: Amount of memory available on a single GPU.
- gpuEmbodiedGWP: GWP embodied impact of a single GPU.
- gpuEmbodiedADPe: ADPe embodied impact of a single GPU.
- gpuEmbodiedPE: PE embodied impact of a single GPU.
- serverGpuCount: Number of available GPUs in the server.
- serverPower: Power consumption of the server in kW.
- serverEmbodiedGWP: GWP embodied impact of the server in kgCO2eq.
- serverEmbodiedADPe: ADPe embodied impact of the server in kgSbeq.
- serverEmbodiedPE: PE embodied impact of the server in MJ.
- serverLifetime: Lifetime duration of the server in seconds.
- datacenterPUE: Power Usage Effectiveness (PUE) of the datacenter.
*/
package impact

import "math"

const (
	MODEL_QUANTIZATION_BITS     = 4
	GPU_ENERGY_ALPHA            = 8.91e-8
	GPU_ENERGY_BETA             = 1.43e-6
	GPU_ENERGY_STDEV            = 5.19e-7
	GPU_LATENCY_ALPHA           = 8.02e-4
	GPU_LATENCY_BETA            = 2.23e-2
	GPU_LATENCY_STDEV           = 7.00e-6
	GPU_MEMORY                  = 80 // GB
	GPU_EMBODIED_IMPACT_GWP     = 143
	GPU_EMBODIED_IMPACT_ADPE    = 5.1e-3
	GPU_EMBODIED_IMPACT_PE      = 1828
	SERVER_GPUS                 = 8
	SERVER_POWER                = 1 // kW
	SERVER_EMBODIED_IMPACT_GWP  = 3000
	SERVER_EMBODIED_IMPACT_ADPE = 0.24
	SERVER_EMBODIED_IMPACT_PE   = 38000
	HARDWARE_LIFESPAN           = 5 * 365 * 24 * 60 * 60
	DATACENTER_PUE              = 1.2
)

type RangeValue struct {
	Min float64
	Max float64
}

//type Energy struct {
//	Value RangeValue
//}

// Energy represents energy measured in kWh.
//type Energy float64

//type GWP struct {
//	Value RangeValue
//}
//
//type ADPe struct {
//	Value RangeValue
//}
//
//type PE struct {
//	Value RangeValue
//}

//type Usage struct {
//	Energy Energy
//	GWP    GWP
//	ADPe   ADPe
//	PE     PE
//}
//
//type Embodied struct {
//	GWP  GWP
//	ADPe ADPe
//	PE   PE
//}

type Usage struct {
	Energy RangeValue
	GWP    RangeValue
	ADPe   RangeValue
	PE     RangeValue
}

type Embodied struct {
	GWP  RangeValue
	ADPe RangeValue
	PE   RangeValue
}

type Total struct {
	GWP  RangeValue
	ADPe RangeValue
	PE   RangeValue
}

type Impacts struct {
	RequestUsage  Usage
	EmbodiedUsage Usage
	TotalUsage    Usage
}

type LLMModel interface{}

type AIModel struct {
	TotalParameterCount  float64
	ActiveParameterCount float64
}

func GetModel(modelName string) AIModel {
	// Example: mistralai open-mixtral-8x22b
	return AIModel{
		TotalParameterCount:  140.6,
		ActiveParameterCount: 39.1,
	}
}

type ElectricityMix struct {
	ADPe float64
	GWP  float64
	PE   float64
}

func GetElectricityMix(geo string) ElectricityMix {
	// Example: USA
	return ElectricityMix{
		ADPe: 0.0000000985548,
		GWP:  0.67978,
		PE:   11.358,
	}
}

// ComputeImpacts computes the environmental and energy impact of the generative AI model.
func ComputeImpacts(
	activeParamCount RangeValue,
	totalParamCount RangeValue,
	outputTokenCount float64,
	requestLatency float64,
	electricityMixADPe *float64,
	electricityMixPE *float64,
	electricityMixGWP *float64,
) Impacts {
	modelRequiredMemory := ModelRequiredMemory(totalParamCount.Max, MODEL_QUANTIZATION_BITS)
	generationLatency := GenerationLatency(activeParamCount.Max, outputTokenCount, GPU_LATENCY_ALPHA, GPU_LATENCY_BETA, GPU_LATENCY_STDEV, requestLatency)
	gpuRequiredCount := GpuRequiredCount(modelRequiredMemory, GPU_MEMORY)
	serverEnergy := ServerEnergy(generationLatency.Max, SERVER_POWER, SERVER_GPUS, gpuRequiredCount)
	gpuEnergy := GpuEnergy(activeParamCount.Max, outputTokenCount, GPU_ENERGY_ALPHA, GPU_ENERGY_BETA, GPU_ENERGY_STDEV)
	requestEnergy := RequestEnergy(DATACENTER_PUE, serverEnergy, gpuRequiredCount, gpuEnergy)
	requestADPe := RequestUsageADPe(requestEnergy, *electricityMixADPe)
	requestGWP := RequestUsageGWP(requestEnergy, *electricityMixGWP)
	requestPE := RequestUsagePE(requestEnergy, *electricityMixPE)

	serverGpuEmbodiedADPe := ServerGpuEmbodiedADPe(SERVER_EMBODIED_IMPACT_ADPE, float64(SERVER_GPUS), GPU_EMBODIED_IMPACT_ADPE, gpuRequiredCount)
	serverGpuEmbodiedGWP := ServerGpuEmbodiedGWP(SERVER_EMBODIED_IMPACT_GWP, float64(SERVER_GPUS), GPU_EMBODIED_IMPACT_GWP, gpuRequiredCount)
	serverGpuEmbodiedPE := ServerGpuEmbodiedPE(SERVER_EMBODIED_IMPACT_PE, float64(SERVER_GPUS), GPU_EMBODIED_IMPACT_PE, gpuRequiredCount)

	embodiedADPe := RequestEmbodiedADPe(serverGpuEmbodiedADPe, HARDWARE_LIFESPAN, generationLatency)
	embodiedGWP := RequestEmbodiedGWP(serverGpuEmbodiedGWP, HARDWARE_LIFESPAN, generationLatency)
	embodiedPE := RequestEmbodiedPE(serverGpuEmbodiedPE, HARDWARE_LIFESPAN, generationLatency)

	totalADPe := RangeValue{
		Min: requestADPe.Min + embodiedADPe.Min,
		Max: requestADPe.Max + embodiedADPe.Max,
	}
	totalGWP := RangeValue{
		Min: requestGWP.Min + embodiedGWP.Min,
		Max: requestGWP.Max + embodiedGWP.Max,
	}
	totalPE := RangeValue{
		Min: requestPE.Min + embodiedPE.Min,
		Max: requestPE.Max + embodiedPE.Max,
	}

	return Impacts{
		RequestUsage: Usage{
			Energy: requestEnergy,
			GWP:    requestGWP,
			ADPe:   requestADPe,
			PE:     requestPE,
		},
		EmbodiedUsage: Usage{
			GWP:  embodiedGWP,
			ADPe: embodiedADPe,
			PE:   embodiedPE,
		},
		TotalUsage: Usage{
			GWP:  totalGWP,
			ADPe: totalADPe,
			PE:   totalPE,
		},
	}
}

// GpuEnergy computes the energy consumption of a single GPU.
//
// Args:
//
//	modelActiveParameterCount: Number of active parameters of the model (in billion).
//	outputTokenCount: Number of generated tokens.
//	gpuEnergyAlpha: Alpha parameter of the GPU linear power consumption profile.
//	gpuEnergyBeta: Beta parameter of the GPU linear power consumption profile.
//	gpuEnergyStdev: Standard deviation of the GPU linear power consumption profile.
//
// Returns:
//
//	The 95% confidence interval of energy consumption of a single GPU in kWh.
func GpuEnergy(
	modelActiveParameterCount float64,
	outputTokenCount float64,
	gpuEnergyAlpha float64,
	gpuEnergyBeta float64,
	gpuEnergyStdev float64,
) RangeValue {
	gpuEnergyPerTokenMean := gpuEnergyAlpha*modelActiveParameterCount + gpuEnergyBeta
	gpuEnergyMin := outputTokenCount * (gpuEnergyPerTokenMean - 1.96*gpuEnergyStdev)
	gpuEnergyMax := outputTokenCount * (gpuEnergyPerTokenMean + 1.96*gpuEnergyStdev)
	return RangeValue{
		Min: math.Max(0, gpuEnergyMin),
		Max: gpuEnergyMax,
	}
}

// GenerationLatency computes the token generation latency in seconds.
//
// Args:
//
//	modelActiveParameterCount: Number of active parameters of the model (in billion).
//	outputTokenCount: Number of generated tokens.
//	gpuLatencyAlpha: Alpha parameter of the GPU linear latency profile.
//	gpuLatencyBeta: Beta parameter of the GPU linear latency profile.
//	gpuLatencyStdev: Standard deviation of the GPU linear latency profile.
//	requestLatency: Measured request latency (upper bound) in seconds.
//
// Returns:
//
//	The token generation latency in seconds.
func GenerationLatency(
	modelActiveParameterCount float64,
	outputTokenCount float64,
	gpuLatencyAlpha float64,
	gpuLatencyBeta float64,
	gpuLatencyStdev float64,
	requestLatency float64,
) RangeValue {
	gpuLatencyPerTokenMean := gpuLatencyAlpha*modelActiveParameterCount + gpuLatencyBeta
	gpuLatencyMin := outputTokenCount * (gpuLatencyPerTokenMean - 1.96*gpuLatencyStdev)
	gpuLatencyMax := outputTokenCount * (gpuLatencyPerTokenMean + 1.96*gpuLatencyStdev)
	gpuLatencyInterval := RangeValue{
		Min: math.Max(0, gpuLatencyMin),
		Max: gpuLatencyMax,
	}
	if gpuLatencyInterval.Max < requestLatency {
		return gpuLatencyInterval
	}
	return RangeValue{
		Min: requestLatency,
		Max: requestLatency,
	}
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
func ModelRequiredMemory(modelTotalParameterCount float64, modelQuantizationBits int) float64 {
	return 1.2 * modelTotalParameterCount * float64(modelQuantizationBits) / 8
}

// GpuRequiredCount computes the number of GPUs required to store the model.
//
// Args:
//
//	modelRequiredMemory: Required memory to load the model on GPU.
//	gpuMemory: Amount of memory available on a single GPU.
//
// Returns:
//
//	The number of required GPUs to load the model.
func GpuRequiredCount(modelRequiredMemory float64, gpuMemory float64) int {
	return int(math.Ceil(modelRequiredMemory / gpuMemory))
}

// ServerEnergy computes the energy consumption of the server.
//
// Args:
//
//	generationLatency: Token generation latency in seconds.
//	serverPower: Power consumption of the server in kW.
//	serverGpuCount: Number of available GPUs in the server.
//	gpuRequiredCount: Number of required GPUs to load the model.
//
// Returns:
//
//	The energy consumption of the server (GPUs are not included) in kWh.
func ServerEnergy(
	generationLatency float64,
	serverPower float64,
	serverGpuCount int,
	gpuRequiredCount int,
) float64 {
	return (generationLatency / 3600) * serverPower * (float64(gpuRequiredCount) / float64(serverGpuCount))
}

// RequestEnergy computes the energy consumption of the request.
//
// Args:
//
//	datacenterPUE: PUE of the datacenter.
//	serverEnergy: Energy consumption of the server in kWh.
//	gpuRequiredCount: Number of required GPUs to load the model.
//	gpuEnergy: Energy consumption of a single GPU in kWh.
//
// Returns:
//
//	The energy consumption of the request in kWh.
func RequestEnergy(datacenterPUE, serverEnergy float64, gpuRequiredCount int, gpuEnergy RangeValue) RangeValue {
	return RangeValue{
		Min: datacenterPUE * (serverEnergy + float64(gpuRequiredCount)*gpuEnergy.Min),
		Max: datacenterPUE * (serverEnergy + float64(gpuRequiredCount)*gpuEnergy.Max),
	}
}

// RequestUsageGWP computes the Global Warming Potential (GWP) usage impact of the request.
//
// Args:
//
//	requestEnergy: Energy consumption of the request in kWh.
//	ifElectricityMixGWP: GWP impact factor of electricity consumption in kgCO2eq / kWh.
//
// Returns:
//
//	The GWP usage impact of the request in kgCO2eq.
func RequestUsageGWP(requestEnergy RangeValue, ifElectricityMixGWP float64) RangeValue {
	return RangeValue{
		Min: requestEnergy.Min * ifElectricityMixGWP,
		Max: requestEnergy.Max * ifElectricityMixGWP,
	}
}

// RequestUsageADPe computes the Abiotic Depletion Potential for Elements (ADPe) usage impact of the request.
//
// Args:
//
//	requestEnergy: Energy consumption of the request in kWh.
//	ifElectricityMixADPe: ADPe impact factor of electricity consumption in kgSbeq / kWh.
//
// Returns:
//
//	The ADPe usage impact of the request in kgSbeq.
func RequestUsageADPe(requestEnergy RangeValue, ifElectricityMixADPe float64) RangeValue {
	return RangeValue{
		Min: requestEnergy.Min * ifElectricityMixADPe,
		Max: requestEnergy.Max * ifElectricityMixADPe,
	}
}

// RequestUsagePE computes the Primary Energy (PE) usage impact of the request.
//
// Args:
//
//	requestEnergy: Energy consumption of the request in kWh.
//	ifElectricityMixPE: PE impact factor of electricity consumption in MJ / kWh.
//
// Returns:
//
//	The PE usage impact of the request in MJ.
func RequestUsagePE(requestEnergy RangeValue, ifElectricityMixPE float64) RangeValue {
	return RangeValue{
		Min: requestEnergy.Min * ifElectricityMixPE,
		Max: requestEnergy.Max * ifElectricityMixPE,
	}
}

// ServerGpuEmbodiedGWP computes the Global Warming Potential (GWP) embodied impact of the server.
//
// Args:
//
//	serverEmbodiedGwp: GWP embodied impact of the server in kgCO2eq.
//	serverGpuCount: Number of available GPUs in the server.
//	gpuEmbodiedGwp: GWP embodied impact of a single GPU in kgCO2eq.
//	gpuRequiredCount: Number of required GPUs to load the model.
//
// Returns:
//
//	The GWP embodied impact of the server and the GPUs in kgCO2eq.
func ServerGpuEmbodiedGWP(
	serverEmbodiedGwp float64,
	serverGpuCount float64,
	gpuEmbodiedGwp float64,
	gpuRequiredCount int,
) float64 {
	return (float64(gpuRequiredCount)/serverGpuCount)*serverEmbodiedGwp + float64(gpuRequiredCount)*gpuEmbodiedGwp
}

// ServerGpuEmbodiedADPe computes the Abiotic Depletion Potential for Elements (ADPe) embodied impact of the server.
//
// Args:
//
//	serverEmbodiedADPe: ADPe embodied impact of the server in kgSbeq.
//	serverGpuCount: Number of available GPUs in the server.
//	gpuEmbodiedADPe: ADPe embodied impact of a single GPU in kgSbeq.
//	gpuRequiredCount: Number of required GPUs to load the model.
//
// Returns:
//
//	The ADPe embodied impact of the server and the GPUs in kgSbeq.
func ServerGpuEmbodiedADPe(
	serverEmbodiedADPe float64,
	serverGpuCount float64,
	gpuEmbodiedADPe float64,
	gpuRequiredCount int,
) float64 {
	return (float64(gpuRequiredCount)/serverGpuCount)*serverEmbodiedADPe + float64(gpuRequiredCount)*gpuEmbodiedADPe
}

// ServerGpuEmbodiedPE computes the Primary Energy (PE) embodied impact of the server.
//
// Args:
//
//	serverEmbodiedPE: PE embodied impact of the server in MJ.
//	serverGpuCount: Number of available GPUs in the server.
//	gpuEmbodiedPE: PE embodied impact of a single GPU in MJ.
//	gpuRequiredCount: Number of required GPUs to load the model.
//
// Returns:
//
//	The PE embodied impact of the server and the GPUs in MJ.
func ServerGpuEmbodiedPE(
	serverEmbodiedPE float64,
	serverGpuCount float64,
	gpuEmbodiedPE float64,
	gpuRequiredCount int,
) float64 {
	return (float64(gpuRequiredCount)/serverGpuCount)*serverEmbodiedPE + float64(gpuRequiredCount)*gpuEmbodiedPE
}

// RequestEmbodiedGWP computes the Global Warming Potential (GWP) embodied impact of the request.
//
// Args:
//
//	serverGpuEmbodiedGwp: GWP embodied impact of the server and the GPUs in kgCO2eq.
//	serverLifetime: Lifetime duration of the server in seconds.
//	generationLatency: Token generation latency in seconds.
//
// Returns:
//
//	The GWP embodied impact of the request in kgCO2eq.
func RequestEmbodiedGWP(serverGpuEmbodiedGwp float64, serverLifetime float64, generationLatency RangeValue) RangeValue {
	return RangeValue{
		Min: (generationLatency.Min / serverLifetime) * serverGpuEmbodiedGwp,
		Max: (generationLatency.Max / serverLifetime) * serverGpuEmbodiedGwp,
	}
}

// RequestEmbodiedADPe computes the Abiotic Depletion Potential for Elements (ADPe) embodied impact of the request.
//
// Args:
//
//	serverGpuEmbodiedADPe: ADPe embodied impact of the server and the GPUs in kgSbeq.
//	serverLifetime: Lifetime duration of the server in seconds.
//	generationLatency: Token generation latency in seconds.
//
// Returns:
//
//	The ADPe embodied impact of the request in kgSbeq.
func RequestEmbodiedADPe(serverGpuEmbodiedADPe float64, serverLifetime float64, generationLatency RangeValue) RangeValue {
	return RangeValue{
		Min: (generationLatency.Min / serverLifetime) * serverGpuEmbodiedADPe,
		Max: (generationLatency.Max / serverLifetime) * serverGpuEmbodiedADPe,
	}
}

// RequestEmbodiedPE computes the Primary Energy (PE) embodied impact of the request.
//
// Args:
//
//	serverGpuEmbodiedPE: PE embodied impact of the server and the GPUs in MJ.
//	serverLifetime: Lifetime duration of the server in seconds.
//	generationLatency: Token generation latency in seconds.
//
// Returns:
//
//	The PE embodied impact of the request in MJ.
func RequestEmbodiedPE(serverGpuEmbodiedPE float64, serverLifetime float64, generationLatency RangeValue) RangeValue {
	return RangeValue{
		Min: (generationLatency.Min / serverLifetime) * serverGpuEmbodiedPE,
		Max: (generationLatency.Max / serverLifetime) * serverGpuEmbodiedPE,
	}
}
