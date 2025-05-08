package impact

import (
	"github.com/omegabytes/ecologits-go/common"
	"github.com/omegabytes/ecologits-go/server"
)

var _ ImpactIface = &ADPe{}

// ADPe represents Abiotic Depletion Potential for Elements (ADPe) impact.
type ADPe struct {
	EmbodiedImpact          common.RangeValue
	RequestImpact           common.RangeValue
	ServerGpuEmbodiedImpact float64
	TotalImpact             common.RangeValue
}

// CalculateRequestUsage computes the ADPe usage impact of the request in kgSbeq.
// The elecImpactFactor is electricity consumption in kgSbeq / kWh.
func (a *ADPe) CalculateRequestUsage(requestEnergyKWH common.RangeValue, elecImpactFactor float64) {
	a.RequestImpact = requestUsage(requestEnergyKWH, elecImpactFactor)
}

// CalculateRequestEmbodied computes the ADPe embodied impact of the request in kgSbeq.
func (a *ADPe) CalculateRequestEmbodied(serverLifespanSecs float64, tokenGenLatSec common.RangeValue) {
	a.EmbodiedImpact = requestEmbodied(a.ServerGpuEmbodiedImpact, serverLifespanSecs, tokenGenLatSec)
}

// CalculateServerGpuEmbodied computes the ADPe embodied impact of the server in kgSbeq.
func (a *ADPe) CalculateServerGpuEmbodied(server *server.ServerInfra, gpuRequiredCount int) {
	a.ServerGpuEmbodiedImpact = serverGpuEmbodied(
		server.EmbodiedImpactADPe, float64(server.AvailableGpuCount), server.Gpu.EmbodiedImpactADPe, gpuRequiredCount)
}

// CalculateTotal computes the total ADPe impact in kgSbeq.
func (a *ADPe) CalculateTotal() {
	a.TotalImpact = totalImpact(a.RequestImpact, a.EmbodiedImpact)
}