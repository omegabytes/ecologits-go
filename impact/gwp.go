package impact

import (
	"github.com/omegabytes/ecologits-go/common"
	"github.com/omegabytes/ecologits-go/gpuserver"
)

var _ ImpactIface = &GWP{}

// GWP represents Global Warming Potential (GWP) impact.
type GWP struct {
	EmbodiedImpact          common.RangeValue
	RequestImpact           common.RangeValue
	ServerGPUEmbodiedImpact float64
	TotalImpact             common.RangeValue
}

// CalculateRequestUsage computes the Global Warming Potential (GWP) usage impact of the request in kgCO2eq.
// The elecImpactFactor is electricity consumption in kgCO2eq / kWh.
func (g *GWP) CalculateRequestUsage(requestEnergyKWH common.RangeValue, elecImpactFactor float64) {
	g.RequestImpact = requestUsage(requestEnergyKWH, elecImpactFactor)
}

// CalculateRequestEmbodied computes the GWP embodied impact of the request in kgCO2eq.
func (g *GWP) CalculateRequestEmbodied(serverLifespanSecs float64, tokenGenLatSecs common.RangeValue) {
	g.EmbodiedImpact = requestEmbodied(g.ServerGPUEmbodiedImpact, serverLifespanSecs, tokenGenLatSecs)
}

// CalculateServerGPUEmbodied computes the GWP embodied impact of the server in kgCO2eq.
func (g *GWP) CalculateServerGPUEmbodied(server *gpuserver.GPUServer, gpuRequiredCount int) {
	g.ServerGPUEmbodiedImpact = serverGPUEmbodied(
		server.EmbodiedImpactGWP, float64(server.AvailableGPUCount), server.GPUModel.EmbodiedImpactGWP,
		gpuRequiredCount)
}

// CalculateTotal computes the total GWP impact in kgCO2eq.
func (g *GWP) CalculateTotal() {
	g.TotalImpact = totalImpact(g.RequestImpact, g.EmbodiedImpact)
}
