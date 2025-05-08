package impact

import (
	"github.com/omegabytes/ecologits-go/common"
	"github.com/omegabytes/ecologits-go/server"
)

var _ ImpactIface = &PE{}

// PE represents Primary Energy (PE) impact.
type PE struct {
	EmbodiedImpact          common.RangeValue
	RequestImpact           common.RangeValue
	ServerGpuEmbodiedImpact float64
	TotalImpact             common.RangeValue
}

// CalculateRequestUsage computes the PE usage impact of the request in MJ.
// The elecImpactFactor is electricity consumption in MJ / kWh.
func (p *PE) CalculateRequestUsage(requestEnergyKWH common.RangeValue, elecImpactFactor float64) {
	p.RequestImpact = requestUsage(requestEnergyKWH, elecImpactFactor)
}

// CalculateRequestEmbodied computes the PE embodied impact of the request in MJ.
func (p *PE) CalculateRequestEmbodied(serverLifespanSecs float64, tokenGenLatSecs common.RangeValue) {
	p.EmbodiedImpact = requestEmbodied(p.ServerGpuEmbodiedImpact, serverLifespanSecs, tokenGenLatSecs)
}

// CalculateServerGpuEmbodied computes the PE embodied impact of the server in MJ.
func (p *PE) CalculateServerGpuEmbodied(server *server.ServerInfra, gpuRequiredCount int) {
	p.ServerGpuEmbodiedImpact = serverGpuEmbodied(server.EmbodiedImpactPE, float64(server.AvailableGpuCount),
		server.Gpu.EmbodiedImpactPE, gpuRequiredCount)
}

// CalculateTotal computes the total Primary Energy (PE) impact of the request.
func (p *PE) CalculateTotal() {
	p.TotalImpact = totalImpact(p.RequestImpact, p.EmbodiedImpact)
}