package request

type Request struct {
	OutputTokenCount float64
	Latency          float64
	Geo              string
}

type ElectricityMix struct {
	ADPe float64
	GWP  float64
	PE   float64
}

func (r *Request) GetElectricityMix() ElectricityMix {
	// todo: implement a function to get the electricity mix based on the geo parameter
	// Example: USA
	return ElectricityMix{
		ADPe: 0.0000000985548,
		GWP:  0.67978,
		PE:   11.358,
	}
}
