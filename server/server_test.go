package server

import (
	"fmt"
	"testing"

	"github.com/omegabytes/ecologits-go/common"
	"github.com/stretchr/testify/assert"
)

func TestGenericGPU(t *testing.T) {
	want := GPU{
		EnergyAlpha:        8.91e-8,
		EnergyBeta:         1.43e-6,
		EnergyStdev:        5.19e-7,
		LatencyAlpha:       8.02e-4,
		LatencyBeta:        2.23e-2,
		LatencyStdev:       7.00e-6,
		AvailMemoryGB:      80,
		EmbodiedImpactADPe: 5.1e-3,
		EmbodiedImpactGWP:  143,
		EmbodiedImpactPE:   1828,
	}

	t.Run("should return default GPU values", func(t *testing.T) {
		assert.Equal(t, want, GenericGPU())
	})
}

func TestGenericServerInfra(t *testing.T) {
	t.Run("should return default ServerInfra values", func(t *testing.T) {
		want := &ServerInfra{
			AvailableGpuCount:  100,
			PowerConsumptionKW: 1,
			EmbodiedImpactADPe: 0.24,
			EmbodiedImpactGWP:  3000,
			EmbodiedImpactPE:   38000,
			HardwareLifespan:   5 * 365 * 24 * 60 * 60,
			DatacenterPue:      1.2,
			Gpu: GPU{
				EnergyAlpha:        8.91e-8,
				EnergyBeta:         1.43e-6,
				EnergyStdev:        5.19e-7,
				LatencyAlpha:       8.02e-4,
				LatencyBeta:        2.23e-2,
				LatencyStdev:       7.00e-6,
				AvailMemoryGB:      80,
				EmbodiedImpactADPe: 5.1e-3,
				EmbodiedImpactGWP:  143,
				EmbodiedImpactPE:   1828,
			},
		}
		got, err := GenericServerInfra()
		assert.NoError(t, err)
		assert.Equal(t, got, want)
	})
}

func TestServerInfra_GenerationLatency(t *testing.T) {
	type fields struct {
		AvailableGpuCount  int
		PowerConsumptionKW float64
		EmbodiedImpactADPe float64
		EmbodiedImpactGWP  float64
		EmbodiedImpactPE   float64
		HardwareLifespan   int64
		Gpu                GPU
	}
	type args struct {
		modelActiveParamCount float64
		outputTokenCount      float64
		requestLatencySecs    float64
	}
	tests := []struct {
		name          string
		fields        fields
		args          args
		want          common.RangeValue
		expectedError error
	}{
		{
			// gpuLatPerTokenMean: 8.02e-4 * 10 + 2.23e-2 = 0.03032
			// gpuLatMin: 100 * (0.03032 - 1.96 * 7.00e-6) = 3.030628
			// gpuLatMax: 100 * (0.03032 + 1.96 * 7.00e-6) = 3.033372
			name: "should calculate generation latency successfully",
			fields: fields{
				AvailableGpuCount:  4,
				PowerConsumptionKW: 1.5,
				Gpu: GPU{
					LatencyAlpha: 8.02e-4,
					LatencyBeta:  2.23e-2,
					LatencyStdev: 7.00e-6,
				},
			},
			args: args{
				modelActiveParamCount: 10,
				outputTokenCount:      100,
				requestLatencySecs:    5,
			},
			want: common.RangeValue{
				Min: 3.030628,
				Max: 3.033372,
			},
			expectedError: nil,
		},
		{
			// gpuLatPerTokenMean: 8.02e-4 * 10 + 2.23e-2 = 0.03032
			// gpuLatMin: 100 * (0.03032 - 1.96 * 7.00e-6) = 3.030628
			// gpuLatMax: 100 * (0.03032 + 1.96 * 7.00e-6) = 3.033372
			name: "should return requestLatencySecs when calculated gpuLatencyInterval is < requestLatencySecs",
			fields: fields{
				AvailableGpuCount:  4,
				PowerConsumptionKW: 1.5,
				Gpu: GPU{
					LatencyAlpha: 8.02e-4,
					LatencyBeta:  2.23e-2,
					LatencyStdev: 7.00e-6,
				},
			},
			args: args{
				modelActiveParamCount: 10,
				outputTokenCount:      100,
				requestLatencySecs:    1,
			},
			want: common.RangeValue{
				Min: 1,
				Max: 1,
			},
			expectedError: nil,
		},
		{
			name: "should return error when modelActiveParamCount is 0",
			fields: fields{
				Gpu: GPU{
					LatencyAlpha: 8.02e-4,
					LatencyBeta:  2.23e-2,
					LatencyStdev: 7.00e-6,
				},
			},
			args: args{
				modelActiveParamCount: 0,
				outputTokenCount:      100,
				requestLatencySecs:    1,
			},
			want:          common.RangeValue{},
			expectedError: fmt.Errorf("modelActiveParamCount must be greater than 0"),
		},
		{
			name: "should return error when outputTokenCount is 0",
			fields: fields{
				Gpu: GPU{
					LatencyAlpha: 8.02e-4,
					LatencyBeta:  2.23e-2,
					LatencyStdev: 7.00e-6,
				},
			},
			args: args{
				modelActiveParamCount: 10,
				outputTokenCount:      0,
				requestLatencySecs:    1,
			},
			want:          common.RangeValue{},
			expectedError: fmt.Errorf("outputTokenCount must be greater than 0"),
		},
		{
			name: "should return error when requestLatencySecs is 0",
			fields: fields{
				Gpu: GPU{
					LatencyAlpha: 8.02e-4,
					LatencyBeta:  2.23e-2,
					LatencyStdev: 7.00e-6,
				},
			},
			args: args{
				modelActiveParamCount: 10,
				outputTokenCount:      100,
				requestLatencySecs:    0,
			},
			want:          common.RangeValue{},
			expectedError: fmt.Errorf("requestLatencySecs must be greater than 0"),
		},
		{
			name: "should return error when GPU latency parameters are invalid",
			fields: fields{
				Gpu: GPU{
					LatencyAlpha: 0,
					LatencyBeta:  0,
					LatencyStdev: 0,
				},
			},
			args: args{
				modelActiveParamCount: 10,
				outputTokenCount:      100,
				requestLatencySecs:    1,
			},
			want:          common.RangeValue{},
			expectedError: fmt.Errorf("GPU latency parameters must be greater than 0"),
		},
		{
			name: "should return error when AvailableGpuCount is 0",
			fields: fields{
				AvailableGpuCount: 0,
				Gpu: GPU{
					LatencyAlpha: 8.02e-4,
					LatencyBeta:  2.23e-2,
					LatencyStdev: 7.00e-6,
				},
			},
			args: args{
				modelActiveParamCount: 10,
				outputTokenCount:      100,
				requestLatencySecs:    1,
			},
			want:          common.RangeValue{},
			expectedError: fmt.Errorf("AvailableGpuCount must be greater than 0"),
		},
		{
			name: "should return error when PowerConsumptionKW is 0",
			fields: fields{
				AvailableGpuCount:  4,
				PowerConsumptionKW: 0,
				Gpu: GPU{
					LatencyAlpha: 8.02e-4,
					LatencyBeta:  2.23e-2,
					LatencyStdev: 7.00e-6,
				},
			},
			args: args{
				modelActiveParamCount: 10,
				outputTokenCount:      100,
				requestLatencySecs:    1,
			},
			want:          common.RangeValue{},
			expectedError: fmt.Errorf("PowerConsumptionKW must be greater than 0"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &ServerInfra{
				AvailableGpuCount:  tt.fields.AvailableGpuCount,
				PowerConsumptionKW: tt.fields.PowerConsumptionKW,
				HardwareLifespan:   tt.fields.HardwareLifespan,
				Gpu:                tt.fields.Gpu,
			}
			got, err := s.GenerationLatency(tt.args.modelActiveParamCount, tt.args.outputTokenCount,
				tt.args.requestLatencySecs)
			if tt.expectedError != nil {
				assert.EqualError(t, err, tt.expectedError.Error())
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestServerInfra_GpuEnergy(t *testing.T) {
	type fields struct {
		Gpu GPU
	}
	type args struct {
		modelActiveParamCount float64
		outputTokenCount      float64
	}
	tests := []struct {
		name          string
		fields        fields
		args          args
		want          common.RangeValue
		expectedError error
	}{
		{
			// perTokenMean: 8.91e-8 * 10 + 1.43e-6 = 0.000002321
			// gpuEnergyMin: 100 * (0.000002321 - 1.96 * 5.19e-7) = 0.00013037599999999997
			// gpuEnergyMax: 100 * (0.000002321 + 1.96 * 5.19e-7) = 0.000333824
			name: "should calculate GPU energy successfully",
			fields: fields{
				Gpu: GPU{
					EnergyAlpha: 8.91e-8,
					EnergyBeta:  1.43e-6,
					EnergyStdev: 5.19e-7,
				},
			},
			args: args{
				modelActiveParamCount: 10,
				outputTokenCount:      100,
			},
			want: common.RangeValue{
				Min: 0.00013037599999999997,
				Max: 0.000333824,
			},
			expectedError: nil,
		},
		{
			// perTokenMean: 8.91e-8 * 1 + 1.43e-8 = 0.0000001034
			// gpuEnergyMin: 100 * (0.0000001034 - 1.96 * 5.19e-7) = -0.000091384
			// gpuEnergyMax: 100 * (0.0000001034 + 1.96 * 5.19e-7) = 0.00011206399999999999
			name: "should set min GPU to 0 when min calculation result is < 0",
			fields: fields{
				Gpu: GPU{
					EnergyAlpha: 8.91e-8,
					EnergyBeta:  1.43e-8,
					EnergyStdev: 5.19e-7,
				},
			},
			args: args{
				modelActiveParamCount: 1,
				outputTokenCount:      100,
			},
			want: common.RangeValue{
				Min: 0,
				Max: 0.00011206399999999999,
			},
			expectedError: nil,
		},
		{
			name: "should return error when modelActiveParamCount is 0",
			fields: fields{
				Gpu: GPU{
					EnergyAlpha: 8.91e-8,
					EnergyBeta:  1.43e-6,
					EnergyStdev: 5.19e-7,
				},
			},
			args: args{
				modelActiveParamCount: 0,
				outputTokenCount:      100,
			},
			want:          common.RangeValue{},
			expectedError: fmt.Errorf("modelActiveParamCount must be greater than 0"),
		},
		{
			name: "should return error when outputTokenCount is 0",
			fields: fields{
				Gpu: GPU{
					EnergyAlpha: 8.91e-8,
					EnergyBeta:  1.43e-6,
					EnergyStdev: 5.19e-7,
				},
			},
			args: args{
				modelActiveParamCount: 10,
				outputTokenCount:      0,
			},
			want:          common.RangeValue{},
			expectedError: fmt.Errorf("outputTokenCount must be greater than 0"),
		},
		{
			name: "should return error when GPU energy parameters are invalid",
			fields: fields{
				Gpu: GPU{
					EnergyAlpha: 0,
					EnergyBeta:  0,
					EnergyStdev: 0,
				},
			},
			args: args{
				modelActiveParamCount: 10,
				outputTokenCount:      100,
			},
			want:          common.RangeValue{},
			expectedError: fmt.Errorf("GPU energy parameters must be greater than 0"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &ServerInfra{
				Gpu: tt.fields.Gpu,
			}
			got, err := s.GpuEnergyKWH(tt.args.modelActiveParamCount, tt.args.outputTokenCount)
			if tt.expectedError != nil {
				assert.EqualError(t, err, tt.expectedError.Error())
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestServerInfra_GpuRequiredCount(t *testing.T) {
	type fields struct {
		AvailableGpuCount int
		Gpu               GPU
	}
	type args struct {
		modelRequiredMemory float64
	}
	tests := []struct {
		name          string
		fields        fields
		args          args
		want          int
		expectedError error
	}{
		{
			name: "should return number of GPUs required to load model when model memory and available GPU memory are provided",
			fields: fields{
				AvailableGpuCount: 4,
				Gpu: GPU{
					AvailMemoryGB: 80,
				},
			},
			args: args{
				modelRequiredMemory: 80,
			},
			want:          1,
			expectedError: nil,
		},
		{
			name: "should return error when model memory is 0",
			fields: fields{
				AvailableGpuCount: 4,
				Gpu: GPU{
					AvailMemoryGB: 80,
				},
			},
			args: args{
				modelRequiredMemory: 0,
			},
			want:          0,
			expectedError: fmt.Errorf("model required memory must be greater than 0"),
		},
		{
			name: "should return error when server gpu available memory is 0",
			fields: fields{
				AvailableGpuCount: 4,
				Gpu: GPU{
					AvailMemoryGB: 0,
				},
			},
			args: args{
				modelRequiredMemory: 80,
			},
			want:          0,
			expectedError: fmt.Errorf("available GPU count must be greater than 0"),
		},
		{
			name: "should round up the returned gpu count",
			fields: fields{
				AvailableGpuCount: 4,
				Gpu: GPU{
					AvailMemoryGB: 30,
				},
			},
			args: args{
				modelRequiredMemory: 80,
			},
			want:          3,
			expectedError: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &ServerInfra{
				AvailableGpuCount: tt.fields.AvailableGpuCount,
				Gpu:               tt.fields.Gpu,
			}
			got, err := s.GpuRequiredCount(tt.args.modelRequiredMemory)
			if tt.expectedError != nil {
				assert.EqualError(t, err, tt.expectedError.Error())
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestServerInfra_ServerEnergyBaseline(t *testing.T) {
	type fields struct {
		AvailableGpuCount  int
		PowerConsumptionKW float64
		Gpu                GPU
	}
	type args struct {
		tokenGenLatencySeconds float64
		gpuRequiredCount       int
	}
	tests := []struct {
		name          string
		fields        fields
		args          args
		want          float64
		expectedError error
	}{
		{
			name: "should calculate energy baseline successfully",
			fields: fields{
				AvailableGpuCount:  4,
				PowerConsumptionKW: 1.5,
			},
			args: args{
				tokenGenLatencySeconds: 10,
				gpuRequiredCount:       2,
			},
			want:          0.0020833333333333333,
			expectedError: nil,
		},
		{
			name: "should return error when tokenGenLatencySeconds is 0",
			fields: fields{
				AvailableGpuCount:  4,
				PowerConsumptionKW: 1.5,
			},
			args: args{
				tokenGenLatencySeconds: 0,
				gpuRequiredCount:       2,
			},
			want:          0,
			expectedError: fmt.Errorf("token generation latency must be greater than 0"),
		},
		{
			name: "should return error when gpuRequiredCount is 0",
			fields: fields{
				AvailableGpuCount:  4,
				PowerConsumptionKW: 1.5,
			},
			args: args{
				tokenGenLatencySeconds: 10,
				gpuRequiredCount:       0,
			},
			want:          0,
			expectedError: fmt.Errorf("gpuRequiredCount must be between 1 and the number of available GPUs"),
		},
		{
			name: "should return error when gpuRequiredCount exceeds available GPUs",
			fields: fields{
				AvailableGpuCount:  4,
				PowerConsumptionKW: 1.5,
			},
			args: args{
				tokenGenLatencySeconds: 10,
				gpuRequiredCount:       5,
			},
			want:          0,
			expectedError: fmt.Errorf("gpuRequiredCount must be between 1 and the number of available GPUs"),
		},
		{
			name: "should return error when PowerConsumptionKW is 0",
			fields: fields{
				AvailableGpuCount:  4,
				PowerConsumptionKW: 0,
			},
			args: args{
				tokenGenLatencySeconds: 10,
				gpuRequiredCount:       2,
			},
			want:          0,
			expectedError: fmt.Errorf("power consumption must be greater than 0"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &ServerInfra{
				AvailableGpuCount:  tt.fields.AvailableGpuCount,
				PowerConsumptionKW: tt.fields.PowerConsumptionKW,
				Gpu:                tt.fields.Gpu,
			}
			got, err := s.ServerEnergyBaseline(tt.args.tokenGenLatencySeconds, tt.args.gpuRequiredCount)
			if tt.expectedError != nil {
				assert.EqualError(t, err, tt.expectedError.Error())
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestServerInfra_RequestEnergy(t *testing.T) {
	type fields struct {
		AvailableGpuCount  int
		PowerConsumptionKW float64
		EmbodiedImpactADPe float64
		EmbodiedImpactGWP  float64
		EmbodiedImpactPE   float64
		HardwareLifespan   int64
		Gpu                GPU
		DatacenterPue      float64
	}
	type args struct {
		serverEnergyKWH  float64
		gpuRequiredCount int
		gpuEnergyKWH     common.RangeValue
	}
	tests := []struct {
		name          string
		fields        fields
		args          args
		want          common.RangeValue
		expectedError error
	}{
		{
			// min: 1.2 * (1.5 + 2 * 0.1) = 2.04
			// max: 1.2 * (1.5 + 2 * 0.2) = 2.28
			name: "should calculate request energy successfully",
			fields: fields{
				AvailableGpuCount: 4,
				DatacenterPue:     1.2,
			},
			args: args{
				serverEnergyKWH:  1.5,
				gpuRequiredCount: 2,
				gpuEnergyKWH:     common.RangeValue{Min: 0.1, Max: 0.2},
			},
			want: common.RangeValue{
				Min: 2.04,
				Max: 2.28,
			},
			expectedError: nil,
		},
		{
			name: "should return error when serverEnergyKWH is 0",
			fields: fields{
				DatacenterPue: 1.2,
			},
			args: args{
				serverEnergyKWH:  0,
				gpuRequiredCount: 2,
				gpuEnergyKWH:     common.RangeValue{Min: 0.1, Max: 0.2},
			},
			want:          common.RangeValue{},
			expectedError: fmt.Errorf("serverEnergyKWH must be greater than 0"),
		},
		{
			name: "should return error when gpuRequiredCount is 0",
			fields: fields{
				AvailableGpuCount: 4,
				DatacenterPue:     1.2,
			},
			args: args{
				serverEnergyKWH:  1.5,
				gpuRequiredCount: 0,
				gpuEnergyKWH:     common.RangeValue{Min: 0.1, Max: 0.2},
			},
			want:          common.RangeValue{},
			expectedError: fmt.Errorf("gpuRequiredCount must be between 1 and the number of available GPUs"),
		},
		{
			name: "should return error when gpuRequiredCount exceeds available GPUs",
			fields: fields{
				AvailableGpuCount: 4,
				DatacenterPue:     1.2,
			},
			args: args{
				serverEnergyKWH:  1.5,
				gpuRequiredCount: 5,
				gpuEnergyKWH:     common.RangeValue{Min: 0.1, Max: 0.2},
			},
			want:          common.RangeValue{},
			expectedError: fmt.Errorf("gpuRequiredCount must be between 1 and the number of available GPUs"),
		},
		{
			name: "should return error when gpuEnergyKWH.Min is negative",
			fields: fields{
				AvailableGpuCount: 4,
				DatacenterPue:     1.2,
			},
			args: args{
				serverEnergyKWH:  1.5,
				gpuRequiredCount: 2,
				gpuEnergyKWH:     common.RangeValue{Min: -0.1, Max: 0.2},
			},
			want:          common.RangeValue{},
			expectedError: fmt.Errorf("gpuEnergyKWH values must be non-negative"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &ServerInfra{
				AvailableGpuCount:  tt.fields.AvailableGpuCount,
				PowerConsumptionKW: tt.fields.PowerConsumptionKW,
				EmbodiedImpactADPe: tt.fields.EmbodiedImpactADPe,
				EmbodiedImpactGWP:  tt.fields.EmbodiedImpactGWP,
				EmbodiedImpactPE:   tt.fields.EmbodiedImpactPE,
				HardwareLifespan:   tt.fields.HardwareLifespan,
				Gpu:                tt.fields.Gpu,
				DatacenterPue:      tt.fields.DatacenterPue,
			}
			got, err := s.RequestEnergy(tt.args.serverEnergyKWH, tt.args.gpuRequiredCount, tt.args.gpuEnergyKWH)
			if tt.expectedError != nil {
				assert.EqualError(t, err, tt.expectedError.Error())
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}