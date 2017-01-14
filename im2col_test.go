package tensor

import (
	"math"
	"testing"
)

func TestIm2Col32(t *testing.T) {
	t.Run("ToMatrix", func(t *testing.T) {
		img := NewFloat32(3, 4, 2)
		copy(img.Data, []float32{
			0.6278542, 0.2407223, 0.7186654, 0.4162109, 0.8660594, 0.9047409,
			0.7351845, 0.5342780, 0.6007710, 0.6002036, 0.2789797, 0.7606597,
			0.0159000, 0.8052780, 0.4503867, 0.2263453, 0.6980444, 0.7397514,
			0.1506396, 0.9062053, 0.0118610, 0.0193349, 0.2847805, 0.0949364,
		})
		dims := &Im2ColDims{
			ImageWidth:   img.Width,
			ImageHeight:  img.Height,
			ImageDepth:   img.Depth,
			FilterWidth:  2,
			FilterHeight: 2,
			FilterStride: 1,
		}
		i2c := defaultImplementation{}.NewIm2Col32(dims)
		out := i2c.ToMatrix(img)
		expected := []float32{
			// Top left
			0.6278542, 0.2407223, 0.7186654, 0.4162109,
			0.7351845, 0.5342780, 0.6007710, 0.6002036,
			// Top right
			0.7186654, 0.4162109, 0.8660594, 0.9047409,
			0.6007710, 0.6002036, 0.2789797, 0.7606597,
			// Middle left
			0.7351845, 0.5342780, 0.6007710, 0.6002036,
			0.0159000, 0.8052780, 0.4503867, 0.2263453,
			// Middle right
			0.6007710, 0.6002036, 0.2789797, 0.7606597,
			0.4503867, 0.2263453, 0.6980444, 0.7397514,
			// Bottom left
			0.0159000, 0.8052780, 0.4503867, 0.2263453,
			0.1506396, 0.9062053, 0.0118610, 0.0193349,
			// Bottom right
			0.4503867, 0.2263453, 0.6980444, 0.7397514,
			0.0118610, 0.0193349, 0.2847805, 0.0949364,
		}
		if !vectorsSimilar32(out, expected) {
			t.Errorf("got %v expected %v", out, expected)
		}

		dims.FilterHeight = 3
		i2c = defaultImplementation{}.NewIm2Col32(dims)
		out = i2c.ToMatrix(img)
		expected = []float32{
			// Top left
			0.6278542, 0.2407223, 0.7186654, 0.4162109,
			0.7351845, 0.5342780, 0.6007710, 0.6002036,
			0.0159000, 0.8052780, 0.4503867, 0.2263453,
			// Top right
			0.7186654, 0.4162109, 0.8660594, 0.9047409,
			0.6007710, 0.6002036, 0.2789797, 0.7606597,
			0.4503867, 0.2263453, 0.6980444, 0.7397514,
			// Bottom left
			0.7351845, 0.5342780, 0.6007710, 0.6002036,
			0.0159000, 0.8052780, 0.4503867, 0.2263453,
			0.1506396, 0.9062053, 0.0118610, 0.0193349,
			// Bottom right
			0.6007710, 0.6002036, 0.2789797, 0.7606597,
			0.4503867, 0.2263453, 0.6980444, 0.7397514,
			0.0118610, 0.0193349, 0.2847805, 0.0949364,
		}
		if !vectorsSimilar32(out, expected) {
			t.Errorf("got %v expected %v", out, expected)
		}

		dims.FilterHeight = 2
		dims.FilterStride = 2
		i2c = defaultImplementation{}.NewIm2Col32(dims)
		out = i2c.ToMatrix(img)
		expected = []float32{
			// Top left
			0.6278542, 0.2407223, 0.7186654, 0.4162109,
			0.7351845, 0.5342780, 0.6007710, 0.6002036,
			// Bottom left
			0.0159000, 0.8052780, 0.4503867, 0.2263453,
			0.1506396, 0.9062053, 0.0118610, 0.0193349,
		}
		if !vectorsSimilar32(out, expected) {
			t.Errorf("got %v expected %v", out, expected)
		}

		dims.FilterHeight = 1
		dims.FilterWidth = 1
		i2c = defaultImplementation{}.NewIm2Col32(dims)
		out = i2c.ToMatrix(img)
		expected = []float32{
			0.6278542, 0.2407223, 0.8660594, 0.9047409,
			0.0159000, 0.8052780, 0.6980444, 0.7397514,
		}
		if !vectorsSimilar32(out, expected) {
			t.Errorf("got %v expected %v", out, expected)
		}
	})
}

func vectorsSimilar32(v1, v2 []float32) bool {
	if len(v1) != len(v2) {
		return false
	}
	for i, x := range v1 {
		if math.Abs(float64(x-v2[i])) > 1e-5 {
			return false
		}
	}
	return true
}
