package tensor

import (
	"math/rand"
	"testing"
)

func TestAddVec32(t *testing.T) {
	v1 := make([]float32, 15)
	v2 := make([]float32, 15)
	expected := make([]float32, 15)
	for i := range v1 {
		v1[i] = float32(rand.NormFloat64())
		v2[i] = float32(rand.NormFloat64())
		expected[i] = v1[i] + v2[i]
	}
	addVec32(v1, v2)
	if !vectorsSimilar32(v1, expected) {
		t.Errorf("expected %v but got %v", expected, v1)
	}
}

func TestAddVec64(t *testing.T) {
	v1 := make([]float64, 15)
	v2 := make([]float64, 15)
	expected := make([]float64, 15)
	for i := range v1 {
		v1[i] = rand.NormFloat64()
		v2[i] = rand.NormFloat64()
		expected[i] = v1[i] + v2[i]
	}
	addVec64(v1, v2)
	if !vectorsSimilar64(v1, expected) {
		t.Errorf("expected %v but got %v", expected, v1)
	}
}
