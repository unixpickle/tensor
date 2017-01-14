//+build !amd64 noasm appengine

package tensor

// addVecFast32 adds two vectors whose lengths are
// divisible by four greater than zero.
// Result is stored in v1.
func addVecFast32(v1, v2 []float32) {
	if len(v1) != len(v2) || len(v1)%4 != 0 {
		panic("invalid lengths")
	}
	for i, x := range v2 {
		v1[i] += x
	}
}

// addVecFast64 adds two vectors whose lengths are
// divisible by two and greater than zero.
// Result is stored in v1.
func addVecFast64(v1, v2 []float64) {
	if len(v1) != len(v2) || len(v1)%4 != 0 {
		panic("invalid lengths")
	}
	for i, x := range v2 {
		v1[i] += x
	}
}
