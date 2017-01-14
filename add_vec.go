package tensor

// addVec32 adds v2 to v1.
func addVec32(v1, v2 []float32) {
	alignLen := len(v1) - (len(v1) % 4)
	if alignLen > 0 {
		addVecFast32(v1[:alignLen], v2[:alignLen])
	}
	for i, x := range v2[alignLen:] {
		v1[i+alignLen] += x
	}
}

// addVec64 adds v2 to v1.
func addVec64(v1, v2 []float64) {
	alignLen := len(v1) - (len(v1) % 2)
	if alignLen > 0 {
		addVecFast64(v1[:alignLen], v2[:alignLen])
	}
	for i, x := range v2[alignLen:] {
		v1[i+alignLen] += x
	}
}
