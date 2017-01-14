package tensor

import "sync"

var curImplementationLock sync.RWMutex
var curImplementation Implementation = defaultImplementation{}

type defaultImplementation struct{}

// An Implementation implements routines for the tensor
// library in a modular fashion, allowing different
// implementations to be swapped in at runtime.
type Implementation interface {
	NewIm2Col32(Im2ColDims) Im2Col32
	NewIm2Col64(Im2ColDims) Im2Col64
}

// Use tells the library to use a given implementation.
func Use(impl Implementation) {
	curImplementationLock.Lock()
	curImplementation = impl
	curImplementationLock.Unlock()
}

// CurrentImplementation returns the current
// Implementation.
func CurrentImplementation() Implementation {
	curImplementationLock.RLock()
	res := curImplementation
	curImplementationLock.RUnlock()
	return res
}

// DefaultImplementation returns the default
// Implementation.
func DefaultImplementation() Implementation {
	return defaultImplementation{}
}
