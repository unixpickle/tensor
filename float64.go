package tensor

// Float64 is a 64-bit 3-dimensional tensor.
type Float64 struct {
	Width  int
	Height int
	Depth  int

	// Data is stored in the following order of priority:
	// rows, columns, depth.
	// Thus, the value at row=1, col=2, depth=3 is right
	// before the value at row=1, col=2, depth=4.
	Data []float64
}

// NewFloat64 creates a new, zero'd out Float64.
func NewFloat64(width, height, depth int) *Float64 {
	return &Float64{
		Width:  width,
		Height: height,
		Depth:  depth,
		Data:   make([]float64, width*height*depth),
	}
}

// Index returns the index in the Data for the given
// coordinate, where x is the column, y is the row, and
// z is the depth.
func (f *Float64) Index(x, y, z int) int {
	return (x+y*f.Width)*f.Depth + z
}

// Get returns the entry at the given coordinate.
// See Index for details on the coordinate system.
func (f *Float64) Get(x, y, z int) float64 {
	return f.Data[(x+y*f.Width)*f.Depth+z]
}

// Set sets the entry at the given coordinate.
// See Index for details on the coordinate system.
func (f *Float64) Set(x, y, z int, value float64) {
	f.Data[(x+y*f.Width)*f.Depth+z] = value
}
