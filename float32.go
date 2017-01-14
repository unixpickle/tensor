package tensor

// Float32 is a 32-bit 3-dimensional tensor.
type Float32 struct {
	Width  int
	Height int
	Depth  int

	// Data is stored in the following order of priority:
	// rows, columns, depth.
	// Thus, the value at row=1, col=2, depth=3 is right
	// before the value at row=1, col=2, depth=4.
	Data []float32
}

// NewFloat32 creates a new, zero'd out Float32.
func NewFloat32(width, height, depth int) *Float32 {
	return &Float32{
		Width:  width,
		Height: height,
		Depth:  depth,
		Data:   make([]float32, width*height*depth),
	}
}

// Index returns the index in the Data for the given
// coordinate, where x is the column, y is the row, and
// z is the depth.
func (f *Float32) Index(x, y, z int) int {
	return (x+y*f.Width)*f.Depth + z
}

// Get returns the entry at the given coordinate.
// See Index for details on the coordinate system.
func (f *Float32) Get(x, y, z int) float32 {
	return f.Data[(x+y*f.Width)*f.Depth+z]
}

// Set sets the entry at the given coordinate.
// See Index for details on the coordinate system.
func (f *Float32) Set(x, y, z int, value float32) {
	f.Data[(x+y*f.Width)*f.Depth+z] = value
}
