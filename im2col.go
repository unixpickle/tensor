package tensor

// Im2ColDims stores info about instances of Im2Col32 and
// Im2Col64.
type Im2ColDims struct {
	FilterWidth  int
	FilterHeight int
	FilterStride int

	ImageWidth  int
	ImageHeight int
	ImageDepth  int
}

// An Im2Col32 converts images (in the form of tensors) to
// and from matrices which can be used to perform
// convolution as matrix multiplication.
type Im2Col32 interface {
	Dims() Im2ColDims

	// ToMatrix creates a matrix from the input image.
	// The matrix is stored in row-major order.
	// Each row in the matrix will be of length
	// filterWidth * filterHeight * imageDepth.
	ToMatrix(img *Float32) []float32

	// ToImage creates an image from a matrix.
	// Entries in the matrix which are mapped to the same
	// entry in the image are added together.
	ToImage(mat []float32) *Float32
}

// An Im2Col64 converts images (in the form of tensors) to
// and from matrices which can be used to perform
// convolution as matrix multiplication.
type Im2Col64 interface {
	Dims() Im2ColDims

	// ToMatrix creates a matrix from the input image.
	// The matrix is stored in row-major order.
	// Each row in the matrix will be of length
	// filterWidth * filterHeight * imageDepth.
	ToMatrix(img *Float64) []float64

	// ToImage creates an image from a matrix.
	// Entries in the matrix which are mapped to the same
	// entry in the image are added together.
	ToImage(mat []float64) *Float64
}

func (d defaultImplementation) NewIm2Col32(dims Im2ColDims) Im2Col32 {
	// TODO: this.
	return nil
}

func (d defaultImplementation) NewIm2Col64(dims Im2ColDims) Im2Col64 {
	// TODO: this.
	return nil
}
