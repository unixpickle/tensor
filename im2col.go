package tensor

import "fmt"

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

// MatrixSize returns the number of components in the
// matirx that would be produced by running Im2Col.
func (i *Im2ColDims) MatrixSize() int {
	w := 1 + (i.ImageWidth-i.FilterWidth)/i.FilterStride
	h := 1 + (i.ImageHeight-i.FilterHeight)/i.FilterStride
	if w <= 0 || h <= 0 {
		return 0
	}
	return w * h * i.FilterWidth * i.FilterHeight * i.ImageDepth
}

func (i *Im2ColDims) verifyDims(w, h, d int) {
	if i.ImageWidth != w || i.ImageHeight != h || i.ImageDepth != d {
		panic(fmt.Sprintf("expected %dx%dx%d but got %dx%dx%d", i.ImageWidth,
			i.ImageHeight, i.ImageDepth, w, h, d))
	}
}

// An Im2Col32 converts images (in the form of tensors) to
// and from matrices which can be used to perform
// convolution as matrix multiplication.
type Im2Col32 interface {
	Dims() *Im2ColDims

	// ToMatrix creates a matrix from the input image.
	// The matrix is stored in row-major order.
	// Each row in the matrix will be of length
	// filterWidth * filterHeight * imageDepth.
	ToMatrix(out []float32, img *Float32)

	// ToImage creates an image from a matrix.
	// Entries in the matrix which are mapped to the same
	// entry in the image are added together.
	ToImage(mat []float32) *Float32
}

// An Im2Col64 converts images (in the form of tensors) to
// and from matrices which can be used to perform
// convolution as matrix multiplication.
type Im2Col64 interface {
	Dims() *Im2ColDims

	// ToMatrix creates a matrix from the input image.
	// The matrix is stored in row-major order.
	// Each row in the matrix will be of length
	// filterWidth * filterHeight * imageDepth.
	ToMatrix(out []float64, img *Float64)

	// ToImage creates an image from a matrix.
	// Entries in the matrix which are mapped to the same
	// entry in the image are added together.
	ToImage(mat []float64) *Float64
}

func (d defaultImplementation) NewIm2Col32(dims *Im2ColDims) Im2Col32 {
	return &im2Col32{
		mapping: im2ColMapping(dims),
		dims:    dims,
	}
}

func (d defaultImplementation) NewIm2Col64(dims *Im2ColDims) Im2Col64 {
	return &im2Col64{
		mapping: im2ColMapping(dims),
		dims:    dims,
	}
}

func im2ColMapping(dims *Im2ColDims) []int {
	w := 1 + (dims.ImageWidth-dims.FilterWidth)/dims.FilterStride
	h := 1 + (dims.ImageHeight-dims.FilterHeight)/dims.FilterStride
	if w <= 0 || h <= 0 {
		return nil
	}

	dest := make([]int, 0, w*h*dims.FilterHeight)

	t := Float32{
		Width:  dims.ImageWidth,
		Height: dims.ImageHeight,
		Depth:  dims.ImageDepth,
	}

	for y := 0; y < h*dims.FilterStride; y += dims.FilterStride {
		for x := 0; x < w*dims.FilterStride; x += dims.FilterStride {
			for subY := 0; subY < dims.FilterHeight; subY++ {
				dest = append(dest, t.Index(x, y+subY, 0))
			}
		}
	}

	return dest
}

type im2Col32 struct {
	mapping []int
	dims    *Im2ColDims
}

func (i *im2Col32) Dims() *Im2ColDims {
	return i.dims
}

func (i *im2Col32) ToMatrix(out []float32, img *Float32) {
	i.dims.verifyDims(img.Width, img.Height, img.Depth)
	rowSize := i.dims.FilterWidth * i.dims.ImageDepth
	if len(out) != len(i.mapping)*rowSize {
		panic("incorrect output matrix size")
	}
	destIdx := 0
	for _, x := range i.mapping {
		copy(out[destIdx:], img.Data[x:x+rowSize])
		destIdx += rowSize
	}
}

func (i *im2Col32) ToImage(mat []float32) *Float32 {
	if len(mat) != i.dims.MatrixSize() {
		panic(fmt.Sprintf("expected matrix size %d but got %d", len(i.mapping),
			len(mat)))
	}
	rowSize := i.dims.FilterWidth * i.dims.ImageDepth
	res := NewFloat32(i.dims.ImageWidth, i.dims.ImageHeight, i.dims.ImageDepth)
	sourceIdx := 0
	for _, x := range i.mapping {
		addVec32(res.Data[x:x+rowSize], mat[sourceIdx:sourceIdx+rowSize])
		sourceIdx += rowSize
	}
	return res
}

type im2Col64 struct {
	mapping []int
	dims    *Im2ColDims
}

func (i *im2Col64) Dims() *Im2ColDims {
	return i.dims
}

func (i *im2Col64) ToMatrix(out []float64, img *Float64) {
	i.dims.verifyDims(img.Width, img.Height, img.Depth)
	rowSize := i.dims.FilterWidth * i.dims.ImageDepth
	if len(out) != len(i.mapping)*rowSize {
		panic("incorrect output matrix size")
	}
	destIdx := 0
	for _, x := range i.mapping {
		copy(out[destIdx:], img.Data[x:x+rowSize])
		destIdx += rowSize
	}
}

func (i *im2Col64) ToImage(mat []float64) *Float64 {
	if len(mat) != i.dims.MatrixSize() {
		panic(fmt.Sprintf("expected matrix size %d but got %d", len(i.mapping),
			len(mat)))
	}
	rowSize := i.dims.FilterWidth * i.dims.ImageDepth
	res := NewFloat64(i.dims.ImageWidth, i.dims.ImageHeight, i.dims.ImageDepth)
	sourceIdx := 0
	for _, x := range i.mapping {
		addVec64(res.Data[x:x+rowSize], mat[sourceIdx:sourceIdx+rowSize])
		sourceIdx += rowSize
	}
	return res
}
