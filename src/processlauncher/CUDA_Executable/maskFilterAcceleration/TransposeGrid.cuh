/*
* Copyright 2020 Aiman bin Murhiz, Timotius Pujianto, James Schubach. All rights reserved.
* Bachelor of Computer Science Final project, Semester 1 2020, Monash University, Australia.

* Please contact one of the developers to have permission to use this software.
* Any kind of use, reproduction, distribution of this software without our permission
* is against the law. 
*/

#ifndef _TRANSPOSE_GRID
#define _TRANSPOSE_GRID


__global__ void transposeGrid(float *input, float *output, int ncols, int nrows){
	/*
	 * Transposed the given grid to be used later on the blur operation
	*/
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int tableSize = ncols*nrows;
	if (i < tableSize){
		int col = i % nrows;
		int row = i / nrows;
		int index = (col*ncols) + row;

		*(output+i) = *(input+index);
	}
}

#endif // _TRANSPOSE_GRID