/*
* Copyright 2020 Aiman bin Murhiz, Timotius Pujianto, James Schubach. All rights reserved.
* Bachelor of Computer Science Final project, Semester 1 2020, Monash University, Australia.

* Please contact one of the developers to have permission to use this software.
* Any kind of use, reproduction, distribution of this software without our permission
* is against the law. 
*/

#ifndef _GRADIENT_OPERATOR_CH_
#define _GRADIENT_OPERATOR_CH_

extern __shared__ float *src; 

__device__ float 
getValue(int ncols, int col, int row, float *src){
    int index = (row*ncols) + col;
	return *(src+index);
}

// Implementation of get8NeighborGradient from Grid.java
__device__ float 
get8NeighborGradient(int col, int row, int ncols, int nrows, int cellsize){
	/*
     * Returns the dimensionless rise/run slope computed from 8 neighboring
     * cells.
     *
     * Equation from:
     * http://help.arcgis.com/en/arcgisdesktop/10.0/help../index.html#/How_Slope_works/009z000000vz000000/
	 *
	 * Taken from Grid.java, get8neighbourgradient function
	*/

	float cellSizeTimes8, a,b,c,d,f,g,h,i,dZdX,dZdY, result;
	int  colLeft, colRight, rowTop, rowBottom;
	
	cellSizeTimes8 = cellsize * 8;

	if (col > 0){
		colLeft = col-1;
	}
	else{
		colLeft = 0;
	}

	if (col < ncols-1){
		colRight = col + 1;
	}
	else{
		colRight = ncols-1;
	}

	if (row > 0){
		rowTop = row-1;
	}
	else{
		rowTop = 0;
	}

	if (row < nrows-1){
		rowBottom = row + 1;
	}
	else{
		rowBottom = nrows-1;
	}
	a = getValue(ncols, colLeft, rowTop, src);
	b = getValue(ncols, col, rowTop, src);
	c = getValue(ncols, colRight, rowTop, src);
	d = getValue(ncols, colLeft, row, src);

	f = getValue(ncols, colRight, row, src);
	g = getValue(ncols, colLeft, rowBottom, src);
	h = getValue(ncols, col, rowBottom, src);
	i = getValue(ncols, colRight, rowBottom, src);

	dZdX = ((c + (2 * f) + i) - (a + (2 * d) + g)) / cellSizeTimes8;
	dZdY = ((g + (2 * h) + i) - (a + (2 * b) + c)) / cellSizeTimes8;

	result = ((dZdX * dZdX) + (dZdY * dZdY));
	return sqrt(result);

}

__global__ void 
gradientOperator(int ncols, int nrows,int cellsize, float tableSize, float *inputGrid, float *resultGradientOperator){
	/*
	* Compute dimensionless gradient (or slope steepness) as rise/run, not an
 	* angle. Use SlopeOperator if slope in degree is needed.
 	* Documentation taken from GradientOperator.java
	*/
	int col, row;
	src = inputGrid;

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < tableSize){
		col = i % ncols;
		row = i / ncols;
		*(resultGradientOperator+i) = get8NeighborGradient(col, row, ncols, nrows, cellsize);
	}
}
#endif // _GRADIENT_OPERATOR_CH_