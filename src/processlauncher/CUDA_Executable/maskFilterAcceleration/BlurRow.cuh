/*
* Copyright 2020 Aiman bin Murhiz, Timotius Pujianto, James Schubach. All rights reserved.
* Bachelor of Computer Science Final project, Semester 1 2020, Monash University, Australia.

* Please contact one of the developers to have permission to use this software.
* Any kind of use, reproduction, distribution of this software without our permission
* is against the law. 
*/

#ifndef _BLUR_ROW
#define _BLUR_ROW


__device__ int extension(int N, int n){
	/*
	 * Translated directly from eduard-AbstractFrequencyOperator.java
	 * Reflect index for out-of-bounds access. This is currently a
	 * bottleneck and could be avoided for the center of the grid (if the
	 * filter is smaller than the grid).
	*/
    while (true){
		if (n<0){
			n= -1-n;
		}else if (n>= N){
			n = 2* N -1 -n; 
		}else {
			break;
		}
	}
	return n;
}

// Blur row, translated from eduard java.
// Blurs a row once.
__global__ void blurRow(float *input, float *output, int N, int nrow, int r, float c1, float c2){

	double sum=0;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i< nrow){
		for (int n = -r; n<=r; ++n){
			sum += input[extension(N,n) + (i*N)];
		}

		sum = c1 * (input[extension(N,r+1) + (i*N)] + input[extension(N,-r-1) + (i*N)]) + (c1+c2) * sum;
		*(output + (i*N)) = (float) sum;
	
		for(int n=1;n<N;++n){
			sum += c1 * (input[extension(N,n+r+1) + (i*N)] - input[extension(N,n-r-2) + (i*N)]) + c2 * (input[extension(N,n+r)+ (i*N)]-input[extension(N,n-r-1)+ (i*N)]);
			*(output+ (i*N) + n) = (float) sum;
		}
	}
}
#endif // _BLUR_ROW