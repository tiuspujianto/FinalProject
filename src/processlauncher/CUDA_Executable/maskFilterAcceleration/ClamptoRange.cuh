/*
* Copyright 2020 Aiman bin Murhiz, Timotius Pujianto, James Schubach. All rights reserved.
* Bachelor of Computer Science Final project, Semester 1 2020, Monash University, Australia.

* Please contact one of the developers to have permission to use this software.
* Any kind of use, reproduction, distribution of this software without our permission
* is against the law. 
*/

#ifndef _CLAMP_TO_RANGE
#define _CLAMP_TO_RANGE

#include <math.h>

__device__ float maximum(float a, float b){
	return a>b? a : b;
}
__device__ float minimum(float a, float b){
	return a<b? a :b;
}

__global__ void clampToRange(float *grid, float min, float max, int tableSize){
	/*
	 * Clamp grid value to the given min-max value
	*/
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<tableSize){
		*(grid+i) = minimum(max, maximum(min, *(grid+i)));
	}
}
__global__ void maskFilter(float *grid, int tableSize, float gainSlopeThreshold, float scale){
	/*
	 * Blurred slope values are now between gainSlopeThreshold and slopeThreshold.
     * Scale all slope values from [gainSlopeThreshold..slopeThreshold] to [0..1].
	 * Inverted mapping of slopeThreshold to 0 and gainSlopeThreshold to 1.
	 * Taken from Main.java, maskFilter function.
	*/
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	float v;
	if (i<tableSize){
		v = *(grid+i);
		if (isfinite(v)){
			v = 1 - (v - gainSlopeThreshold) * scale;
			v = maximum(0, minimum(1, v));
		}else{
			v = -1;
		}
		*(grid+i) = v;
	}
}
#endif // _CLAMP_TO_RANGE