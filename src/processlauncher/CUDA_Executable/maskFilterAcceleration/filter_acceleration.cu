/*
* Copyright 2020 Aiman bin Murhiz, Timotius Pujianto, James Schubach. All rights reserved.
* Bachelor of Computer Science Final project, Semester 1 2020, Monash University, Australia.

* Please contact one of the developers to have permission to use this software.
* Any kind of use, reproduction, distribution of this software without our permission
* is against the law. 
*/


/*
* Project: "GPU Acceleration of Raster Filters."
*
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// cuda kernel
#include "GradientOperator.cuh"
#include "ClampToRange.cuh"
#include "BlurRow.cuh"
#include "TransposeGrid.cuh"

// std
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <errno.h>


float ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value;
float *LowPass1st, *LowPass2nd, *ClampToRange, *transGrid1, *transGrid2;
int numberOfBlocks, numberOfThreads=1024; 

int readHeader(char* strFromFile, float valueHolder){
	/*
	* Read header function
	* this function intended to get the value of the header format
	* and put it in the right variable.
	* strFromFile = the header name
	* valueHolder = the value from the header part
	* return -1 when the value and the header name do not match
	* it means that the buffer is in grid value instead of header part
	*/

    if (strcmp(strFromFile,"ncols") == 0){
        ncols = valueHolder;
	}else if (strcmp(strFromFile, "nrows") == 0){
        nrows = valueHolder;
	}else if (strcmp(strFromFile, "xllcorner") == 0){
        xllcorner = valueHolder;
	}else if (strcmp(strFromFile, "yllcorner") == 0){
        yllcorner = valueHolder;
	}else if (strcmp(strFromFile, "cellsize") == 0){
        cellsize = valueHolder;
	}else if (strcmp(strFromFile, "nodata_value") == 0){
        nodata_value = valueHolder;
		return -1;
	}else {
        return -1;
	}
	return 0;
}

void LowPassOperator(float *input, float *output, float sigma, bool firstpass){
	/*
	* Low Pass Operator function
	* this function applies gaussian algorithm to each row in order to blur the image
	*
	* Taken from Eduard java, under the AbstractFrequencyOperator.java
	*
	*  * For an introduction to the topic:
	* https://fgiesen.wordpress.com/2012/07/30/fast-blurs-1/
	* https://fgiesen.wordpress.com/2012/08/01/fast-blurs-2/
	*
	* Paper comparing different blur algorithms with reference code:
	*
	* Getreuer 2013 A Survey of Gaussian Convolution Algorithms
	* http://dev.ipol.im/~getreuer/code/doc/gaussian_20131215_doc/index.html
	*
	* Original paper introducing extended box filters:
	*
	* P. Gwosdek, S. Grewenig, A. Bruhn, J. Weickert, “Theoretical foundations of
	* Gaussian convolution by extended box filtering,” International Conference on
	* Scale Space and Variational Methods in Computer Vision, pp. 447–458, 2011.
	* http://dx.doi.org/10.1007/ 978-3-642-24785-9_38
	*
	* Input = the input grid
	* Output = the result grid of the low pass operation
	* sigma = the sigma blur coefficient
	* firstpass = a flag indication if the input grid has been transposed or not
	*/

	float *dstRow, *tmpRow;
	
	float alpha, c1,c2;
	int r, ITERATIONS = 4, block, N, totalrow;

	r = floor((0.5 * sqrt(((12.0*sigma*sigma)/ITERATIONS)+1.0)) - 0.5);
	alpha = (2 * r + 1) * (r * (r + 1) - 3.0 * sigma * sigma / ITERATIONS) / (6.0 * (sigma * sigma / ITERATIONS - (r + 1) * (r + 1)));
	c1 = (alpha / (2.0 * (alpha + r) + 1));
	c2 = ((1.0 - alpha) / (2.0 * (alpha + r) + 1));

	cudaMallocManaged(&dstRow, nrows*ncols*sizeof(float));
	cudaMallocManaged(&tmpRow, nrows*ncols*sizeof(float));

	// Setting up the number of block will be used in cuda multi processing
	// Firstpass means that the grid has not been transposed
	// then the value of coloumn and rows will not be switched. 
	if (firstpass){
		block = (nrows/1024)+1;
		N = ncols;
		totalrow = nrows;
	}else{
		block = (ncols/1024)+1;
		N = nrows;
		totalrow = ncols;
	}

	/* Calling cuda function to do blur
	* If the number of iterations is change,
	* the following code block must be change.
	*/
	blurRow<<<block, numberOfThreads>>>(input, tmpRow, N, totalrow, r, c1,c2);
	cudaDeviceSynchronize();
	blurRow<<<block, numberOfThreads>>>(tmpRow, dstRow, N, totalrow, r, c1,c2);
	cudaDeviceSynchronize();
	blurRow<<<block, numberOfThreads>>>(dstRow, tmpRow, N, totalrow, r, c1,c2);
	cudaDeviceSynchronize();
	blurRow<<<block, numberOfThreads>>>(tmpRow, output, N, totalrow, r, c1,c2);
	cudaDeviceSynchronize();
	
	// Free the memory
	cudaFree(tmpRow);
	cudaFree(dstRow);
}

float toRadians(float deg){
	/*
	* To radians function, convert the given input (in degree) into radians
	* Paramater1 = the degree
	* return = radians in float data type
	*/
	return deg*22/7/180;
}

void writeOutput(float *transGrid2, int gridTotalSize){
	/*
	* Write output function to write the filtered grid into the asc file
	* The asc file will be used in Eduard to convert it into a png file
	*/

	// Open the output file (or create it if it jas not been created before)
	char* fileName = "out.asc";
	FILE *file = fopen(fileName, "w+");
	// Error handling if the program failed to create the file
	if (file == NULL){
		fprintf(stderr, "%s\n", strerror(errno));
		return;
	}else{
		//header part
		fprintf(file, "%s %d\n", "ncols", int(ncols));
		fprintf(file, "%s %d\n", "nrows", int(nrows));
		fprintf(file, "%s %.1f\n", "xllcorner", xllcorner);
		fprintf(file, "%s %.1f\n", "yllcorner", yllcorner);
		fprintf(file, "%s %.1f\n", "cellsize", cellsize);
		fprintf(file, "%s %.1f\n", "nodata_value", nodata_value);
		
		// Writing the grid value into the file.
		for (int index =0; index<gridTotalSize; index++){
			fprintf(file, "%.3f ", *(transGrid2+index));
		}
	}
	fclose(file);
}

int main(int argc, char** argv){
	/*
	* Main function to run the masking filter in cuda.
	*
	*/
    
    char strHeader[256];
    float valueHolder, gridTotalSize;
	float *inputGrid, *resultGradientOperator;
	time_t start, end, subStart, subEnd;
	double totalTime;
	char* fileName;
	FILE *f;


	if (argc>0){
		fileName = argv[1];
	}else{
		return 1;
	}
    
    // Open the input grid file ***.asc
    
	f = fopen(fileName, "r");

	if (f==NULL){
		fprintf(stderr, "%s\n", strerror(errno));
	}else{

		fscanf(f,"%s %f", strHeader, &valueHolder);

		// Flag to indicate where the buffer is. 0 means it is still in the header part, 
		// -1 means it reached the grid value.
		int flag = readHeader(strHeader, valueHolder);
		
		// A loop to read trough the header of the input file
		// flag = 0 means that it is still reading the header	
		while (flag == 0){
			fscanf(f,"%s %f", strHeader, &valueHolder);
			// Make the string of the header all lower case.
			for (int i = 0; strHeader[i]; i++){
				strHeader[i] = tolower(strHeader[i]);
			}
			// update the flag
			flag = readHeader(strHeader, valueHolder);
		}

		// Check if the grid header is valid or not
		
		if ((ncols<0 || nrows < 0 || cellsize < 0) == 1){
			// return error here
			return 1;
		}

		gridTotalSize = ncols * nrows;
		
		cudaMallocManaged(&inputGrid, gridTotalSize * sizeof(float));
		cudaMallocManaged(&resultGradientOperator, gridTotalSize * sizeof(float));

		// Scan the grid values and put them in the buffer called inputGrid
		for (int i = 0; i < int(gridTotalSize); i++){
			fscanf(f,"%f", &valueHolder);
			*(inputGrid+i) = valueHolder;
		}
		fclose(f);

		if (int(gridTotalSize)%numberOfThreads == 0){
			numberOfBlocks = gridTotalSize / numberOfThreads;
		}else{
			numberOfBlocks = (gridTotalSize / numberOfThreads) + 1;
		}

		// compute grid with dimensionless rise/run slope values instead of slope in 
		// degrees, which would require an expensive atan() operation for each 
		// cell. Results with rise/run are almost identical to results with degrees.
		start = clock();
		subStart= clock();
		gradientOperator<<<numberOfBlocks,numberOfThreads>>>(ncols, nrows,cellsize, gridTotalSize, inputGrid, resultGradientOperator);
		cudaDeviceSynchronize();
		cudaFree(inputGrid);
		subEnd = clock();
		totalTime = (double) (subEnd-subStart)/ CLOCKS_PER_SEC;
		printf("The gradient operator time : %lf\n", totalTime);
		
		// Allocate shared memory between host and device (gpu)
		cudaMallocManaged(&LowPass1st, gridTotalSize*sizeof(float));
		cudaMallocManaged(&LowPass2nd, gridTotalSize*sizeof(float));
		cudaMallocManaged(&transGrid1, gridTotalSize*sizeof(float));
		cudaMallocManaged(&transGrid2, gridTotalSize*sizeof(float));

		// Blur row to smooth the sharp edge via lowpassoperator
		float sigma = 6.;
		subStart = clock();
		LowPassOperator(resultGradientOperator, LowPass1st, sigma, true);
		transposeGrid<<<numberOfBlocks,numberOfThreads>>>(LowPass1st, transGrid1, ncols,nrows);
		cudaDeviceSynchronize();	

		LowPassOperator(transGrid1, LowPass2nd, sigma, false);
		transposeGrid<<<numberOfBlocks,numberOfThreads>>>(LowPass2nd, transGrid2, nrows,ncols);
		cudaDeviceSynchronize();
		subEnd = clock();
		totalTime = (double) (subEnd-subStart)/ CLOCKS_PER_SEC;
		printf("The first low pass operator time : %lf\n", totalTime);

		// Clamp slope values to range between gainSlopeThreshold and slopeThreshold
		float relativeGain =0.5, slopeThresholdDeg = 6.;
		float slopeThreshold = tan(toRadians(slopeThresholdDeg));
		float gainSlopeThresholdDeg = slopeThreshold * fmin(0.995, relativeGain);
		float gainSlopeThreshold = tan(toRadians(gainSlopeThresholdDeg));

		subStart = clock();
		clampToRange<<<numberOfBlocks,numberOfThreads>>>(transGrid2, gainSlopeThreshold, slopeThreshold, gridTotalSize);
		cudaDeviceSynchronize();
		subEnd = clock();
		totalTime = (double) (subEnd-subStart)/ CLOCKS_PER_SEC;
		printf("The clamp to range operator time : %lf\n", totalTime);

		// Blur the sharp edges once more via lowpassoperator
		sigma = 20.;
		subStart = clock();
		LowPassOperator(transGrid2, LowPass1st, sigma, true);
		transposeGrid<<<numberOfBlocks,numberOfThreads>>>(LowPass1st, transGrid1, ncols,nrows);
		cudaDeviceSynchronize();

		LowPassOperator(transGrid1, LowPass2nd, sigma, false);
		transposeGrid<<<numberOfBlocks,numberOfThreads>>>(LowPass2nd, transGrid2, nrows,ncols);
		cudaDeviceSynchronize();
		subEnd = clock();
		totalTime = (double) (subEnd-subStart)/ CLOCKS_PER_SEC;
		printf("The second low pass operator time : %lf\n", totalTime);

		// Mask Filter
		float scale = 1/(slopeThreshold-gainSlopeThreshold);
		subStart = clock();
		maskFilter<<<numberOfBlocks,numberOfThreads>>>(transGrid2, gridTotalSize, gainSlopeThreshold, scale);
		cudaDeviceSynchronize();
		subEnd = clock();
		totalTime = (double) (subEnd-subStart)/ CLOCKS_PER_SEC;
		printf("The gradient operator time : %lf\n", totalTime);

		
		end = clock();
		totalTime = (double) (end-start)/ CLOCKS_PER_SEC;
		printf("The total filter time : %lf\n", totalTime);

		writeOutput(transGrid2, gridTotalSize);
	}
	return 0;
}