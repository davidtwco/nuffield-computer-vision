// This demo loads an image and performs efficient bilateral filtering using
// Coasine Integral Images (CII).
//
// CII are described in the paper:
// Cosine integral images for fast spatial and range filtering
// Elhanan Elboher, Michael Werman
// ICIP 2011
//
//
// NOTE: We apply box bilateral filtering, which means that the weights of the
// neighbors within the window N(x0) only depend on their intensity difference
// from the current location x0, but NOT on spatial distance.
//
//
// PARAMETERS
//
// spatial-radius: the radius of the neighborhood N(x0), which is a rectangular window
// of size 2 * radius + 1 around each pixel x0.
//
// range-std: the standard deviation of the range Gaussian kernel; should be in (0,1].
//
// out-file-name: use this parameter in case that you want to save the output image.

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <time.h>

#include "ciiBF.h"
#include "ciiBF.c"

using namespace cv;

int main(int argc, char *argv[]) {

	int height, width, step, channels;
	int i, j;
	float *data2;

	/////////////////////////////////////////////////////////////////////////////

	// (1) Read parameters

	if (argc < 4) {
		printf("Usage: bf_demo <image-file-name> <spatial-radius> <range-std> <optional: out-file-name> \n");
		exit(0);
	}

	char *imname = argv[1];

	int r = atoi(argv[2]); // spatial raduis

	float sx = atof(argv[3]); // range std

	int nc = ceil(1.f / sx); // number of coeffs. to use
	printf("using %d DCT coefficients.\n", nc);

	char *outname = 0; // out file name
	if (argc > 4) {
		outname = argv[4];
	}

	/////////////////////////////////////////////////////////////////////////////

	// (2) Read an image

	// load a grayscale image
	Mat imgWithoutTheDamnPointers = imread(imname, CV_LOAD_IMAGE_GRAYSCALE);
	Mat* img = &imgWithoutTheDamnPointers;
	if (!img) {
		printf("Could not load image file: %s\n", imname);
		exit(0);
	}

	// get the image data
	height = (*img).rows;
	width = (*img).cols;
	step = (*img).step;
	channels = (*img).channels();

	uchar *imdata = (uchar *) (*img).data;

	uchar *data = new uchar[height * width];
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			data[i * width + j] = imdata[i * step + j];
		}
	}

	/////////////////////////////////////////////////////////////////////////////
	// (3) Allocate memory

	// result image
	Mat fimg2WithoutTheDamnPointers = Mat(Size(width, height), CV_32FC1, 1);
	Mat *fimg2 = &fimg2WithoutTheDamnPointers;
	//cvSet(fimg2, cvScalar(0));
	data2 = (float *) (*fimg2).data;

	int hw = height * width;

	// auxiliary images
	float *II = new float[hw]; // integral image
	float *W = new float[hw]; // normalization factor for each window (sum of weights)

	////////////////////////////////////////////////////////////////////////////////

	// (4) CII range filtering

//	clock_t cl_start, cl_end;
//	cl_start = clock();

	// (4a) cosine transform for Gaussian with std = sx

	sx *= (EE_MAX_IM_RANGE - 1);
	float dctc[EE_MAX_IM_RANGE];
	float gker[EE_MAX_IM_RANGE];
	for (int i = 0; i < EE_MAX_IM_RANGE; ++i) {
		gker[i] = exp(-0.5 * pow(i / sx, 2));
	}
	cv::Mat G(1, EE_MAX_IM_RANGE, CV_32F, gker);
	cv::Mat D(1, EE_MAX_IM_RANGE, CV_32F, dctc);
	cv::dct(G, D);
	dctc[0] /= sqrt(2); // for the inverse computation

	// (4b) range filtering

	ciiBF(data, data2, dctc, II, W, height, width, nc, r);

//	cl_end = clock();
//	float cpu_time = float(cl_end - cl_start) / CLOCKS_PER_SEC;
//	printf("cpu time = %g seconds.\n", cpu_time);

	// save the result

	if (argc > 4) {
		Mat tmpoutWithoutTheDamnPointers = Mat(Size(width, height), (*img).depth(), channels);
		Mat* tmpout = &tmpoutWithoutTheDamnPointers;
		cvConvertScaleAbs(fimg2, tmpout, 255, 0);
		printf("saving image\n");
		cvSaveImage(outname, tmpout);
		(*tmpout).release();
	}

	/////////////////////////////////////////////////////////////////////////////////////

	// (5) display the results

	// show the input:
	namedWindow("Input", CV_WINDOW_AUTOSIZE);
	imshow("Input", (*img));

	// show cii result:
	namedWindow("CII", CV_WINDOW_AUTOSIZE);
	imshow("CII", (*fimg2));

	cvWaitKey(0);

	// release the images
	(*img).release();
	(*fimg2).release();

	delete[] data;
	delete[] II;
	delete[] W;

	return 0;

}

// Copyright (c) 2011, Elhanan Elboher
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met: 
//    * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//    * Neither the name of the The Hebrew University of Jerusalem nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

