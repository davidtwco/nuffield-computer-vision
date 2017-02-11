#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include "bilateralFiltering/ciiBF.h"
#include "bilateralFiltering/ciiBF.c"

#include "cld/imatrix.h"
#include "cld/ETF.h"
#include "cld/fdog.h"

using namespace std;
using namespace cv;

#define SHOW_CONTROLS true
#define USE_VIDEO false
#define SAVE_IMAGE true

int quantLevel = 4;
int bilatFilterSize = 1;

void update();
Mat runComputations(Mat originalFrame, int bilatFilterSize = 5, int quantizationLevel = 7, bool filterTwice = true, float bilatAlpha = 255);
Mat runBilteralFilter(Mat input, int spatialRadius, float rangeStd);
void convertToKangMatrix(Mat frame, imatrix& img);
void convertFromKangMatrix(Mat& frame, imatrix img);
void runCLDWork(imatrix& img);
void quantize(Mat& image, int quadrants);
void updateCallback(int, void*);

int main() {

	if (SHOW_CONTROLS) {
		namedWindow("Control", CV_WINDOW_AUTOSIZE);
		createTrackbar("Quantisation Level", "Control", &quantLevel, 12, updateCallback);
		createTrackbar("Bilateral Filter Size", "Control", &bilatFilterSize, 12, updateCallback);
	}

	update();

	waitKey(0);
	cvDestroyWindow("Output Image");

	return 0;
}

void update() {
	if (USE_VIDEO) {
		namedWindow("Output Image", CV_WINDOW_NORMAL);
		CvCapture* capture;
		capture = cvCaptureFromCAM(-1);
		cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 320);
		cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 240);

		while (true) {
			Mat originalFrame = cvQueryFrame(capture);
			imshow("Output Image", runComputations(originalFrame, bilatFilterSize, quantLevel));
			waitKey(10);
		}
	} else {
		namedWindow("Output Image", CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

		Mat originalFrame = imread("/home/student/Pictures/Finals/horsecarriage.jpg");
		Mat outputFrame = runComputations(originalFrame, bilatFilterSize, quantLevel);
		imshow("Output Image", outputFrame);
		if (SAVE_IMAGE) {
			imwrite("/home/student/Pictures/abstraction.jpg", outputFrame);
		}
	}
}

Mat runComputations(Mat originalFrame, int bilatFilterSize, int quantizationLevel, bool filterTwice, float bilatAlpha) {
	Mat yCrCbFrame;
	cvtColor(originalFrame, yCrCbFrame, CV_RGB2YCrCb);

	// Split into channels.
	Mat yCh[3];
	split(yCrCbFrame, yCh);

	// Running the bilateral filter.
	Mat postBilat = runBilteralFilter(yCh[0], bilatFilterSize, bilatFilterSize);
	// Bilat filter converts to 32F depth, this changes back to 8U
	postBilat.convertTo(postBilat, CV_8UC1, 180);

	// Kang-ing the bilateral filtered frame.
	Mat postKang;
	imatrix img;
	convertToKangMatrix(postBilat, img);
	runCLDWork(img);
	convertFromKangMatrix(postKang, img);

	if (filterTwice) {
		// Running the bilateral filter.
		postBilat = runBilteralFilter(yCh[0], bilatFilterSize, bilatFilterSize);
		// Bilat filter converts to 32F depth, this changes back to 8U
		postBilat.convertTo(postBilat, CV_8UC1, bilatAlpha);
	}

	// Quantize.
	Mat postQuant = postBilat.clone();
	quantize(postQuant, quantizationLevel);

	// Removing the white, from the Kang'd image; The cols and rows may seem the wrong way around - they aren't.
	Mat mask, allBlack = Mat::zeros(Size(postKang.cols, postKang.rows), CV_8UC1);
	inRange(postKang, Scalar(0, 0, 0), Scalar(80, 80, 80), mask);
	// Merges the frames by copying all non-zero elements from an all black image where the mask is to postQuant
	allBlack.copyTo(postQuant, mask);

	// Join the old CrCb channels with the Y channel.
	Mat finishedFrame = Mat(Size(postQuant.rows, postQuant.cols), CV_8UC3);
	yCh[0] = postQuant;
	// Channels must have same size and depth. 2nd parameter must be equal to total no. of channels in yCh.
	merge(yCh, 3, finishedFrame);

	//Convert back to RGB.
	Mat finishedRGBFrame;
	cvtColor(finishedFrame, finishedRGBFrame, CV_YCrCb2RGB);
	return finishedRGBFrame;
}

Mat runBilteralFilter(Mat input, int spatialRadius, float rangeStd) {
	int height, width, step;
	int i, j;
	float *data2;

	int nc = ceil(1.f / rangeStd); // number of coeffs. to use

	// load a grayscale image
	Mat* img = &input;

	// get the image data
	height = (*img).rows;
	width = (*img).cols;
	step = (*img).step;

	uchar *imdata = (uchar *) (*img).data;

	uchar *data = new uchar[height * width];
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			data[i * width + j] = imdata[i * step + j];
		}
	}

	// result image
	Mat fimg2WithoutTheDamnPointers = Mat(Size(width, height), CV_32FC1, 1);
	Mat *fimg2 = &fimg2WithoutTheDamnPointers;
	//cvSet(fimg2, cvScalar(0));
	data2 = (float *) (*fimg2).data;

	int hw = height * width;

	// auxiliary images
	float *II = new float[hw]; // integral image
	float *W = new float[hw]; // Normalisation factor for each window (sum of weights)

	rangeStd *= (EE_MAX_IM_RANGE - 1);
	float dctc[EE_MAX_IM_RANGE];
	float gker[EE_MAX_IM_RANGE];
	for (int i = 0; i < EE_MAX_IM_RANGE; ++i) {
		gker[i] = exp(-0.5 * pow(i / rangeStd, 2));
	}
	Mat G(1, EE_MAX_IM_RANGE, CV_32F, gker);
	Mat D(1, EE_MAX_IM_RANGE, CV_32F, dctc);
	dct(G, D);
	dctc[0] /= sqrt(2); // for the inverse computation

	ciiBF(data, data2, dctc, II, W, height, width, nc, spatialRadius);

	return (*fimg2);
}

void convertToKangMatrix(Mat frame, imatrix& img) {
	img.init(frame.rows, frame.cols);
	for (int y = 0; y < frame.rows; y++) {
		for (int x = 0; x < frame.cols; x++) {
			img[y][x] = frame.at<unsigned char>(y, x);
		}
	}
}

void convertFromKangMatrix(Mat& frame, imatrix img) {
	frame = Mat(Size(img.getCol(), img.getRow()), CV_8UC1);
	for (int y = 0; y < img.getRow(); y++) {
		for (int x = 0; x < img.getCol(); x++) {
			frame.at<unsigned char>(y, x) = img[y][x];
		}
	}
}

void runCLDWork(imatrix& img) {
	// We assume that you have loaded your input image into an imatrix named "img"
	int image_x = img.getRow();
	int image_y = img.getCol();

	ETF e;
	e.init(image_x, image_y);
	e.set2(img); // get gradients from gradient map
	e.Smooth(4, 2);

	double tao = 0.99;
	double thres = 0.7;
	GetFDoG(img, e, 1.0, 3.0, tao);
	GrayThresholding(img, thres);
}

void quantize(Mat& image, int quadrants = 8) {
	int step = floor(255 / quadrants - 1);
	int rem = 255 - (step * quadrants - 1);

	for (int j = 0; j < image.rows; j++) {
		for (int i = 0; i < image.cols; i++) {
			uchar data = image.ptr<uchar>(j)[i];

			for (int k = 0; k < quadrants; k++) {
				int lbound = (k * step);
				int ubound = ((k + 1) * step);

				if (data >= lbound && data < ubound) {
					data = ubound;
					break;
				}
			}
			if (rem != 0) {
				if (data >= (255 - rem) && data < 255) {
					data = 255;
				}
			}

			image.at<unsigned char>(j, i) = data;
		}
	}
}

void updateCallback(int, void*) {
	update();
}
