#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "imatrix.h"
#include "ETF.h"
#include "fdog.h"
#include "myvec.h"

#define USE_VIDEO false
#define SAVE_IMAGE false

using namespace cv;
using namespace std;

void withVideo(CvCapture* capture);
void withoutVideo(Mat& outputImage, Mat originalImage);
void convertToMat(Mat& frame, imatrix img, int height, int width);
void runCLDWork(imatrix& img);

int main() {
	CvCapture* capture;

	// Read the video stream
	capture = cvCaptureFromCAM(-1);

	namedWindow("Output Image", CV_WINDOW_AUTOSIZE);

	if (USE_VIDEO) {
		withVideo(capture);
	} else {
		Mat originalImage = imread("/home/student/Pictures/Webcam/test1.jpg");
		Mat finalImage;
		withoutVideo(finalImage, originalImage);
		imshow("Output Image", finalImage);
		if (SAVE_IMAGE) {
			imwrite("/home/student/Pictures/cartoon.jpg", finalImage);
		}
	}

	waitKey(0);
	cvDestroyWindow("Output Image");
}

void withVideo(CvCapture* capture) {
	Mat originalFrame, grayFrame;
	while (true) {
		// get the next video frame
		originalFrame = cvQueryFrame(capture);
		cvtColor(originalFrame, grayFrame, CV_RGB2GRAY);
		imatrix img;

		int height = originalFrame.rows;
		int width = originalFrame.cols;

		img.init(height, width);

		// copy from dst (unsigned char) to img (int)
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				img[y][x] = grayFrame.at<unsigned char>(y, x);
			}
		}

		runCLDWork(img);

		convertToMat(grayFrame, img, height, width);
		imshow("Output Image", grayFrame);

		waitKey(10);
	}
}

void withoutVideo(Mat& outputImage, Mat originalImage) {
	Mat grayFrame;
	cvtColor(originalImage, grayFrame, CV_RGB2GRAY);
	imatrix img;

	int height = originalImage.rows;
	int width = originalImage.cols;

	img.init(height, width);

	// copy from dst (unsigned char) to img (int)
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img[y][x] = grayFrame.at<unsigned char>(y, x);
		}
	}

	runCLDWork(img);

	convertToMat(grayFrame, img, height, width);
	outputImage = grayFrame;
}

void convertToMat(Mat& frame, imatrix img, int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
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
	//e.set(img); // get gradients from input image
	e.set2(img); // get gradients from gradient map
	e.Smooth(4, 2);

	double tao = 0.99;
	double thres = 0.7;
	GetFDoG(img, e, 1.0, 3.0, tao);
	GrayThresholding(img, thres);
}
