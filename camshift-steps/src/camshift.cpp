#define _USE_MATH_DEFINES
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

#define ENABLE_CONTROLS          false

Mat originalFrame;
Point2f contourCentre;
int largestContourIndex = 0;

Point lastTop;

// These need to be global to be accessed from the callback.
bool regionSelected;
Point mousePos;
Point mouseDownPos;
Point mouseUpPos;
Point properMouseUpPos;
Point lastMouseDownPos;

void waitAndShow(Mat frame, int delay);
void getRangeFromSelection(int& lowerBound, int& upperBound);
void findMoments(vector<Moments>& imageMoments, Mat thresholdFrame);
void drawOverlay(float length, float width, float angle, Point h1, Point h2, Point v1, Point v2);
void drawNonRelatedOverlay(int minH, int maxH);
void drawText(string label, float value, Point position, Scalar colour, bool fromBottom);
void drawText(string text, Point position, Scalar colour);
void drawRotatedBox(float angle, Point h1, Point h2, Point v1, Point v2);
void runCalculations(float& angle, float& width, float& length, Point& h1, Point& h2, Point& v1, Point& v2, vector<Moments> mu);
void calculateLengthWidth(float& length, float& width, vector<Moments> mu, Point2f centre, int i);
float calculateAngleInRad(vector<Moments> mu, Point2f centre, int i);
void getPoints(Point& h1, Point& h2, Point& v1, Point& v2, float length, float width, float angle, Point2f centre);
Point getTop(Point p1, Point p2);

int delay = 1000;

int main() {

	// Faces.
	int minHSV = 0, maxHSV = 15;
	int gaussianKernalSize = 3;

	Mat HSV_Image, H_Image, H_Image_Threshold;
	Mat HSV_Ch[3];

	if (ENABLE_CONTROLS) {
		// finding different colour range
		namedWindow("Control", CV_WINDOW_AUTOSIZE);

		//Create control bars for H S V
		cvCreateTrackbar("HSV_Min_H", "Control", &minHSV, 180);
		cvCreateTrackbar("HSV_Max_H", "Control", &maxHSV, 180);
	}

	namedWindow("Output Original Image", CV_WINDOW_AUTOSIZE);

//	while (!regionSelected) {
//		// get the next video frame
//		originalFrame = cvQueryFrame(capture);
//		drawNonRelatedOverlay(minHSV, maxHSV);
//
//		if (mouseDownPos != Point() && properMouseUpPos != Point()) {
//			getRangeFromSelection(minHSV, maxHSV);
//		}
//
//		imshow("Output Original Image", originalFrame);
//		waitKey(10);
//	}

// Get the next video frame
	originalFrame = imread("/home/student/Pictures/bluepen.jpg");
	waitAndShow(originalFrame, delay);

	// Convert to gray
	cvtColor(originalFrame, HSV_Image, CV_RGB2HSV);
	waitAndShow(HSV_Image, delay);

	// Get the HUE channel
	split(HSV_Image, HSV_Ch);

	// Threshold HSV image
	inRange(HSV_Ch[0], Scalar(minHSV, 0, 0), Scalar(maxHSV, 0, 0), H_Image_Threshold);
	waitAndShow(H_Image_Threshold, delay);
	imwrite("/home/student/Pictures/inrange.jpg", H_Image_Threshold);

	// Erosion
	erode(H_Image_Threshold, H_Image_Threshold, Mat(), Point(-1, -1), 3, 1, 1);
	waitAndShow(H_Image_Threshold, delay);

	// Gaussian filter to smooth the H frame
	GaussianBlur(H_Image_Threshold, H_Image_Threshold, Size(gaussianKernalSize, gaussianKernalSize), 0, 0);
	waitAndShow(H_Image_Threshold, delay);

	// Dilation
	dilate(H_Image_Threshold, H_Image_Threshold, Mat(), Point(-1, -1), 9, 1, 1);
	waitAndShow(H_Image_Threshold, delay);
	imwrite("/home/student/Pictures/complete.jpg", H_Image_Threshold);

	// Get moments
	vector<Moments> imageMoments;
	findMoments(imageMoments, H_Image_Threshold);
	waitAndShow(H_Image_Threshold, delay);

	if (imageMoments.size() > 0) {
		float angle, width, length;
		// float realAngle;
		Point h1, h2, v1, v2;
		runCalculations(angle, width, length, h1, h2, v1, v2, imageMoments);

		drawOverlay(length, width, angle, h1, h2, v1, v2);
	}
//	drawNonRelatedOverlay(minHSV, maxHSV);
	imwrite("/home/student/Pictures/completedraw.jpg",originalFrame);

	waitKey(0);
	cvDestroyWindow("Output Original Image");
}

void waitAndShow(Mat frame, int delay) {
	imshow("Original Output Image", frame);
	waitKey(delay);
}

void getRangeFromSelection(int& lowerBound, int& upperBound) {
	Rect area = Rect(mouseDownPos.x, mouseDownPos.y, abs(mouseUpPos.x - mouseDownPos.x), abs(mouseUpPos.y - mouseDownPos.y));
	Mat innerImage = originalFrame(area);
	Mat innerImageHSV;

	int rangeOffset = 8;

	// Convert to HSV
	cvtColor(innerImage, innerImageHSV, CV_RGB2HSV);

	MatND hist;
	int channels[] = { 0 };
	int hbins = 60; // Split into 60 sections, 3 values per bin.
	int histSize[] = { hbins };
	float hueRange[] = { 0, 180 };
	const float* ranges[] = { hueRange };
	calcHist(&innerImageHSV, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);

	double maxValue = 0;
	minMaxLoc(hist, 0, &maxValue, 0, 0);

	int highestBinVal = 0;
	int indexOfHighestBinVal = 0;
	for (int h = 0; h < hbins; h++) {
		int binVal = hist.at<int>(h);
		if (highestBinVal < binVal) {
			highestBinVal = binVal;
			indexOfHighestBinVal = h;
		}
	}

	int hueValue = (indexOfHighestBinVal * (180 / hbins));

	lowerBound = hueValue - rangeOffset;
	upperBound = hueValue + rangeOffset;

	if (lowerBound > upperBound) {
		int tempBound = lowerBound;
		lowerBound = upperBound;
		upperBound = tempBound;
	}

	if (lowerBound < 0)
		lowerBound = 0;
	if (upperBound > 180)
		upperBound = 180;

	regionSelected = true;
}

void findMoments(vector<Moments>& imageMoments, Mat thresholdFrame) {

	double largest_Area = 0, new_Area = 0;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Find contours
	findContours(thresholdFrame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Approximate contours to polygons + get bounding rects
	vector<vector<Point> > contours_poly(contours.size());

	// Get the moments
	vector<Moments> mu(contours.size());

	for (int i = 0; i < contours.size(); i++) {
		mu[i] = moments((Mat) contours[i], false); // get the moment of each contours

		new_Area = contourArea((Mat) contours[i]); //  Find the area of contour

		if (new_Area > largest_Area) {
			largest_Area = new_Area; // update the largest area

			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			// find the centre of largest contour
			contourCentre = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);

			largestContourIndex = i; // store the index of largest contour
			imageMoments = mu;
		}
	}
}

void drawOverlay(float length, float width, float angle, Point h1, Point h2, Point v1, Point v2) {
	float angleDeg = (angle / M_PI) * 180;
	float angleDegFixed = angleDeg;
	if (angleDeg < 0) {
		angleDegFixed = 360 + angleDeg;
	}

	RotatedRect rect = RotatedRect(contourCentre, Size2f(length * 4, width * 4), angleDegFixed);
	ellipse(originalFrame, rect, Scalar(0, 0, 255), 2, 8);
	waitAndShow(originalFrame, delay);

	line(originalFrame, h1, h2, Scalar(0, 255, 0));
	line(originalFrame, v1, v2, Scalar(0, 255, 0));
	waitAndShow(originalFrame, delay);

	Point top = getTop(h1, h2);
	circle(originalFrame, top, 5, Scalar(8, 240, 240));
	lastTop = top;
	waitAndShow(originalFrame, delay);

	drawRotatedBox(angleDeg, h1, h2, v1, v2);
	waitAndShow(originalFrame, delay);

//	drawText("Angle", angleDeg, Point(10, 40), Scalar(0, 255, 0), false);
//	drawText("Length", length, Point(10, 60), Scalar(0, 255, 0), false);
//	drawText("Width", width, Point(10, 80), Scalar(0, 255, 0), false);
}

void drawNonRelatedOverlay(int minH, int maxH) {
	if (regionSelected == false) {
		drawText("Mouse X", mousePos.x, Point(500, 40), Scalar(0, 255, 0), false);
		drawText("Mouse Y", mousePos.y, Point(500, 60), Scalar(0, 255, 0), false);
		drawText("MDown X", mouseDownPos.x, Point(500, 80), Scalar(0, 255, 0), false);
		drawText("MDown Y", mouseDownPos.y, Point(500, 100), Scalar(0, 255, 0), false);
		drawText("MUp X", mouseUpPos.x, Point(500, 120), Scalar(0, 255, 0), false);
		drawText("MUp Y", mouseUpPos.y, Point(500, 140), Scalar(0, 255, 0), false);

		rectangle(originalFrame, mouseDownPos, mouseUpPos, Scalar(255, 0, 0), 1, 8, 0);
	}
	drawText("Min H", minH, Point(10, 440), Scalar(0, 255, 0), false);
	drawText("Max H", maxH, Point(10, 460), Scalar(0, 255, 0), false);
}

void drawText(string text, Point position, Scalar colour) {
	putText(originalFrame, text, position, FONT_HERSHEY_PLAIN, 1.0f, colour, 1, 8, false);
}

void drawText(string label, float value, Point position, Scalar colour, bool fromBottom) {
	ostringstream convert;
	convert << value;
	putText(originalFrame, label + ": " + convert.str(), position, FONT_HERSHEY_PLAIN, 1.0f, colour, 1, 8, fromBottom);
}

void drawRotatedBox(float angle, Point h1, Point h2, Point v1, Point v2) {
	float a = abs(h2.x - contourCentre.x);
	float b = abs(h2.y - contourCentre.y);

	Point c1, c2, c3, c4;
	if (angle < 0) {
		c1 = Point(v1.x - a, v1.y + b);
		c2 = Point(v1.x + a, v1.y - b);
		c3 = Point(v2.x - a, v2.y + b);
		c4 = Point(v2.x + a, v2.y - b);
	} else {
		c1 = Point(v1.x - a, v1.y - b);
		c2 = Point(v1.x + a, v1.y + b);
		c3 = Point(v2.x - a, v2.y - b);
		c4 = Point(v2.x + a, v2.y + b);
	}

	line(originalFrame, c1, c2, Scalar(0, 0, 255), 2);
	line(originalFrame, c2, c4, Scalar(0, 0, 255), 2);
	line(originalFrame, c4, c3, Scalar(0, 0, 255), 2);
	line(originalFrame, c3, c1, Scalar(0, 0, 255), 2);
}

void runCalculations(float& angle, float& width, float& length, Point& h1, Point& h2, Point& v1, Point& v2, vector<Moments> mu) {
	angle = calculateAngleInRad(mu, contourCentre, largestContourIndex);

	calculateLengthWidth(length, width, mu, contourCentre, largestContourIndex);

	getPoints(h1, h2, v1, v2, length, width, angle, contourCentre);
}

void calculateLengthWidth(float& length, float& width, vector<Moments> mu, Point2f centre, int i) {
	float a = (mu[i].m20 / mu[i].m00) - pow(centre.x, 2);
	float b = 2 * ((mu[i].m11 / mu[i].m00) - (centre.x * centre.y));
	float c = (mu[i].m02 / mu[i].m00) - pow(centre.y, 2);

	length = sqrt(((a + c) + (sqrt(pow(b, 2) + pow((a - c), 2)))) * 0.5);
	width = sqrt(((a + c) - (sqrt(pow(b, 2) + pow((a - c), 2)))) * 0.5);
}

float calculateAngleInRad(vector<Moments> mu, Point2f centre, int i) {
	float top = 2 * ((mu[i].m11 / mu[i].m00) - (centre.x * centre.y));
	float bottom = ((mu[i].m20 / mu[i].m00) - pow(centre.x, 2)) - ((mu[i].m02 / mu[i].m00) - pow(centre.y, 2));
	return (atan2(top, bottom)) * 0.5;
}

void getPoints(Point& h1, Point& h2, Point& v1, Point& v2, float length, float width, float angle, Point2f centre) {
	float innerAngle = (0.5 * M_PI) - abs(angle);

	float vC = width;
	float vA = (vC * sin(innerAngle));
	float vB = sqrt(pow(vC, 2) - pow(vA, 2));

	float hC = length;
	float hA = (hC * sin(abs(angle)));
	float hB = sqrt(pow(hC, 2) - pow(hA, 2));

	vB = vB * 2;
	vA = vA * 2;
	hB = hB * 2;
	hA = hA * 2;

	if (angle < 0) {
		v2 = Point(centre.x + vB, centre.y + vA);
		v1 = Point(centre.x - vB, centre.y - vA);

		h2 = Point(centre.x - hB, centre.y + hA);
		h1 = Point(centre.x + hB, centre.y - hA);
	} else {
		v2 = Point(centre.x + vB, centre.y - vA);
		v1 = Point(centre.x - vB, centre.y + vA);

		h2 = Point(centre.x - hB, centre.y - hA);
		h1 = Point(centre.x + hB, centre.y + hA);
	}
}

Point getTop(Point p1, Point p2) {
	Point topPoint;
	if (lastTop == Point()) {
		lastTop = p2;
	}

	float a = abs(lastTop.x - p1.x);
	float b = abs(lastTop.y - p1.y);
	float c1 = sqrt(pow(a, 2) + pow(b, 2));

	a = abs(lastTop.x - p2.x);
	b = abs(lastTop.y - p2.y);
	float c2 = sqrt(pow(a, 2) + pow(b, 2));

	if (c1 < c2) {
		topPoint = p1;
	} else {
		topPoint = p2;
	}
	return topPoint;
}
