#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include "math_neon.hpp"

/*
#include "math_atan2f.cpp"
#include "math_sqrtf.cpp"
#include "math_powf.cpp"
#include "math_runfast.cpp"
*/
//#include <iostream> // Comment this out when not printing anything.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

using namespace cv;
using namespace std;

// Keeps track of all relevant data for one individual object being tracked.
struct TrackingObject {
	int minHSV, maxHSV;

	vector<Moments> moments;
	int contourIndex;
	Point2f centre;

	float angle, width, length;
	Point v1, v2, h1, h2;
	Point topPoint, lastTop;
};

// Used to send multiple variables to the mouse events and keep track of mouse state without the use of global variables.
// This is horribly messy and without a doubt the worse code in here, but due to the nature of the click events, I had to keep track of a lot of crap somewhere.
struct MouseParams {
	bool currentlyTracking;
	int trackingType;
	TrackingObject currentObject; // Current object being added.
	Point currentPos, downPos, upPos, properUpPos, lastDownPos, dblClickPos; // Keeping track of state, used when in selection mode.
	Point histogramPos;
	bool clickOccur;
};

// Getting object
void getNewObject(Mat& frame, int hueOffset, int position);
void getRangeFromSelection(Mat& histogramImg, int& hueValue, Mat image, int bins = 60, int scaleX = 12);
void drawTrackingOverlay(Mat& frame, int position);
void resetMouseData();
// Track object
void updateObject(TrackingObject& object, Mat originalFrame, int gaussianKernalSize = 3);
void findMoments(TrackingObject& obj, Mat thresholdFrame);
void runCalculations(TrackingObject& obj);
void getPoints(TrackingObject& obj);
void getTop(TrackingObject& obj);
// Drawing
void drawOverlay(TrackingObject& obj, Mat& frame);
void drawRotatedBox(TrackingObject obj, Mat& frame);
void drawDebugInformation(Mat frame, int id, TrackingObject obj);
void drawObjList(Mat& frame, map<int, TrackingObject> objs);
void drawHelp(Mat& frame);
// Commands
int processKey(int keyVal);
// Misc
void onMouse(int event, int x, int y, int, void*);
void onMouseHist(int event, int x, int y, int, void*);
string convertToString(float val);

// I tried using as little global variables as possible. I'm fine with the capture stuff being here.
// But if it weren't for that damn mouse callback I'd get rid of the stupid params variable.
int captureWidth, captureHeight;
MouseParams mouseData;

int main() {
	VideoCapture capture = VideoCapture(-1);
	if (!capture.isOpened()) {
		return -1;
	}
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	captureWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	captureHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	Mat frame;
	namedWindow("Output", CV_WINDOW_AUTOSIZE);
	// This gets reset later on to send more recent data
	setMouseCallback("Output", onMouse, 0);

	map<int, TrackingObject> objects;

	mouseData.trackingType = 2;

	// FPS
	time_t start, end;
	int counter = 0;
	double sec;
	double fps;

	// Start a loop to track objects and add new ones if necessary.
	int positionToInsert = -1;
	int debugIndex = -1;
	bool showHelp = false;
	bool showFPS = false;

	enable_runfast();

	while (true) {
		if (counter == 0) {
			time(&start);
		}
		capture.read(frame);

		if (positionToInsert != -1) {
			mouseData.currentlyTracking = true;
			getNewObject(frame, 4, positionToInsert);
			// If object has been completed, as signified by a value in non-zero max HSV value, then add it to the tracking list and reset mouseData.
			if (mouseData.currentObject.maxHSV != 0) {
				objects.insert(make_pair(positionToInsert, mouseData.currentObject));
				positionToInsert = -1;
				mouseData.currentObject = TrackingObject();
			}
		} else {
			destroyWindow("Histogram");
			resetMouseData();
		}

		if (objects.size() > 0) {
			for (map<int, TrackingObject>::iterator item = objects.begin(), end = objects.end(); item != end; ++item) {
				updateObject(item->second, frame);

				if ((item->second).moments.size() > 0) {
					runCalculations(item->second);

					drawOverlay((item->second), frame);
					(item->second).lastTop = (item->second).topPoint;
				}
				if (item->first == debugIndex) {
					drawDebugInformation(frame, item->first, item->second);
				}
			}
		}
		drawObjList(frame, objects);

		if (showHelp) {
			drawHelp(frame);
		}
		if (showFPS) {
			Point pt = Point((75.0 / 100.0) * captureWidth, (90.0 / 100.0) * captureHeight);
			putText(frame, "FPS: " + convertToString(fps), pt, FONT_HERSHEY_PLAIN, 1.0f, Scalar(0, 255, 0));
		}

		Point pt = Point((75.0 / 100.0) * captureWidth, (95.0 / 100.0) * captureHeight);
		putText(frame, "Press h for Help", pt, FONT_HERSHEY_PLAIN, 1.0f, Scalar(0, 255, 0));

		// Display image
		imshow("Output", frame);

		time(&end);
		counter++;
		sec = difftime(end, start);
		fps = counter / sec;
		// Overflow protection
		if (counter == (INT_MAX - 1000))
			counter = 0;

		int keyVal = processKey(waitKey(10));
		if (keyVal == -2) {
			return 0; //Quit
		} else if (keyVal == -4) {
			debugIndex++;
			if ((unsigned) debugIndex >= objects.size()) {
				debugIndex = -1;
			}
		} else if (keyVal == -5) {
			showHelp = !showHelp;
		} else if (keyVal == -6) {
			showFPS = !showFPS;
		} else if (keyVal > 0) {
			// Keyboards go 1-9, we don't want this, since arrays start at 0, so really, when you click 1,
			// you are removing item 0, or adding if that's what you are into.
			int id = keyVal - 1;
			if (id == positionToInsert) {
				positionToInsert = -1;
			} else if (objects.count(id)) { // Target exists so we remove it.
				objects.erase(id);
			} else { // doesn't exist so let's add it.
				positionToInsert = id;
			}
		}
	}

	waitKey(0);

	// Destroy windows
	destroyWindow("Histogram");
	destroyWindow("Output");
}

// Although this returns a new object eventually, we do not use a return type as it will not return something every frame.
void getNewObject(Mat& frame, int hueOffset, int position) {
	int hueValue;
	Mat histImg;

	int bins = 60;
	int histScaleX = 12;
	int pointRange = 15;

	if (mouseData.trackingType == 0) { // Selection

		// At this point nothing happens except the overlay being drawn, until we get the mouse call back gets the click events.
		if (mouseData.downPos != Point() && mouseData.properUpPos != Point()) {
			Rect area = Rect(mouseData.downPos.x, mouseData.downPos.y, abs(mouseData.upPos.x - mouseData.downPos.x),
					abs(mouseData.upPos.y - mouseData.downPos.y));
			getRangeFromSelection(histImg, hueValue, frame(area), bins, histScaleX);

			mouseData.currentObject.minHSV = hueValue - hueOffset;
			mouseData.currentObject.maxHSV = hueValue + hueOffset;
		}

	} else if (mouseData.trackingType == 1) { // Histogram

		// Waiting till we get a double click event.
		if (mouseData.clickOccur == true) {
			mouseData.clickOccur = false; // This should only fire once.
			getRangeFromSelection(histImg, hueValue, frame, bins, histScaleX);

			imshow("Histogram", histImg);
			setMouseCallback("Histogram", onMouseHist, 0);
		}
		if (mouseData.histogramPos != Point()) {
			int realX = mouseData.histogramPos.x / histScaleX;
			hueValue = realX * (180 / bins);

			mouseData.currentObject.minHSV = hueValue - hueOffset;
			mouseData.currentObject.maxHSV = hueValue + hueOffset;
		}

	} else if (mouseData.trackingType == 2) { // Point
		if (mouseData.clickOccur == true) {
			mouseData.clickOccur = false; // This should only fire once.
			int x = mouseData.dblClickPos.x, y = mouseData.dblClickPos.y;
			Rect area = Rect(x - pointRange, y - pointRange, pointRange * 2, pointRange * 2);
			getRangeFromSelection(histImg, hueValue, frame(area), bins, histScaleX);

			mouseData.currentObject.minHSV = hueValue - hueOffset;
			mouseData.currentObject.maxHSV = hueValue + hueOffset;
		}
	}

	// Make sure everything is within the correct ranges.
	if (mouseData.currentObject.minHSV > mouseData.currentObject.maxHSV) {
		int tempBound = mouseData.currentObject.minHSV;
		mouseData.currentObject.minHSV = mouseData.currentObject.maxHSV;
		mouseData.currentObject.maxHSV = tempBound;
	}

	if (mouseData.currentObject.minHSV < 0)
		mouseData.currentObject.minHSV = 0;
	if (mouseData.currentObject.maxHSV > 180)
		mouseData.currentObject.maxHSV = 180;

	// Draw overlay for tracking
	drawTrackingOverlay(frame, position);
}

void getRangeFromSelection(Mat& histogramImg, int& hueValue, Mat image, int bins, int scaleX) {
	Mat imageHSV;

	// Convert to HSV
	cvtColor(image, imageHSV, CV_RGB2HSV);

	Mat hist;
	int channels[] = { 0 };
	int histSize[] = { bins };
	float hueRange[] = { 0, 180 };
	const float* ranges[] = { hueRange };
	calcHist(&imageHSV, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);

	double maxValue = 0;
	int maxValueLocation = 0;
	minMaxLoc(hist, 0, &maxValue, 0, &maxValueLocation);
	double highestBinVal = hist.at<int>(maxValueLocation);

	int scaleY = maxValue * 0.006;
	scaleY = scaleY == 0 ? 1 : scaleY;
	int padding = 25;
	// Add 40 for some padding at the top.
	histogramImg = Mat::zeros((maxValue / scaleY) + padding, bins * scaleX, CV_8UC3);

	for (int h = 0; h < bins; h++) {
		double binVal = hist.at<int>(h);
		int value = (binVal / highestBinVal) * maxValue;

		int hueValue = (h * (180 / bins));
		rectangle(histogramImg, Point(h * scaleX, (maxValue / scaleY) + padding),
				Point((h + 1) * scaleX, (maxValue - value) / scaleY + padding), Scalar(hueValue, 255, 255), CV_FILLED);
	}

	hueValue = (maxValueLocation * (180 / bins));

	cvtColor(histogramImg, histogramImg, CV_HSV2RGB);
}

void drawTrackingOverlay(Mat& frame, int i) {
	string str = "TRACKING for ";
	str.append(convertToString(i + 1));
	str.append(": Using ");
	switch (mouseData.trackingType) {
	case 0:
		str.append("SELECTION");
		break;
	case 1:
		str.append("HISTOGRAM");
		break;
	case 2:
		str.append("POINT");
		break;
	}
	Point pt = Point((2.0 / 100.0) * captureWidth, (5.0 / 100.0) * captureHeight);
	putText(frame, str, pt, FONT_HERSHEY_PLAIN, 1.0f, Scalar(0, 255, 0));

	rectangle(frame, mouseData.downPos, mouseData.upPos, Scalar(255, 0, 0), 1, 8, 0);
}

void resetMouseData() {
	mouseData.currentlyTracking = false;
	mouseData.currentPos = Point();
	mouseData.downPos = Point();
	mouseData.upPos = Point();
	mouseData.properUpPos = Point();
	mouseData.lastDownPos = Point();
	mouseData.dblClickPos = Point();
	mouseData.histogramPos = Point();
	mouseData.clickOccur = false;
}

void updateObject(TrackingObject& object, Mat originalFrame, int gaussianKernalSize) {
	Mat imageHSV, channelsHSV[3], thresholdHSV;

	// Convert to gray
	cvtColor(originalFrame, imageHSV, CV_RGB2HSV);

	// Get the HUE channel
	split(imageHSV, channelsHSV);

	// Threshold HSV image
	inRange(channelsHSV[0], Scalar(object.minHSV, 0, 0), Scalar(object.maxHSV, 0, 0), thresholdHSV);

	// Erosion
	erode(thresholdHSV, thresholdHSV, Mat(), Point(-1, -1), 3, 1, 1);

	// Gaussian filter to smooth the H frame
	GaussianBlur(thresholdHSV, thresholdHSV, Size(gaussianKernalSize, gaussianKernalSize), 0, 0);

	// Dilation
	dilate(thresholdHSV, thresholdHSV, Mat(), Point(-1, -1), 9, 1, 1);

	// Get moments
	findMoments(object, thresholdHSV);
}

void findMoments(TrackingObject& obj, Mat thresholdFrame) {
	double largestArea = 0, newArea = 0;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(thresholdFrame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > contoursPoly(contours.size());
	vector<Moments> mu(contours.size());

	for (std::size_t i = 0, max = contours.size(); i < max; i++) {
		mu[i] = moments((Mat) contours[i], false); // get the moment of each contours
		newArea = contourArea((Mat) contours[i]); //  Find the area of contour

		if (newArea > largestArea) {
			largestArea = newArea; // update the largest area

			approxPolyDP(Mat(contours[i]), contoursPoly[i], 3, true);
			// find the centre of largest contour
			obj.centre = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);

			obj.contourIndex = i; // store the index of largest contour
			obj.moments = mu;
		}
	}
}

void runCalculations(TrackingObject& obj) {
	Moments m = obj.moments[obj.contourIndex];
	float a = (m.m20 / m.m00) - powf_neon_sfp(obj.centre.x, 2);
	float b = 2 * ((m.m11 / m.m00) - (obj.centre.x * obj.centre.y));
	float c = (m.m02 / m.m00) - powf_neon_sfp(obj.centre.y, 2);

	obj.length = sqrtf_neon_sfp(((a + c) + (sqrtf_neon_sfp(powf_neon_sfp(b, 2) + powf_neon_sfp((a - c), 2)))) * 0.5);
	obj.width = sqrtf_neon_sfp(((a + c) - (sqrtf_neon_sfp(powf_neon_sfp(b, 2) + powf_neon_sfp((a - c), 2)))) * 0.5);

	obj.angle = (atan2(b, a - c)) * 0.5;

	getTop(obj);

	getPoints(obj);
}

void getPoints(TrackingObject& obj) {
	float innerAngle = (0.5 * M_PI) - abs(obj.angle);

	float vC = obj.width;
	float vA = (vC * sin(innerAngle));
	float vB = sqrtf_neon_sfp(powf_neon_sfp(vC, 2) - powf_neon_sfp(vA, 2));

	float hC = obj.length;
	float hA = (hC * sin(abs(obj.angle)));
	float hB = sqrtf_neon_sfp(powf_neon_sfp(hC, 2) - powf_neon_sfp(hA, 2));

	vB = vB * 2;
	vA = vA * 2;
	hB = hB * 2;
	hA = hA * 2;

	if (obj.angle < 0) {
		obj.v2 = Point(obj.centre.x + vB, obj.centre.y + vA);
		obj.v1 = Point(obj.centre.x - vB, obj.centre.y - vA);

		obj.h2 = Point(obj.centre.x - hB, obj.centre.y + hA);
		obj.h1 = Point(obj.centre.x + hB, obj.centre.y - hA);
	} else {
		obj.v2 = Point(obj.centre.x + vB, obj.centre.y - vA);
		obj.v1 = Point(obj.centre.x - vB, obj.centre.y + vA);

		obj.h2 = Point(obj.centre.x - hB, obj.centre.y - hA);
		obj.h1 = Point(obj.centre.x + hB, obj.centre.y + hA);
	}
}

void getTop(TrackingObject& obj) {
	if (obj.lastTop == Point()) {
		obj.lastTop = obj.h2;
	}

	float a = abs(obj.lastTop.x - obj.h1.x);
	float b = abs(obj.lastTop.y - obj.h1.y);
	float c1 = sqrtf_neon_sfp(powf_neon_sfp(a, 2) + powf_neon_sfp(b, 2));

	a = abs(obj.lastTop.x - obj.h2.x);
	b = abs(obj.lastTop.y - obj.h2.y);
	float c2 = sqrtf_neon_sfp(powf_neon_sfp(a, 2) + powf_neon_sfp(b, 2));

	if (c1 < c2) {
		obj.topPoint = obj.h1;
	} else {
		obj.topPoint = obj.h2;
	}
}

void drawOverlay(TrackingObject& obj, Mat& frame) {
	float angleDeg = (obj.angle / M_PI) * 180;
	float angleDegFixed = angleDeg;
	if (angleDeg < 0) {
		angleDegFixed = 360 + angleDeg;
	}

	RotatedRect rect = RotatedRect(obj.centre, Size2f(obj.length * 4, obj.width * 4), angleDegFixed);
	ellipse(frame, rect, Scalar(0, 0, 255), 2, 8);

	line(frame, obj.h1, obj.h2, Scalar(0, 255, 0));
	line(frame, obj.v1, obj.v2, Scalar(0, 255, 0));

	circle(frame, obj.topPoint, 5, Scalar(8, 240, 240));

	drawRotatedBox(obj, frame);
}

void drawRotatedBox(TrackingObject obj, Mat& frame) {
	float a = abs(obj.h2.x - obj.centre.x);
	float b = abs(obj.h2.y - obj.centre.y);

	Point c1, c2, c3, c4;
	if (obj.angle < 0) {
		c1 = Point(obj.v1.x - a, obj.v1.y + b);
		c2 = Point(obj.v1.x + a, obj.v1.y - b);
		c3 = Point(obj.v2.x - a, obj.v2.y + b);
		c4 = Point(obj.v2.x + a, obj.v2.y - b);
	} else {
		c1 = Point(obj.v1.x - a, obj.v1.y - b);
		c2 = Point(obj.v1.x + a, obj.v1.y + b);
		c3 = Point(obj.v2.x - a, obj.v2.y - b);
		c4 = Point(obj.v2.x + a, obj.v2.y + b);
	}

	line(frame, c1, c2, Scalar(0, 0, 255), 2);
	line(frame, c2, c4, Scalar(0, 0, 255), 2);
	line(frame, c4, c3, Scalar(0, 0, 255), 2);
	line(frame, c3, c1, Scalar(0, 0, 255), 2);
}

// This is somewhat messy, because I need to print each thing and to do that requires 4 lines.
void drawDebugInformation(Mat frame, int id, TrackingObject obj) {
	double xPercent = 78;

	Point pt = Point((xPercent / 100.0) * captureWidth, (5.0 / 100.0) * captureHeight);
	string str = "DEBUG: ";
	str.append(convertToString(id + 1));
	putText(frame, str, pt, FONT_HERSHEY_PLAIN, 1.0f, Scalar(0, 255, 0));

	pt = Point((xPercent / 100.0) * captureWidth, (10.0 / 100.0) * captureHeight);
	str = "Angle: ";
	str.append(convertToString((obj.angle / M_PI) * 180));
	putText(frame, str, pt, FONT_HERSHEY_PLAIN, 1.0f, Scalar(0, 255, 0));

	pt = Point((xPercent / 100.0) * captureWidth, (15.0 / 100.0) * captureHeight);
	str = "Length: ";
	str.append(convertToString(obj.length));
	putText(frame, str, pt, FONT_HERSHEY_PLAIN, 1.0f, Scalar(0, 255, 0));

	pt = Point((xPercent / 100.0) * captureWidth, (20.0 / 100.0) * captureHeight);
	str = "Width: ";
	str.append(convertToString(obj.width));
	putText(frame, str, pt, FONT_HERSHEY_PLAIN, 1.0f, Scalar(0, 255, 0));

	pt = Point((xPercent / 100.0) * captureWidth, (25.0 / 100.0) * captureHeight);
	str = "Min H: ";
	str.append(convertToString(obj.minHSV));
	putText(frame, str, pt, FONT_HERSHEY_PLAIN, 1.0f, Scalar(0, 255, 0));

	pt = Point((xPercent / 100.0) * captureWidth, (30.0 / 100.0) * captureHeight);
	str = "Max H: ";
	str.append(convertToString(obj.maxHSV));
	putText(frame, str, pt, FONT_HERSHEY_PLAIN, 1.0f, Scalar(0, 255, 0));
}

void drawObjList(Mat& frame, map<int, TrackingObject> objs) {
	string output = "CURRENT:";
	for (int i = 0; i < 9; i++) {
		if (objs.count(i)) {
			output.append(" ");
			output.append(convertToString(i + 1));
		} else {
			output.append(" -");
		}
	}
	Point pt = Point((2.0 / 100.0) * captureWidth, (95.0 / 100.0) * captureHeight);
	putText(frame, output, pt, FONT_HERSHEY_PLAIN, 1.0f, Scalar(0, 255, 0));
}

void drawHelp(Mat& frame) {
	vector<string> lines;
	lines.push_back("HELP:");
	lines.push_back("q    Quit");
	lines.push_back("h    Toggle Help");
	lines.push_back("s    Switch Selection Mode");
	lines.push_back("d    Toggle Debug");
	lines.push_back("f    Toggle FPS");
	lines.push_back("");
	lines.push_back("1-9    If target exists: remove target.");
	lines.push_back("        If target does not exist: create target.");
	double yPercent = 35;
	double yStep = 5;

	for (int i = 0; (unsigned) i < lines.size(); i++) {
		string str = lines[i];
		Point pt = Point((20.0 / 100.0) * captureWidth, (yPercent / 100.0) * captureHeight);
		putText(frame, str, pt, FONT_HERSHEY_PLAIN, 1.0f, Scalar(0, 255, 0));
		yPercent += yStep;
	}
}

// This returns a negative value when the key pressed is used for a command, like switch tracking type or quit, if it's a number however
// that number is parroted back so that we can add a tracking object at that index or remove one if it already exists.
int processKey(int keyVal) {
	// No key pressed.
	if (keyVal == -1)
		return -1;
	// Key pressed
	if (keyVal == 1048689) // Q
		return -2;
	else if (keyVal == 1048691) { // S
		mouseData.trackingType += 1;
		if (mouseData.trackingType > 2)
			mouseData.trackingType = 0;
		return -3;
	} else if (keyVal == 1048680) { // H
		return -5;
	} else if (keyVal == 1048676) { // D
		return -4;
	} else if (keyVal == 1048678) {
		return -6;
	} else if (keyVal == 1048625) // 1
		return 1;
	else if (keyVal == 1048626) // 2
		return 2;
	else if (keyVal == 1048627) // 3
		return 3;
	else if (keyVal == 1048628) // 4
		return 4;
	else if (keyVal == 1048629) // 5
		return 5;
	else if (keyVal == 1048630) // 6
		return 6;
	else if (keyVal == 1048631) // 7
		return 7;
	else if (keyVal == 1048632) // 8
		return 8;
	else if (keyVal == 1048633) // 9
		return 9;
// Key that I don't care about, pressed.
	return 0;
}

void onMouse(int event, int x, int y, int, void*) {
	if (mouseData.currentlyTracking == true) {
		mouseData.currentPos = Point(x, y);
		if (mouseData.trackingType == 0) {
			if (mouseData.downPos != mouseData.lastDownPos) {
				mouseData.upPos = mouseData.currentPos;
			}

			if (event == CV_EVENT_LBUTTONDOWN)
				mouseData.downPos = mouseData.currentPos;
			if (event == CV_EVENT_LBUTTONUP) {
				mouseData.upPos = mouseData.currentPos;
				mouseData.properUpPos = mouseData.currentPos;
				mouseData.lastDownPos = mouseData.downPos;
			}
		} else if (mouseData.trackingType == 1) {
			if (event == CV_EVENT_LBUTTONDBLCLK) {
				mouseData.clickOccur = true;
				mouseData.dblClickPos = mouseData.currentPos;
			}
		} else if (mouseData.trackingType == 2) {
			if (event == CV_EVENT_LBUTTONDBLCLK) {
				mouseData.clickOccur = true;
				mouseData.dblClickPos = mouseData.currentPos;
			}
		}
	}
}

void onMouseHist(int event, int x, int y, int, void*) {
	if (mouseData.currentlyTracking == true) {
		if (event == CV_EVENT_LBUTTONDOWN) {
			mouseData.histogramPos = Point(x, y);
		}
	}
}

string convertToString(float val) {
	ostringstream convert;
	convert << val;
	return convert.str();
}
