#ifndef _FACETOOL_H
#define _FACETOOL_H
#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "findEyeCenter.h"
#include <math.h>

#define PI 3.14159265f

using namespace std;
using namespace cv;

extern CascadeClassifier face, mouth, eye, nose;
extern Mat frame, grayframe, testframe;
extern Mat temp_rgb, temp_ycbcr, temp_hsv, temp_lab, temp_face;
extern Mat thres_ycbcr, thres_hsv, thres_lab;
extern Point pMouth1, pMouth2, pLeye1, pLeye2, pReye1, pReye2, pLeyec, pReyec, peyec, pMouthc;
extern vector<string> templist;

struct eyeInfo {
	Point topNode;
	Point botNode;
	Point cenNode;
	Point pupil;
	int type; //type: 1 means left eye, 2 means right eye.
	int size;
};

struct mouthInfo {
	Point topNode;
	Point botNode;
	Point centerNode;
	int size;
	int length;
};

struct noseInfo {
	Point topNode;
	Point botNode;
	Point centerNode;
	int size;
	int length;
};

class FaceTools {
public:
	//First jugement rule of face skin.
	bool R1(int R, int G, int B);
	//Second jugement rule of face skin.
	bool R2(float Y, float Cr, float Cb);
	//Third jugement rule of face skin.
	bool R3(float H, float S, float V);
	//Rule for bak.
	bool R1_bak(int R, int G, int B);
	//Rule given by Prof.Hanaizumi in 2015/12
	bool R4_Hana();
	//Show the skin detection result in binary chart.
	Mat GetSkin(Mat const &src, Mat &dst2);
	//
	Mat labThreshold(const Mat & frame);
	//Detect the facial features in RGB chart.
	int detectFacialFeatures(vector< Rect > faces);
	//Put the binary image onto the RGB image like a mask.
	Mat maskOnImage(Mat const &src, Mat const &src2);
	//Execute the template match procedure.
	Mat templateMatch(Mat &src, vector<string> templist, int match_method);
	//Detect face by skin
	int detectFaceSkin(Mat &src);
	//Detect face by skin from live video
	int detectFaceSkinInVideo(Mat &src);
	//Detect face by skin and detect corner points from live video
	Mat detectFaceCornerInVideo(Mat &src, Mat &pre);
	//Erase extra object in the image
	int processImage(Mat &src, Mat &dst);
	//Erase extra object in the image
	int getObjectSize(Mat &src, int type, int x, int y, int &size, int (&judge)[1000][1000]);
	//Erase hole size in the image
	int getHoleSize(Mat &src, int x, int y, int &size, int(&judge)[1000][1000]);
	//Copy object to the image
	int copyObject(Mat &src, Mat &dst, int x, int y);
	//Erase extra object in the image
	int eraseObject(Mat &src, int x, int y, int deep);
	//Scan the object
	int scanObject(Mat &src, int type, int x, int y, Point &p1);
	//Find face with object
	Point findFace(Mat &src, Mat &dst, Mat &result);
	//Find the mass center of the face, src is the result of skin detection
	int findMass(Mat &src);
	//Find face with object, this is a bak version
	int findFacialFeatures_20160128(Mat &src, Mat &dst, Mat &result);
	//Find face with object
	int findFacialFeatures(Mat &src, Mat &dst, Mat &result);
	//Scan the hole
	int scanHole(Mat &src, int type, int x, int y, Point &p1);
	////Get the sobel border of the input image
	Mat getSobelBorder(Mat src);
	//Get the horizontal projection of the image
	Mat getHorizontalProjection(Mat &src);
	//Get the vertical projection of the image
	Mat getVerticalProjection(Mat &src);
	//Transform the src image from rgb format to binary format
	Mat getBinaryFormat(Mat &src, int value);
	//Get the accurate position of eyes, threshold is the filter size
	Mat getExactEyes(Mat &src_binary, vector<eyeInfo> &eyeVec, int threshold);
	//Get the accurate position of mouth, threshold is the filter size
	Mat getExactMouth(Mat &src_binary, vector<mouthInfo> &mouVec, int threshold);
	//Get the general nose area
	Mat getNoseArea(Mat &src, vector<eyeInfo> &eyeVec, vector<mouthInfo> &mouVec, Point &border);
	//Get the accurate position of nose, threshold is the filter size
	Mat getExactNose(Mat &src_binary, vector<noseInfo> &noseVec, int threshold);
	//Get the accurate position of nose, threshold is the filter size
	Mat getExactNoseGradient(Mat &src_binary, vector<noseInfo> &noseVec, int threshold);
	//Move the nose position if only one nose hole was found
	int assumeNose(vector<noseInfo> &noseVec, Point eyeP, Point mouthP);
	//Move the nose position
	int moveNose(vector<noseInfo> &noseVec, int type);
	//Calculate the angle and display it on the screen
	int calculateFace(Mat &src, Mat &eyeBin, vector<eyeInfo> &eyeVec, vector<mouthInfo> &mouVec, Point noseCenter);
	//Change the mouth position if necessary
	int changeMouPosition(vector<mouthInfo> &mouVec, int x, int y);
	//Draw the line on the image to identify facial features
	void drawFacialFeatures(Mat &src, Mat &faceBin, vector<eyeInfo> &eyeVec, vector<mouthInfo> &mouVec, Point &noseCenter);
	//Avoid border out of bound for a mat
	int getBoundValue(Mat src, int range, int type);
	//Compute the gradient of a mat
	Mat computeMatGradient(const Mat &mat);
	//Get the final gradient of the mat
	Mat matrixMagnitude(const Mat &matX, const Mat &matY, double &maxMag, int type);
	//Mark the peak point in a gradient mat
	Mat findPeakPoint(const Mat &src, const Mat &grad, double threshold, int grayThres, Point &nosePoint);
	//Mark the peak point in a gradient mat for eyes
	Mat findPeakPointEyes(const Mat &src, const Mat &grad, double threshold, int grayThres, Point &nosePoint);
	//Mark the peak point in a gradient mat
	int findFaceInDB(char dbPath[256]);
	//Get the variance of the image
	float getVariance(const Mat &src, int start, int end);
	//Get circles in the image by using Hough
	Mat getHoughCircles(const Mat &src, int minRadius = 0, int maxRadius = 0);
	//Get contours in the image by using findContours
	Mat getContoursByCplus(const Mat &src, int mode = 0, double minarea = 0, double whRatio = 1);
	//Detect the eyes using gray level
	Mat getExactEyesGray(Mat &src, int threshold);
	//Detect corner points of the face
	Mat detectCornerPoints(Mat &src, Mat &dst, Point startP);
};

#endif