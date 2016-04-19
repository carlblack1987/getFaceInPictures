// getFaceInPictures.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay(Mat frame);

CascadeClassifier face_cascade;
string opencvLibPath = "E:\\opencv\\sources\\data\\haarcascades\\";
string face_cascade_name = opencvLibPath + "haarcascade_frontalface_alt.xml";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\20151125141830\\";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\20160122115308\\";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\20160220122305\\";
string picFilePath = "E:\\faceTemplate\\screenShotOutput\\20160418160120\\";
//string picFilePath = "E:\\faceTemplate\\HeadPoseImageDatabase\\";
string window_name = "Capture - Face detection";
extern string outputpath;
int filenumber; // Number of file to be saved
string filename;
//Path of the face database files
char dbPath[256] = "E:/faceTemplate/HeadPoseImageDatabase/";

// Global variables
// Copy this file from opencv/data/haarscascades to target folder

// Function main
int main(void)
{
	cout << "Please enter 1 to use video mode or 2 to use picture mode." << endl;
	FaceTools facetool;
	int openMode;
	//1 means videomode, 2 means picture mode, 3 means using face database
	cin >> openMode;

	if (openMode == 1) {
		VideoCapture cap(0);
		cap.open(0);

		while (1)
		{
			cap.read(frame);
			cap >> frame;
			if (!cap.read(frame))
			{
				printf("an error while taking the frame from cap");
			}
			else {
				facetool.detectFaceSkinInVideo(frame);
				if (cvWaitKey(33) == 27)
					return 0;
			}
		}
	}
	else if (openMode == 2) {
		//New border image
		//Right low ok 0.39
		//Mat frame = imread(picFilePath + "20160122115308 044.jpg");
		//Left low ok 0.57
		//Mat frame = imread(picFilePath + "20160122115308 082.jpg");
		//Right high ok 0.30
		//Mat frame = imread(picFilePath + "20160122115308 264.jpg");
		//Left high ok 0.31
		//Mat frame = imread(picFilePath + "20160122115308 307.jpg");

		// Normal 0.03 2 degrees 0.100933
		//Mat frame = imread(picFilePath + "20160220122305 28.jpg");
		//right 001 0.19 10 degrees 0.19532
		//Mat frame = imread(picFilePath + "20160220122305 29.jpg");
		//right 002 0.24 20 degrees 0.270496
		//Mat frame = imread(picFilePath + "20160220122305 30.jpg");
		//right 003 0.39 33 degrees 0.440347
		//Mat frame = imread(picFilePath + "20160220122305 31.jpg");
		//right 004 0.68 50 degrees 0.74404
		//Mat frame = imread(picFilePath + "20160220122305 32.jpg");
		//right 005 0.78 60 degrees 0.848039
		//Mat frame = imread(picFilePath + "20160220122305 34.jpg");
		//left 001 0.033 5 degrees 0.0332
		//Mat frame = imread(picFilePath + "20160220122305 42.jpg");
		//left 002 0.41 38 degrees 0.535377
		//Mat frame = imread(picFilePath + "20160220122305 43.jpg");
		//left 003 0.57 45 degrees 0.725759
		//Mat frame = imread(picFilePath + "20160220122305 44.jpg");
		//left 004 0.81 65 degrees 0.879749
		//Mat frame = imread(picFilePath + "20160220122305 46.jpg");
		//left 005 1.05 70 degrees 1.09199
		//Mat frame = imread(picFilePath + "20160220122305 48.jpg");
		//left 006
		//Mat frame = imread(picFilePath + "20160220122305 50.jpg");

		//Atsushi Sann no pictures
		//Mat frame = imread(picFilePath + "20160301133624 14.jpg");

		//Using face Database's pictures
		//Person06 not ok
		//Mat frame = imread(picFilePath + "Person06\\person06120-30+0.jpg");
		//
		//Mat frame = imread(picFilePath + "Person06\\person06246+0+0.jpg");

		//New Images from 20160418
		//Mat frame = imread(picFilePath + "20160418160120 01.jpg");
		//Right 30 degree
		Mat frame = imread(picFilePath + "20160418160120 07.jpg");

		if (!frame.empty()){
			facetool.detectFaceSkin(frame);
		}
		else{
			printf(" --(!) No captured frame -- Break!");
			return 0;
		}

		int c = waitKey(10);

		if (27 == char(c)){
			return 0;
		}
	}
	else if (openMode == 3) {
		facetool.findFaceInDB(dbPath);
	}
	else {
		cout << "Unkown Operation" << endl;
	}

	return 0;
}