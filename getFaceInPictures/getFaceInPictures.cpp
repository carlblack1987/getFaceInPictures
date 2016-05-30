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
		Mat pre;
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
				//facetool.detectFaceSkinInVideo(frame);
				//Added 20160517, use calcOpticalFlowPyrLK to get the graphic change per frame.
				pre = facetool.detectFaceCornerInVideo(frame, pre);
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

		//Atsushi Sann's pictures
		//Right in front
		//Mat frame = imread(picFilePath + "20160301133624 13.jpg");

		//Using face Database's pictures
		//Person06 not ok
		//Mat frame = imread(picFilePath + "Person06\\person06120-30+0.jpg");
		//
		//Mat frame = imread(picFilePath + "Person06\\person06246+0+0.jpg");
		//Person 03 binaryThres = 70 ok
		//Mat frame = imread(picFilePath + "Person03\\person03146+0+0.jpg");
		//Person 03 Right 30 degree Right eye not ok
		//Mat frame = imread(picFilePath + "Person03\\person03144+0-30.jpg");
		//Person 03 Right Up 30 degree
		//Mat frame = imread(picFilePath + "Person03\\person03270+30-30.jpg");
		//Person 03 Right Down 30 degree not ok
		//Mat frame = imread(picFilePath + "Person03\\person03218-30-30.jpg");
		//Person 03 Left Up 30 degree left eye not ok
		//Mat frame = imread(picFilePath + "Person03\\person03274+30+30.jpg");
		//Person 03 Left Down 30 degree left eye not ok
		//Mat frame = imread(picFilePath + "Person03\\person03222-30+30.jpg");
		//Person 10 binaryThres = 60 ok
		//Mat frame = imread(picFilePath + "Person10\\person10146+0+0.jpg");
		//Person 10 Right 30 degree ok
		//Mat frame = imread(picFilePath + "Person10\\person10131-15-30.jpg");
		//Person 10 Right Up 30 degree mouth not ok
		//Mat frame = imread(picFilePath + "Person10\\person10170+30-30.jpg");
		//Person 10 Right Down 30 degree ok
		//Mat frame = imread(picFilePath + "Person10\\person10118-30-30.jpg");
		//Person 10 Left Up 30 degree left eye not ok
		//Mat frame = imread(picFilePath + "Person10\\person10161+15+30.jpg");
		//Person 10 Left Down 30 degree left eye not ok
		//Mat frame = imread(picFilePath + "Person10\\person10122-30+30.jpg");
		//Person 14 binaryThres = 60 ok
		//Mat frame = imread(picFilePath + "Person14\\person14146+0+0.jpg");
		//Person 14 Right 30 degree ok
		//Mat frame = imread(picFilePath + "Person14\\person14131-15-30.jpg");
		//Person 14 Right Up 30 degree mouth not ok
		//Mat frame = imread(picFilePath + "Person14\\person14170+30-30.jpg");
		//Person 14 Right Down 30 degree ok
		//Mat frame = imread(picFilePath + "Person14\\person10118-30-30.jpg");
		//Person 14 Left Up 30 degree binary = 80 ok
		//Mat frame = imread(picFilePath + "Person14\\person14174+30+30.jpg");
		//Person 14 Left Down 30 degree binary = 90 mouth not ok
		//Mat frame = imread(picFilePath + "Person14\\person14122-30+30.jpg");

		//New Images from 20160418
		//Mat frame = imread(picFilePath + "20160418160120 01.jpg");
		//Right 30 degree gradient ok
		//Mat frame = imread(picFilePath + "20160418160120 07.jpg");
		//Right Up 30 degree gradient ok
		//Mat frame = imread(picFilePath + "20160418160120 13.jpg");
		//Right Down 30 degree gradient ok
		//Mat frame = imread(picFilePath + "20160418160120 19.jpg");
		//Left 10 degree gradient ok
		//Mat frame = imread(picFilePath + "20160418160120 32.jpg");
		//Left 15 degree gradient ok
		//Mat frame = imread(picFilePath + "20160418160120 37.jpg");
		//Left 20
		//Mat frame = imread(picFilePath + "20160418160120 48.jpg");
		//Left 30 degree gradient ok
		//Mat frame = imread(picFilePath + "20160418160120 39.jpg");
		//Left Down 30 degree
		//Mat frame = imread(picFilePath + "20160418160120 46.jpg");
		//Left Up 30 degree
		//Mat frame = imread(picFilePath + "20160418160120 43.jpg");

		//New Images from 20160530
		//Right in front refine not ok
		Mat frame = imread(picFilePath + "20160530133431 04.jpg");
		//Right 15 degrees
		//Mat frame = imread(picFilePath + "20160530133431 15.jpg");
		//Right 15 degrees
		//Mat frame = imread(picFilePath + "20160530133431 15.jpg");
		//Right 15 degrees
		//Mat frame = imread(picFilePath + "20160530133431 15.jpg");
		//Right 15 degrees
		//Mat frame = imread(picFilePath + "20160530133431 15.jpg");
		//Right 15 degrees
		//Mat frame = imread(picFilePath + "20160530133431 15.jpg");
		//Right 15 degrees
		//Mat frame = imread(picFilePath + "20160530133431 15.jpg");



		//With glasses:
		//Left Up 30 degree
		//Mat frame = imread(picFilePath + "20160529154711 05.jpg");

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