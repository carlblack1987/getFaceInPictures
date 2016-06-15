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
//string opencvLibPath = "E:\\opencv\\sources\\data\\haarcascades\\";
//string face_cascade_name = opencvLibPath + "haarcascade_frontalface_alt.xml";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\20151125141830\\";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\20160122115308\\";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\20160220122305\\";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\20160418160120\\";
//string picFilePath = "E:\\faceTemplate\\HeadPoseImageDatabase\\";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\sasaki\\";
string picFilePath = "E:\\faceTemplate\\screenShotOutput\\yoshida\\";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\yamazaki\\";
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

		//Fan Image
		//Right in front refine not ok
		//Mat frame = imread(picFilePath + "20160530133431 04.jpg");
		//Right 15 degrees
		//Mat frame = imread(picFilePath + "20160530133431 15.jpg");
		//Right 15 degrees
		//Mat frame = imread(picFilePath + "20160614183220 13.jpg");
		//Right 15 degrees
		//Mat frame = imread(picFilePath + "20160530133431 15.jpg");
		//Right 75 down 15 degrees
		//Mat frame = imread(picFilePath + "20160530133431 23.jpg");
		//Right 75 up 15 degrees not ok
		//Mat frame = imread(picFilePath + "20160530133431 26.jpg");
		//Right 75 degrees ok
		//Mat frame = imread(picFilePath + "20160530133431 21.jpg");

		//Sasaki
		//Normal Problem: neck is included
		//Mat frame = imread(picFilePath + "20160614191358 0091.jpg");

		//Yoshida
		//Normal_1 OK
		//Mat frame = imread(picFilePath + "20160614191547 070.jpg");
		//Normal_2 OK
		//Mat frame = imread(picFilePath + "20160614191547 087.jpg");
		//Left_1 OK
		//Mat frame = imread(picFilePath + "20160614191547 129.jpg");
		//Left_2 bad Problem: left eye not detected
		//Mat frame = imread(picFilePath + "20160614191547 143.jpg");
		//Right_1 ok
		//Mat frame = imread(picFilePath + "20160614191547 284.jpg");
		//Right_2 bad Problem: Right eye not detected
		//Mat frame = imread(picFilePath + "20160614191547 305.jpg");
		//RightUp_1 bad Problem: Wrong detection on mouth
		//Mat frame = imread(picFilePath + "20160614191547 773.jpg");
		//Right_Right_1 bad Problem: Wrong detection on left eye
		//Mat frame = imread(picFilePath + "20160614191547 814.jpg");
		//Right_Right_2 bad Problem: Wrong detection on left eye
		//Mat frame = imread(picFilePath + "20160614191547 828.jpg");
		//Left_Left_1 bad Problem: Left eye not detected
		//Mat frame = imread(picFilePath + "20160614191547 920.jpg");
		//Left_Left_2 bad Problem: Left eye not detected
		//Mat frame = imread(picFilePath + "20160614191547 928.jpg");
		//Normal_3 OK
		Mat frame = imread(picFilePath + "20160614191547 974.jpg");

		//Yamazaki
		//Normal bad Problem: clothes's color close to skin
		//Mat frame = imread(picFilePath + "20160614192038 067.jpg");
		//Left
		//Mat frame = imread(picFilePath + "20160614192038 190.jpg");

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