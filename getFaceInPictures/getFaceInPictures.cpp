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
string picFilePath = "E:\\faceTemplate\\screenShotOutput\\sasaki\\";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\yoshida\\";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\yamazaki\\";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\yamazaki_w\\";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\fan\\";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\border\\";
//string picFilePath = "E:\\faceTemplate\\screenShotOutput\\Saitou\\";
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
		//Right in front ok
		//Mat frame = imread(picFilePath + "20160601190750 009.jpg");
		//Right 15 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 028.jpg");
		//Right 30 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 030.jpg");
		//Right 60 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 033.jpg");
		//Left 15 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 048.jpg");
		//Left 30 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 050.jpg");
		//Left 60 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 053.jpg");
		//Right 30 Up 10 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 138.jpg");
		//Right 10 LeftYaw 10 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 400.jpg");
		//Right 30 LeftYaw 10 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 423.jpg");
		//Right 0 LeftYaw 15 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 501.jpg");
		//Right 0 LeftYaw 25 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 502.jpg");
		//Right 0 RightYaw 15 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 573.jpg");
		//Right 30 RightYaw 20 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 623.jpg");
		//RightYaw 30 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 669.jpg");
		//Right 30 RightYaw 25 degrees ok
		//Mat frame = imread(picFilePath + "20160601190750 674.jpg");
		//Right 50 RightYaw 25 degrees
		//Mat frame = imread(picFilePath + "20160601190750 676.jpg");


		//Sasaki Man
		//Normal Problem: neck is included
		//Mat frame = imread(picFilePath + "20160614191358 0091.jpg");
		//Normal Problem: neck is included
		//Mat frame = imread(picFilePath + "20160614191358 0245.jpg");

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
		//Mat frame = imread(picFilePath + "20160614191547 974.jpg");

		//Yamazaki
		//Normal bad Problem: clothes's color close to skin
		//Mat frame = imread(picFilePath + "20160614192038 067.jpg");
		//Left
		//Mat frame = imread(picFilePath + "20160614192038 190.jpg");

		//Yamazaki Woman
		//Normal no
		//Mat frame = imread(picFilePath + "20160620173336 02.jpg");
		//Left 10 ok
		//Mat frame = imread(picFilePath + "20160620173336 06.jpg");
		//Left 15 no smile ruin the result
		//Mat frame = imread(picFilePath + "20160620173336 33.jpg");
		//Right 20 down 15 degrees not ok
		//Mat frame = imread(picFilePath + "20160620173336 46.jpg");
		//
		//Mat frame = imread(picFilePath + "20160620173336 06.jpg");

		//Distraction Evaluation
		//Normal no ok
		//Mat frame = imread(picFilePath + "20160621152911 041.jpg");
		//Normal ok
		//Mat frame = imread(picFilePath + "20160621152911 075.jpg");
		//Normal bad skin detection failed
		//Mat frame = imread(picFilePath + "20160621152911 082.jpg");
		//Normal ok
		//Mat frame = imread(picFilePath + "20160621152911 114.jpg");
		//Normal ok
		//Mat frame = imread(picFilePath + "20160621152911 118.jpg");
		//Normal ok
		//Mat frame = imread(picFilePath + "20160621152911 120.jpg");
		//Normal ok
		//Mat frame = imread(picFilePath + "20160621152911 140.jpg");
		//Normal ok
		//Mat frame = imread(picFilePath + "20160621152911 153.jpg");
		//Normal ok
		//Mat frame = imread(picFilePath + "20160621152911 158.jpg");
		//Normal ok
		//Mat frame = imread(picFilePath + "20160621152911 164.jpg");
		//Normal ok
		//Mat frame = imread(picFilePath + "20160621152911 172.jpg");
		//Yoshida ok
		//Mat frame = imread(picFilePath + "20160614191547 287.jpg");
		//Yoshida ok
		//Mat frame = imread(picFilePath + "20160614191547 571.jpg");
		//Sasaki ok
		//Mat frame = imread(picFilePath + "20160614191358 0261.jpg");
		//Sasaki ok
		//Mat frame = imread(picFilePath + "20160614191358 0327.jpg");
		//Sasaki ok
		//Mat frame = imread(picFilePath + "20160614191358 0148.jpg");
		//Sasaki
		Mat frame = imread(picFilePath + "20160614191358 0327.jpg");
		//Saitou
		//Mat frame = imread(picFilePath + "20160625165602 224.jpg");


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