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
string picFilePath = "E:\\faceTemplate\\screenShotOutput\\20160220122305\\";
string window_name = "Capture - Face detection";
extern string outputpath;
int filenumber; // Number of file to be saved
string filename;

// Global variables
// Copy this file from opencv/data/haarscascades to target folder

// Function main
int main(void)
{
	cout << "Please enter 1 to use video mode or 2 to use picture mode." << endl;
	FaceTools facetool;
	int openMode;
	//1 means videomode, 2 means picture mode
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
		// Load the cascade
		if (!face_cascade.load(face_cascade_name)){
			printf("--(!)Error loading\n");
			return (-1);
		}

		// Read the image file
		//Mat frame = imread(picFilePath + "20151125141830 00002.jpg");
		//Eye closed status
		//Mat frame = imread(picFilePath + "20151125141830 00012.jpg");

		//Mat frame = imread(picFilePath + "20151125141830 19478.jpg");

		// Inclining Status with glasses
		//Mat frame = imread(picFilePath + "201512140935351_incline.jpg");
		//Mat frame = imread(picFilePath + "201512140935428_incline.jpg");

		// Inclining Status without glasses
		//Mat frame = imread(picFilePath + "201601111035190_incline.jpg");
		//Mat frame = imread(picFilePath + "201601111035257_incline.jpg");

		//New border image
		//Right low ok
		//Mat frame = imread(picFilePath + "20160122115308 044.jpg");
		//Left low ok
		//Mat frame = imread(picFilePath + "20160122115308 082.jpg");
		//Right high
		//Mat frame = imread(picFilePath + "20160122115308 264.jpg");
		//Left high
		//Mat frame = imread(picFilePath + "20160122115308 307.jpg");

		// Normal 0.03
		//Mat frame = imread(picFilePath + "20160220122305 28.jpg");
		//right 001 0.19
		//Mat frame = imread(picFilePath + "20160220122305 29.jpg");
		//right 002 
		//Mat frame = imread(picFilePath + "20160220122305 30.jpg");
		//right 003
		//Mat frame = imread(picFilePath + "20160220122305 31.jpg");
		//right 004
		//Mat frame = imread(picFilePath + "20160220122305 32.jpg");
		//right 005
		//Mat frame = imread(picFilePath + "20160220122305 34.jpg");
		//left 001 0.03
		//Mat frame = imread(picFilePath + "20160220122305 42.jpg");
		//left 002 0.41
		//Mat frame = imread(picFilePath + "20160220122305 43.jpg");
		//left 003 0.57
		Mat frame = imread(picFilePath + "20160220122305 44.jpg");
		//left 004 0.81
		//Mat frame = imread(picFilePath + "20160220122305 46.jpg");
		//left 005 1.05
		//Mat frame = imread(picFilePath + "20160220122305 48.jpg");
		//left 006
		//Mat frame = imread(picFilePath + "20160220122305 50.jpg");

		//Atsushi Sann no pictures
		//Mat frame = imread(picFilePath + "20160301133624 14.jpg");

		vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_PXM_BINARY);
		compression_params.push_back(1);

		/*
		//This four lines is to use the new skin detection rule, not finish yet.
		imwrite(outputpath + "1_Temp.pgm", frame, compression_params);

		Mat frame2 = imread(picFilePath + "outPut.pgm", CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Test", frame2);
		waitKey();
		*/

		//for (int i = 0; i < frame2.rows; i++) {
		//	for (int j = 0; j < frame2.cols; j++) {
		//		// Get the pixel in BGR space: 
		//		cout << i << " " << j << ": " << frame2.ptr<Vec3b>(i)[j] << endl;
		//		Vec3b pix_bgr = frame2.ptr<Vec3b>(i)[j];
		//		float B = pix_bgr.val[0];
		//		float G = pix_bgr.val[1];
		//		float R = pix_bgr.val[2];
		//	}
		//}

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
	else {
		cout << "Unkown Operation" << endl;
	}

	return 0;
}

// Function detectAndDisplay
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	Mat crop;
	Mat res;
	Mat gray;
	string text;
	stringstream sstm;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	// Set Region of Interest
	cv::Rect roi_b;
	cv::Rect roi_c;

	size_t ic = 0; // ic is index of current element
	int ac = 0; // ac is area of current element

	size_t ib = 0; // ib is index of biggest element
	int ab = 0; // ab is area of biggest element

	for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)

	{
		roi_c.x = faces[ic].x;
		roi_c.y = faces[ic].y;
		roi_c.width = (faces[ic].width);
		roi_c.height = (faces[ic].height);

		ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

		roi_b.x = faces[ib].x;
		roi_b.y = faces[ib].y;
		roi_b.width = (faces[ib].width);
		roi_b.height = (faces[ib].height);

		ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

		if (ac > ab)
		{
			ib = ic;
			roi_b.x = faces[ib].x;
			roi_b.y = faces[ib].y;
			roi_b.width = (faces[ib].width);
			roi_b.height = (faces[ib].height);
		}

		crop = frame(roi_b);
		resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
		cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

		// Form a filename
		filename = "";
		stringstream ssfn;
		ssfn << filenumber << ".jpeg";
		filename = ssfn.str();
		filenumber++;

		imwrite(filename, gray);

		Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}

	// Show image
	sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
	text = sstm.str();

	putText(frame, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	imshow("original", frame);

	if (!crop.empty())
	{
		imshow("detected", crop);
	}
	else
		destroyWindow("detected");

	waitKey();
}