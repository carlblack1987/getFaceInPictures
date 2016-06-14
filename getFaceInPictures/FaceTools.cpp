#include "stdafx.h"

const float eyeSearchRowStartRatio = 0.102;
const float eyeSearchRowEndRatio = 0.488;
const float mouthSearchRowStartRatio = 0.530;
const float mouthSearchRowEndRatio = 0.909;
const float noseSearchRowStartRatio = 0.5;
const float noseSearchRowEndRatio = 0.75;
const float feaSearchColStartRatio = 0.02;
const float feaSearchColEndRatio = 0.98;
const int noseMinSize = 20;
const int noseMaxSize = 300;
const int nosePosRange = 15;
const int noseDistance = 30;
const int noseVectorX = 17;
const int noseVectorY = 13;
const int noseAreaRange = 10;
const float noseStartRatio = 0.3;
const float noseEndRatio = 0.85;
const int eyeMinSize = 100;
const int mouMinSize = 110;
const int mouthPosRange = 5;
const int eyesDistance = 40;
const int eyeSizeLimit = 1250;
const int binaryThres = 40;
const int eyeBrowDis = 40;
const double eyeWidthRatioLimit = 0.50;
const double eyeHeightRatioLimit = 0.64;
const float eyeStartRatio = 0.3;
const float eyeEndRatio = 0.95;
const int cannyLowThres = 60;
const int cannyHighThres = 120;

CascadeClassifier face, mouth, eye, nose;
Mat frame, grayframe, testframe;
Mat temp_rgb, temp_ycbcr, temp_hsv, temp_lab, temp_face;
Mat thres_ycbcr, thres_hsv, thres_lab;
Mat template1;
Point pMouth1, pMouth2, pLeye1, pLeye2, pReye1, pReye2, pLeyec, pReyec, peyec, pMouthc;
Vec3b cwhite = Vec3b::all(255);
Vec3b cblack = Vec3b::all(0);
vector<string> templist;
string outputpath = "E:\\faceTemplate\\FaceOutput\\";

struct binaryPoint {
	int i;
	int j;
	binaryPoint & operator = (binaryPoint &b);
};

binaryPoint & binaryPoint::operator = (binaryPoint &b) {
	this->i = b.i;
	this->j = b.j;
	return *this;
}

binaryPoint operator + (binaryPoint a, binaryPoint b) {
	binaryPoint temp;
	temp.i = a.i + b.i;
	temp.j = a.j + b.j;
	return temp;
}

bool operator == (binaryPoint a, binaryPoint b) {
	if (a.i == b.i && a.j == b.j)
		return true;
	else
		return false;
}

binaryPoint movements[4] = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };

Mat norm_0_255(const Mat& src) {
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

bool FaceTools::R1(int R, int G, int B) {
	//Old rule
	bool e1 = (R > 95) && (G > 40) && (B > 20) && ((max(R, max(G, B)) - min(R, min(G, B))) > 15) && (abs(R - G) > 15) && (R > G) && (R > B);
	bool e2 = (R > 220) && (G > 210) && (B > 170) && (abs(R - G) <= 15) && (R > B) && (G > B);
	return (e1 || e2);
	//New rule
	/*bool e1 = R > G && R > B && G > B;
	return e1;*/
}

bool FaceTools::R2(float Y, float Cr, float Cb) {
	bool e3 = Cr <= 1.5862*Cb + 20;
	bool e4 = Cr >= 0.3448*Cb + 76.2069;
	bool e5 = Cr >= -4.5652*Cb + 234.5652;
	bool e6 = Cr <= -1.15*Cb + 301.75;
	bool e7 = Cr <= -2.2857*Cb + 432.85;
	return e3 && e4 && e5 && e6 && e7;
}

bool FaceTools::R3(float H, float S, float V) {
	return (H < 25) || (H > 230);
}

Mat FaceTools::GetSkin(Mat const &src, Mat &dst2) {
	// allocate the result matrix
	Mat dst = src.clone();

	Mat src_ycrcb, src_hsv;
	// OpenCV scales the YCrCb components, so that they
	// cover the whole value range of [0,255], so there's
	// no need to scale the values:
	cvtColor(src, src_ycrcb, CV_BGR2YCrCb);
	// OpenCV scales the Hue Channel to [0,180] for
	// 8bit images, so make sure we are operating on
	// the full spectrum from [0,360] by using floating
	// point precision:
	src.convertTo(src_hsv, CV_32FC3);
	cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
	// Now scale the values between [0,255]:
	normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

	//Record max rectangle in the frame
	int max_pixel_num = 0, temp_pixel_num = 0, left_pos = -1, right_pos = -1, top_pos = -1, bottom_pos = -1;
	int start_flag = 0;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
			int B = pix_bgr.val[0];
			int G = pix_bgr.val[1];
			int R = pix_bgr.val[2];
			// apply rgb rule
			bool a = R1(R, G, B);

			Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
			int Y = pix_ycrcb.val[0];
			int Cr = pix_ycrcb.val[1];
			int Cb = pix_ycrcb.val[2];
			// apply ycrcb rule
			bool b = R2(Y, Cr, Cb);

			Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
			float H = pix_hsv.val[0];
			float S = pix_hsv.val[1];
			float V = pix_hsv.val[2];
			// apply hsv rule
			bool c = R3(H, S, V);

			if (!(a&&b&&c)) {
				dst.ptr<Vec3b>(i)[j] = cblack;
				//dst2.ptr<Vec3b>(i)[j] = cblack;
				//start_flag = 0;
			}
		}
	}

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			if (dst.ptr<Vec3b>(i)[j] == cblack) {
				continue;
				//start_flag = 0;
			}
			else {
				if (start_flag == 0) {
					start_flag = 1;
					temp_pixel_num = 1;
				}
				else {
					temp_pixel_num++;
				}
			}
		}
	}

	return dst;
}

//unsigned char FaceTools::R4_Hana(unsigned char *in_ppm, int wid, int hei, int bnd) {
//	int           i, j, k, widhei = wid * hei;
//	double        ev1[3] = { 0.504743, 0.606139, 0.614679 }, ev2[3] = { -0.774481, 0.003424, 0.632588 };
//	double        em1 = -253.50332, em2 = 66.71714;
//	double        a1 = 113.570871, b1 = 4753.807542, r1 = 4724.444164, r11 = r1 * r1;
//	double        a2 = -22.60376, b2 = 253.301218, r2 = 302.548677, r22 = r2 * r2;
//	double        a3 = -23.0 / 417.0, b3 = 31.12470024;
//	double        x, y, z;
//	//pa pb pc
//	unsigned char *out, *pa, *pb, *pc;
//
//	if ((out = calloc(wid * hei, sizeof(unsigned char))) == NULL){
//		fprintf(stderr, "memory allocation error : out in skin_color()\n");
//		exit(-1);
//	}
//	for (pa = in_ppm, pc = out, i = 0; i < widhei; i++, pa += bnd, pc++){
//		if (*pa <= *(pa + 1)) continue;
//		if (*(pa + 1) <= *(pa + 2)) continue;
//		for (pb = pa, x = em1, y = em2, j = 0; j < bnd; j++, pb++){
//			x += (z = (double)*pb) * ev1[j];
//			y += z * ev2[j];
//		}
//		//if(y - a3 * x - b3 > 0.0){*pc = 0x00; continue;}
//		if (b1 < 0.0){
//			if (((z = x - a1) * z + (z = y - b1) * z - r11) >= 0.0){ *pc = 0x00; continue; }
//		}
//		else {
//			if (((z = x - a1) * z + (z = y - b1) * z - r11) <= 0.0){ *pc = 0x00; continue; }
//		}
//		if (((z = x - a2) * z + (z = y - b2) * z - r22) >= 0.0){ *pc = 0x00; continue; }
//		*pc = 0xf0;
//	}
//	return(out);
//}

Mat FaceTools::labThreshold(const Mat & frame)
{
	Mat abc = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	cvtColor(frame, temp_lab, CV_BGR2XYZ);
	for (int i = 0; i < frame.rows; i++) {
		for (int j = 0; j < frame.cols; j++) {
			// Get the pixel in BGR space: 
			Vec3b pix_bgr = frame.ptr<Vec3b>(i)[j];
			float B = pix_bgr.val[0];
			float G = pix_bgr.val[1];
			float R = pix_bgr.val[2];
			// And apply RGB rule:
			bool bgr = R1(R, G, B);

			float X = pix_bgr.val[2] * 0.4124 + pix_bgr.val[1] * 0.3576 + pix_bgr.val[0] * 0.1805;
			float Y = pix_bgr.val[2] * 0.2127 + pix_bgr.val[1] * 0.7152 + pix_bgr.val[0] * 0.0722;
			float Z = pix_bgr.val[2] * 0.0193 + pix_bgr.val[1] * 0.1192 + pix_bgr.val[0] * 0.9505;

			float Xn = X / 95.047;
			float Yn = Y / 100.00;
			float Zn = Z / 108.883;

			if (Xn > 0.008856)
				Xn = powf(Xn, (1 / 3));
			else
				Xn = (7.787 * Xn) + (16 / 116);
			if (Yn > 0.008856)
				Yn = powf(Yn, (1 / 3));
			else
				Yn = (7.787 * Yn) + (16 / 116);
			if (Zn > 0.008856)
				Zn = powf(Zn, (1 / 3));
			else
				Zn = (7.787 * Zn) + (16 / 116);

			float L = (116 * Yn) - 16;
			float a = 500 * (Xn - Yn);
			float b = 200 * (Yn - Zn);
			// If not skin, then black 
			if (bgr) {
				abc.at<unsigned char>(i, j) = 255;
			}
		}
	}
	return abc;
}

int FaceTools::detectFacialFeatures(vector< Rect > faces) {
	if (faces.size() <= 0)
		return 0;
	for (int i = 0; i < faces.size(); i++)
	{
		//rectangle(frame, faces[i], Scalar(255, 0, 0), 1, 1, 0);
		Mat face = frame(faces[i]);
		cvtColor(face, face, CV_BGR2GRAY);
		vector <Rect> mouthi;
		vector <Rect> eyei;
		mouth.detectMultiScale(face, mouthi);
		eye.detectMultiScale(face, eyei);
		cout << "Mouth size: " << mouthi.size() << endl;
		cout << "Eyes size: " << eyei.size() << endl;
		for (int k = 0; k < mouthi.size(); k++)
		{
			//Draw the position of the mouth.
			pMouth1.x = mouthi[0].x + faces[i].x;
			pMouth1.y = mouthi[0].y + faces[i].y;
			pMouth2.x = pMouth1.x + mouthi[0].width;
			pMouth2.y = pMouth1.y + mouthi[0].height;
			pMouthc.x = (pMouth1.x + pMouth2.x) / 2;
			pMouthc.y = (pMouth1.y + pMouth2.y) / 2;
			//Point pt1(mouthi[0].x + faces[i].x, mouthi[0].y + faces[i].y);
			//Point pt2(pt1.x + mouthi[0].width, pt1.y + mouthi[0].height);
			//rectangle(frame, pt1, pt2, Scalar(255, 0, 0), 1, 1, 0);
			//rectangle(frame, pMouth1, pMouth2, Scalar(255, 0, 0), 1, 1, 0);
			//cout << "Mouth position: " <<pt1.x<<" "<<pt1.y<< endl;
		}

		if (eyei.size() == 1) {
			//Draw the position of eyes.
			pLeye1.x = eyei[0].x + faces[i].x;
			pLeye1.y = eyei[0].y + faces[i].y;
			pLeye2.x = pLeye1.x + eyei[0].width;
			pLeye2.y = pLeye1.y + eyei[0].height;
			//rectangle(frame, pLeye1, pLeye2, Scalar(0, 255, 0), 1, 1, 0);
		}
		else if (eyei.size() >= 2) {
			//Draw the position of eyes.
			pLeye1.x = eyei[0].x + faces[i].x;
			pLeye1.y = eyei[0].y + faces[i].y;
			pLeye2.x = pLeye1.x + eyei[0].width;
			pLeye2.y = pLeye1.y + eyei[0].height;
			//Calculate the center point of left eye.
			pLeyec.x = (pLeye1.x + pLeye2.x) / 2;
			pLeyec.y = (pLeye1.y + pLeye2.y) / 2;
			pReye1.x = eyei[1].x + faces[i].x;
			pReye1.y = eyei[1].y + faces[i].y;
			pReye2.x = pReye1.x + eyei[0].width;
			pReye2.y = pReye1.y + eyei[0].height;
			//Calculate the center point of right eye.
			pReyec.x = (pReye1.x + pReye2.x) / 2;
			pReyec.y = (pReye1.y + pReye2.y) / 2;
			//Calculate the center point of eye line.
			peyec.x = (pLeyec.x + pReyec.x) / 2;
			peyec.y = (pLeyec.y + pReyec.y) / 2;

			//rectangle(frame, pLeye1, pLeye2, Scalar(0, 255, 0), 1, 1, 0);
			//rectangle(frame, pReye1, pReye2, Scalar(0, 255, 0), 1, 1, 0);
			//line(frame, pLeyec, pReyec, Scalar(0, 255, 0), 1, 1, 0);
			//if (mouthi.size() > 0)
			//line(frame, peyec, pMouthc, Scalar(0, 255, 0), 1, 1, 0);
		}
	}
	return 1;
}


Mat FaceTools::maskOnImage(Mat const &src, Mat const &src2) {
	// allocate the result matrix
	Mat src_clone = src.clone();
	Mat dst = src2.clone();
	int minDis = 99999, maxDis = 0, currentDis;
	int minX, minY, maxX, maxY;

	cout << "rows: " << src_clone.rows << " " << src_clone.cols << endl;
	cout << "rows2: " << dst.rows << " " << dst.cols << endl;

	for (int i = 0; i < src_clone.rows; i++) {
		for (int j = 0; j < src_clone.cols; j++) {
			if (src_clone.ptr<Vec3b>(i)[j] == cblack) {
			/*if (src_clone.ptr<Vec3b>(i)[j] == cblack || i < 80 || i > src_clone.rows - 80
				|| j < 150 || j > src_clone.cols - 150) {*/

				Vec3b pix_bgr = dst.ptr<Vec3b>(i)[j];
				int B = pix_bgr.val[0];
				int G = pix_bgr.val[1];
				int R = pix_bgr.val[2];

				pix_bgr = cblack;
				dst.ptr<Vec3b>(i)[j] = Vec3b(100, 100, 100);
			}
			else {

			}
		}
	}
	cout << "MarkonImage end" << endl;
	return dst;
}

Mat FaceTools::templateMatch(Mat &src, vector<string> templist, int match_method) {
	// allocate the result matrix
	Mat src_clone = src.clone();
	Mat result;
	Point lefteye, righteye, enterEye, nose;
	int eyeDistance = 9999, noseEyeDistance = 9999;

	cout << "rows: " << src_clone.rows << " " << src_clone.cols << endl;
	for (int i = 0; i < templist.size(); i++) {
		//If the template is a gray image, use type CV_LOAD_IMAGE_GRAYSCALE.
		Mat templ_mat = imread(templist[i], CV_LOAD_IMAGE_GRAYSCALE);
		cout << "templ: " << templ_mat.rows << " " << templ_mat.cols << endl;

		//Do the match and normalize.
		matchTemplate(src_clone, templ_mat, result, match_method);
		normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
		//Localizing the best match with points
		double minVal, maxVal;
		Point minLoc, maxLoc, matchLoc;

		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

		// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
		if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
			matchLoc = minLoc;
		else
			matchLoc = maxLoc;
		/// Show me what you got
		//line(frame, pLeyec, pReyec, Scalar(0, 255, 0), 1, 1, 0);
		cout << templist[i] << "____" << templist[i].find("template_nose_normal.jpg") << endl;
		if (templist[i].find("template_nose_normal.jpg") > 0 && templist[i].find("template_nose_normal.jpg") < templist[i].length()) {
			nose.x = matchLoc.x + templ_mat.cols / 2;
			nose.y = matchLoc.y + templ_mat.rows / 2;
		}
		else if (templist[i].find("template_lefteye_normal.jpg") > 0 && templist[i].find("template_lefteye_normal.jpg") < templist[i].length()) {
			lefteye.x = matchLoc.x + templ_mat.cols / 2;
			lefteye.y = matchLoc.y + templ_mat.rows / 2;
		}
		else if (templist[i].find("template_righteye_normal.jpg") > 0 && templist[i].find("template_righteye_normal.jpg") < templist[i].length()) {
			righteye.x = matchLoc.x + templ_mat.cols / 2;
			righteye.y = matchLoc.y + templ_mat.rows / 2;
		}

		rectangle(src, matchLoc, Point(matchLoc.x + templ_mat.cols, matchLoc.y + templ_mat.rows), Scalar(0, 255, 0), 1, 1, 0);
		rectangle(result, matchLoc, Point(matchLoc.x + templ_mat.cols, matchLoc.y + templ_mat.rows), Scalar(0, 255, 0), 1, 1, 0);
	}

	eyeDistance = sqrt(pow(lefteye.x - righteye.x, 2) + pow(lefteye.y - righteye.y, 2));
	cout << "Eye distance: " << eyeDistance << endl;
	line(src, lefteye, righteye, Scalar(0, 255, 0), 1, 1, 0);

	return result;
}

int FaceTools::detectFaceSkin(Mat &src) {
	if (!src.empty()){
		Mat frame = src.clone(), finalresult = src.clone(), faceArea;
		Mat faceBorder = getSobelBorder(frame);
		
		cvtColor(frame, grayframe, CV_BGR2GRAY);
		equalizeHist(grayframe, testframe);
		Mat thres_lab = this->GetSkin(frame, testframe);
		Mat testframe2 = this->maskOnImage(thres_lab, frame);
		Mat testframe3, testframe4, eyeSkin;
		erode(testframe2, testframe3, Mat(5, 5, CV_8U), Point(-1, -1), 2);
		dilate(testframe3, testframe4, Mat(5, 5, CV_8U), Point(-1, -1), 2);
		Mat orginal2 = testframe4.clone();
		Mat testframe5 = testframe4.clone();
		for (int i = 0; i < testframe5.rows; i++) {
			for (int j = 0; j < testframe5.cols; j++) {
				testframe5.ptr<Vec3b>(i)[j] = Vec3b(100, 100, 100);
			}
		}
		this->processImage(testframe4, testframe5);
		this->findFace(testframe5, frame, faceArea);
		//imwrite(outputpath + "TotalFaceAvg" + temp + ".jpg", norm_0_255(mean.reshape(1, db[0].rows)));
		this->findMass(testframe5);
		this->findFacialFeatures(faceArea, eyeSkin, faceArea);

		vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_PXM_BINARY);
		compression_params.push_back(1);

		imwrite(outputpath + "faceGet_Test.pgm", faceArea, compression_params);
		//imwrite(outputpath + "faceGet_Test.jpg", faceArea, CV_IMWRITE_PXM_BINARY);

		imshow("original", src);
		//imshow("gray222", grayframe);
		//imshow("gray", testframe);
		imshow("face", orginal2);
		imshow("operated", testframe5);
		imshow("find", frame);
		imshow("result", faceArea);
		//imshow("Eys Skin", eyeSkin);
		//imshow("Face Border", faceBorder);
		waitKey();
	}
	else{
		printf(" No input frame detected.");
		return 0;
	}
}

int FaceTools::detectFaceSkinInVideo(Mat &src) {
	if (!src.empty()){
		Mat frame = src.clone(), finalresult = src.clone(), faceArea;
		Mat faceBorder = getSobelBorder(frame);

		cvtColor(frame, grayframe, CV_BGR2GRAY);
		equalizeHist(grayframe, testframe);
		Mat thres_lab = this->GetSkin(frame, testframe);
		Mat testframe2 = this->maskOnImage(thres_lab, frame);
		Mat testframe3, testframe4, eyeSkin;
		erode(testframe2, testframe3, Mat(5, 5, CV_8U), Point(-1, -1), 2);
		dilate(testframe3, testframe4, Mat(5, 5, CV_8U), Point(-1, -1), 2);
		Mat orginal2 = testframe4.clone();
		Mat testframe5 = testframe4.clone();
		for (int i = 0; i < testframe5.rows; i++) {
			for (int j = 0; j < testframe5.cols; j++) {
				testframe5.ptr<Vec3b>(i)[j] = Vec3b(100, 100, 100);
			}
		}
		this->processImage(testframe4, testframe5);
		//imshow("7777", testframe5);
		this->findFace(testframe5, frame, faceArea);
		//imwrite(outputpath + "TotalFaceAvg" + temp + ".jpg", norm_0_255(mean.reshape(1, db[0].rows)));
		this->findMass(testframe5);
		this->findFacialFeatures(faceArea, eyeSkin, faceArea);

		vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_PXM_BINARY);
		compression_params.push_back(1);

		imwrite(outputpath + "faceGet_Test.pgm", faceArea, compression_params);
		//imwrite(outputpath + "faceGet_Test.jpg", faceArea, CV_IMWRITE_PXM_BINARY);

		imshow("original", src);
		//imshow("gray222", grayframe);
		imshow("gray", testframe);
		imshow("face", orginal2);
		imshow("operated", testframe5);
		imshow("find", frame);
		imshow("result", faceArea);
		//imshow("Eys Skin", eyeSkin);
		//imshow("Face Border", faceBorder);
	}
	else{
		printf(" No input frame detected.");
		return 0;
	}
}

Mat FaceTools::detectFaceCornerInVideo(Mat &src, Mat &pre) {
	Mat srcClone = src.clone(), next;
	vector<Point>cornerVec;
	/*
	//This part uses calcOpticalFlowPyrLK to get change by frame in the video.
	vector<Point2f> prepoint, nextpoint;
	vector<uchar> state;
	vector<float>err;
	if (src.empty())
		return srcClone;

	cvtColor(srcClone, next, CV_BGR2GRAY);

	if (!next.empty() && !pre.empty())
	{

		goodFeaturesToTrack(pre, prepoint, 500, 0.001, 10, Mat(), 3, false, 0.04);
		cornerSubPix(pre, prepoint, Size(10, 10), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
		calcOpticalFlowPyrLK(pre, next, prepoint, nextpoint, state, err, Size(31, 31), 3);
		for (int i = 0; i<state.size(); i++)
		{
			if (state[i] != 0)
			{
				line(frame, Point((int)prepoint[i].x, (int)prepoint[i].y), Point((int)nextpoint[i].x, (int)nextpoint[i].y), Scalar::all(-1));
			}
		}
		namedWindow("frame", 0);
		imshow("frame", frame);
		waitKey(1);
	}*/

	if (!src.empty()){
		Mat frame = src.clone(), finalresult = src.clone(), faceArea;
		Mat faceBorder = getSobelBorder(frame);

		cvtColor(frame, grayframe, CV_BGR2GRAY);
		equalizeHist(grayframe, testframe);
		Mat thres_lab = this->GetSkin(frame, testframe);
		Mat testframe2 = this->maskOnImage(thres_lab, frame);
		Mat testframe3, testframe4, eyeSkin;
		erode(testframe2, testframe3, Mat(5, 5, CV_8U), Point(-1, -1), 2);
		dilate(testframe3, testframe4, Mat(5, 5, CV_8U), Point(-1, -1), 2);
		Mat orginal2 = testframe4.clone();
		Mat testframe5 = testframe4.clone();
		for (int i = 0; i < testframe5.rows; i++) {
			for (int j = 0; j < testframe5.cols; j++) {
				testframe5.ptr<Vec3b>(i)[j] = Vec3b(100, 100, 100);
			}
		}
		this->processImage(testframe4, testframe5);
		Point startP = this->findFace(testframe5, frame, faceArea);
		this->findMass(testframe5);
		//this->findFacialFeatures(faceArea, eyeSkin, faceArea);
		detectCornerPoints(faceArea, frame, startP, cornerVec);

		imshow("original", src);
		//imshow("gray222", grayframe);
		imshow("gray", testframe);
		imshow("face", orginal2);
		imshow("operated", testframe5);
		imshow("find", frame);
		imshow("result", faceArea);
		//imshow("Eys Skin", eyeSkin);
		//imshow("Face Border", faceBorder);
	}
	else{
		printf(" No input frame detected.");
	}

	return next;
}

int FaceTools::processImage(Mat &src, Mat &dst) {
	int size = 0;
	int judgeM[1000][1000];
	memset(judgeM, 0, sizeof(judgeM));
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.ptr<Vec3b>(i)[j] != Vec3b(100, 100, 100)){
				cout << "Pos: " << i << " " << j << endl;
				size = 0;
				this->getObjectSize(src, 1, i, j, size, judgeM);
				cout << "current size: " << size << endl;
				//If object size is less than 100 * 100, just filter it.
				if (size < 7500)
					eraseObject(src, i, j, 0, Vec3b(100, 100, 100));
				else {
					copyObject(src, dst, i, j);
				}
				//imshow("Processed Picture", src);
				//imshow("Processed Picture2", dst);
				//waitKey();
			}
		}
	}
	cout << "Function processImage end" << endl;
	return 1;
}

int FaceTools::getObjectSize(Mat &src, int type, int x, int y, int &size, int(&judge)[1000][1000]) {
	//cout << "Here pos: " << x << " " << y << endl;
	if (x < 0 || x >= src.rows - 1)
		return 0;
	if (y < 0 || y >= src.cols - 1)
		return 0;
	if (src.ptr<Vec3b>(x)[y] == Vec3b(100, 100, 100) && type == 1) {
		return 0;
	}
	if (src.at<uchar>(x, y) == 255 && type == 2) {
		return 0;
	}
	if (judge[x][y] == 1) {
		return 0;
	}
	/*if (size > 40000) {
		return 0;
	}*/
	//cout << "Here size: " << size << ". judge: " << judge[x][y] << endl;
	judge[x][y] = 1;
	size++;

	getObjectSize(src, type, x, y - 1, size, judge);
	getObjectSize(src, type, x, y + 1, size, judge);
	getObjectSize(src, type, x - 1, y, size, judge);
	getObjectSize(src, type, x + 1, y, size, judge);

	return 1;
}

int FaceTools::getHoleSize(Mat &src, int x, int y, int &size, int(&judge)[1000][1000]) {
	if (x < 0 || x >= src.rows)
		return 0;
	if (y < 0 || y >= src.cols)
		return 0;
	if (src.ptr<Vec3b>(x)[y] != Vec3b(100, 100, 100)) {
		return 0;
	}
	if (judge[x][y] == 1) {
		return 0;
	}
	/*if (size > 40000) {
	return 0;
	}*/
	//cout << "Here size: " << size << ". judge: " << judge[x][y] << endl;
	judge[x][y] = 1;
	size++;

	getHoleSize(src, x, y - 1, size, judge);
	getHoleSize(src, x, y + 1, size, judge);
	getHoleSize(src, x - 1, y, size, judge);
	getHoleSize(src, x + 1, y, size, judge);

	return 1;
}

int FaceTools::copyObject(Mat &src, Mat &dst, int x, int y) {
	if (x < 0 || x >= src.rows)
		return 0;
	if (y < 0 || y >= src.cols)
		return 0;
	if (src.ptr<Vec3b>(x)[y] == Vec3b(100, 100, 100)) {
		return 0;
	}

	dst.ptr<Vec3b>(x)[y] = src.ptr<Vec3b>(x)[y];
	src.ptr<Vec3b>(x)[y] = Vec3b(100, 100, 100);
	//cout << deep << endl;

	copyObject(src, dst, x, y - 1);
	copyObject(src, dst, x, y + 1);
	copyObject(src, dst, x - 1, y);
	copyObject(src, dst, x + 1, y);

	return 1;
}

int FaceTools::copyObjectBin(Mat &src, Mat &dst, int x, int y) {
	if (x < 0 || x >= src.rows)
		return 0;
	if (y < 0 || y >= src.cols)
		return 0;
	if (src.at<uchar>(x, y) == 255) {
		return 0;
	}

	dst.at<uchar>(x, y) = src.at<uchar>(x, y);
	src.at<uchar>(x, y) = 255;
	//cout << deep << endl;

	copyObjectBin(src, dst, x, y - 1);
	copyObjectBin(src, dst, x, y + 1);
	copyObjectBin(src, dst, x - 1, y);
	copyObjectBin(src, dst, x + 1, y);

	return 1;
}

int FaceTools::eraseObject(Mat &src, int x, int y, int deep, Vec3b a) {
	if (x < 0 || x >= src.rows)
		return 0;
	if (y < 0 || y >= src.cols)
		return 0;
	if (src.ptr<Vec3b>(x)[y] == a) {
		return 0;
	}

	src.ptr<Vec3b>(x)[y] = a;
	//cout << deep << endl;

	eraseObject(src, x, y - 1, deep + 1, a);
	eraseObject(src, x, y + 1, deep + 1, a);
	eraseObject(src, x - 1, y, deep + 1, a);
	eraseObject(src, x + 1, y, deep + 1, a);

	return 1;
}

int FaceTools::eraseObject(Mat &src, int x, int y, int deep, uchar a) {
	if (x < 0 || x >= src.rows)
		return 0;
	if (y < 0 || y >= src.cols)
		return 0;
	if (src.at<uchar>(x, y) == a) {
		return 0;
	}

	src.at<uchar>(x, y) = a;
	//cout << deep << endl;

	eraseObject(src, x, y - 1, deep + 1, a);
	eraseObject(src, x, y + 1, deep + 1, a);
	eraseObject(src, x - 1, y, deep + 1, a);
	eraseObject(src, x + 1, y, deep + 1, a);

	return 1;
}

Point FaceTools::findFace(Mat &src, Mat &dst, Mat &result) {
	cout << "Function findFace start" << endl;
	int minDis = 99999, maxDis = 0, currentDis;
	int minX = src.cols - 1, minY = src.rows - 1, maxX = 0 , maxY = 0;
	if (src.rows <= 0 || dst.rows <= 0)
		return Point(0, 0);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.ptr<Vec3b>(i)[j] == Vec3b(100, 100, 100)){
				continue;
			}
			else {
				minY = i < minY ? i : minY;
				maxY = i > maxY ? i : maxY;
				minX = j < minX ? j : minX;
				maxX = j > maxX ? j : maxX;
			}
		}
	}

	//Erase all image out of this rectangle.
	rectangle(dst, Point(minX, minY), Point(maxX, maxY), Scalar(0, 255, 0), 1, 1, 0);
	//Use the face area to generate a new mat.
	result = dst(Range(minY, maxY), Range(minX, maxX));
	src = src(Range(minY, maxY), Range(minX, maxX));
	cout << "Function findFace end" << endl;
	return Point(minX, minY);

}

int FaceTools::findFacialFeatures(Mat &src, Mat &dst, Mat &result) {
	vector<eyeInfo> eyeVec;
	vector<mouthInfo> mouVec;
	vector<Point>cornerVec;

	Point noseCenter;

	imshow("RGB Face", src);

	detectCornerPoints(src, src.clone(), Point(0, 0), cornerVec);

	if (!src.empty()){
		Mat frame = src.clone();

		Mat faceBin = getBinaryFormat(frame, binaryThres);
		imshow("bin", faceBin);

		Mat faceBin2 = faceBin.clone();

		refineBinaryByCorner(faceBin, faceBin2, cornerVec, 0);

		//Added 20160518
		//Try to find eyes through finding circles in the image by Hough.
		getHoughCircles(frame, 1, 10);
		//Try to find contours of eyes and nose by findContours
		Mat cornerFrame = getContoursByCplus(frame, 0);
		//Ended 20160518

		cvtColor(frame, frame, CV_RGB2GRAY);

		//Generate a new mat only contains eye area
		Mat eyeAreaOri = faceBin2(Range(frame.rows * eyeSearchRowStartRatio, frame.rows * eyeSearchRowEndRatio),
			Range(frame.cols * feaSearchColStartRatio, frame.cols * feaSearchColEndRatio));
		//Generate a new mat only contains mouth area
		Mat mouthAreaOri = faceBin2(Range(frame.rows * mouthSearchRowStartRatio, frame.rows * mouthSearchRowEndRatio),
			Range(frame.cols * feaSearchColStartRatio, frame.cols * feaSearchColEndRatio));
		//Generate a new mat only contains mouth area
		Mat eyeAreaGray = frame(Range(frame.rows * eyeSearchRowStartRatio, frame.rows * eyeSearchRowEndRatio),
			Range(frame.cols * feaSearchColStartRatio, frame.cols * feaSearchColEndRatio));

		Mat eyeResult = getExactEyes(eyeAreaOri, eyeVec, eyeMinSize);
		Mat mouthResult = getExactMouth(mouthAreaOri, mouVec, mouMinSize);
		Mat eyeResultGray = getExactEyesGray(eyeAreaGray, eyeMinSize);
		imshow("eyes", eyeResult);
		imshow("Mouth", mouthResult);
		//imshow("Nose", noseResult);

		drawFacialFeatures(result, faceBin, eyeVec, mouVec, noseCenter);

		calculateFace(result, eyeAreaOri, eyeVec, mouVec, noseCenter);

		dst = result;
	}
	else{
		printf(" No input frame detected.");
		return 0;
	}
	return 1;
}

int FaceTools::scanHole(Mat &src, int type, int x, int y, Point &p1) {
	if (x < 0 || x >= src.rows)
		return 0;
	if (y < 0 || y >= src.cols)
		return 0;
	if (src.ptr<Vec3b>(x)[y] != Vec3b(100, 100, 100)) {
		return 0;
	}

	src.ptr<Vec3b>(x)[y] = Vec3b(200, 200, 200);

	if (type == 1) {
		p1.x = p1.x < y ? p1.x : y;
		p1.y = p1.y < x ? p1.y : x;
	}
	else {
		p1.x = p1.x > y ? p1.x : y;
		p1.y = p1.y > x ? p1.y : x;
	}
	//cout << deep << endl;

	scanHole(src, type, x, y - 1, p1);
	scanHole(src, type, x, y + 1, p1);
	scanHole(src, type, x - 1, y, p1);
	scanHole(src, type, x + 1, y, p1);

	return 1;
}

int FaceTools::scanObject(Mat &src, int type, int x, int y, Point &p1) {
	if (x < 0 || x >= src.rows)
		return 0;
	if (y < 0 || y >= src.cols)
		return 0;
	if (src.at<uchar>(x, y) == 255) {
		return 0;
	}

	src.at<uchar>(x, y) = 255;

	if (type == 1) {
		p1.x = p1.x < y ? p1.x : y;
		p1.y = p1.y < x ? p1.y : x;
	}
	else {
		p1.x = p1.x > y ? p1.x : y;
		p1.y = p1.y > x ? p1.y : x;
	}
	//cout << deep << endl;

	scanObject(src, type, x, y - 1, p1);
	scanObject(src, type, x, y + 1, p1);
	scanObject(src, type, x - 1, y, p1);
	scanObject(src, type, x + 1, y, p1);

	return 1;
}

Mat FaceTools::getVerticalProjection(Mat &src) {
	Mat srcImage = src.clone();
	//imshow("Source", srcImage);
	cvtColor(srcImage, srcImage, CV_RGB2GRAY);
	blur(srcImage, srcImage, Size(3, 3), Point(-1, -1));
	equalizeHist(srcImage, srcImage);
	threshold(srcImage, srcImage, 30, 255, CV_THRESH_BINARY);
	//imshow("d",srcImage);
	int *colheight = new int[srcImage.cols];
	memset(colheight, 0, srcImage.cols * 4);
	//  memset(colheight,0,src->width*4);    
	// CvScalar value;   
	int value;
	for (int i = 0; i<srcImage.rows; i++)
		for (int j = 0; j<srcImage.cols; j++)
		{
			//value=cvGet2D(src,j,i);  
			value = srcImage.at<uchar>(i, j);
			if (value == 255)
			{
				colheight[j]++;
			}

		}

	Mat histogramImage(srcImage.rows, srcImage.cols, CV_8UC1);
	for (int i = 0; i<srcImage.rows; i++)
		for (int j = 0; j<srcImage.cols; j++)
		{
			value = 0;
			histogramImage.at<uchar>(i, j) = value;
		}
	//imshow("d", histogramImage);
	for (int i = 0; i<srcImage.cols; i++)
		for (int j = 0; j<colheight[i]; j++)
		{
			value = 255;
			histogramImage.at<uchar>(j, i) = value;
		}

	//imshow("C",srcImage);
	imshow("Vertical", histogramImage);

	return histogramImage;
}

Mat FaceTools::getSobelBorder(Mat src) {
	if (src.rows <= 0 || src.cols <= 0) {
		cout << "Illegal size of face area." << endl;
		exit(1);
	}
	else {
		Mat dst_x, dst_y, dst;
		Sobel(src, dst_x, src.depth(), 1, 0);
		Sobel(src, dst_y, src.depth(), 0, 1);
		convertScaleAbs(dst_x, dst_x);
		convertScaleAbs(dst_y, dst_y);
		addWeighted(dst_x, 0.5, dst_y, 0.5, 0, dst);
		return dst;
	}
}

Mat FaceTools::getHorizontalProjection(Mat &src) {
	Mat srcImage = src.clone();
	Mat src2 = src.clone();
	imshow("Source", srcImage);
	cvtColor(srcImage, srcImage, CV_RGB2GRAY);
	equalizeHist(srcImage, srcImage);
	threshold(srcImage, srcImage, 30, 255, CV_THRESH_BINARY);
	//imshow("d",srcImage);
	int *rowheight = new int[srcImage.rows];
	memset(rowheight, 0, srcImage.rows * 4);
	//  memset(colheight,0,src->width*4);    
	// CvScalar value;   
	int value;
	for (int i = 0; i<srcImage.cols; i++)
		for (int j = 0; j<srcImage.rows; j++)
		{
			//value=cvGet2D(src,j,i);  
			value = srcImage.at<uchar>(j, i);
			if (value == 255)
			{
				rowheight[j]++;
			}
			else
				src2.ptr<Vec3b>(j)[i] = Vec3b(0, 0, 255);

		}

	Mat histogramImage(srcImage.rows, srcImage.cols, CV_8UC1);
	for (int i = 0; i<srcImage.rows; i++)
		for (int j = 0; j<srcImage.cols; j++)
		{
			value = 0;
			histogramImage.at<uchar>(i, j) = value;
		}
	//imshow("d", histogramImage);
	for (int i = 0; i<srcImage.rows; i++)
		for (int j = 0; j<rowheight[i]; j++)
		{
			value = 255;
			histogramImage.at<uchar>(i, j) = value;
		}

	//imshow("Test2", src2);
	imshow("C", srcImage);
	imshow("Horizontal", histogramImage);
	//waitKey();
	return histogramImage;
}

Mat FaceTools::getBinaryFormat(Mat &src, int value) {
	Mat srcImage = src.clone();
	cvtColor(srcImage, srcImage, CV_RGB2GRAY);
	equalizeHist(srcImage, srcImage);
	threshold(srcImage, srcImage, value, 255, CV_THRESH_BINARY);
	return srcImage;
}

Mat FaceTools::getExactEyes(Mat &src, vector<eyeInfo> &eyeVec, int threshold) {
	Mat srcClone = src.clone();
	Mat srcClone2 = src.clone();
	Mat srcClone3 = src.clone();
	Mat result = src.clone();
	//Change the image from RGB to gray format and apply equalization
	Mat srcImage = src.clone();
	//Mat srcImage = src_gray.clone();
	//cvtColor(srcImage, srcImage, CV_RGB2GRAY);
	//equalizeHist(srcImage, srcImage);
	//imshow("eye area", src);

	eyeVec.clear();
	int judgeM[1000][1000];
	memset(judgeM, 0, sizeof(judgeM));
	int size = 0;
	if (srcClone.rows <= 0 || srcClone.cols <= 0) {
		cout << "Illegal size of face area." << endl;
		exit(1);
	}
	for (int i = 0; i < srcClone.rows; i++) {
		for (int j = 0; j < srcClone.cols; j++) {
			if (srcClone.at<uchar>(i, j) == 0){
				//cout << "Eye Pos: " << i << " " << j << endl;
				size = 0;
				this->getObjectSize(srcClone, 2, i, j, size, judgeM);
				//If object size is less than 100 * 100, just filter it.
				if (size > threshold) {
					Point topNode(j, i);
					Point botNode(0, 0);
					cout << "Pos: " << i << " " << j << endl;
					cout << "current EyeObject size: " << size << endl;
					this->scanObject(srcClone, 1, i, j, topNode);
					this->scanObject(srcClone2, 2, i, j, botNode);
					int centerX = (topNode.x + botNode.x) / 2;
					int centerY = (topNode.y + botNode.y) / 2;
					double eyeWidthRatio = (double)abs(topNode.x - botNode.x) / src.cols;
					double eyeHeightRatio = (double)abs(topNode.y - botNode.y) / src.rows;
					rectangle(result, topNode, botNode, Scalar(0, 255, 0), 1, 1, 0);
					cout << "Eye size limit: " << eyeSizeLimit << ", real size: " << size << endl;
					cout << "eyeWidthRatioLimit: " << eyeWidthRatio << " " << eyeHeightRatio << endl;
					/*imshow("222", result);
					waitKey();*/
					//if (centerX >= 20 && centerX <= srcClone.cols - 20) {
					if (size < eyeSizeLimit && eyeWidthRatio < eyeWidthRatioLimit 
						&& eyeHeightRatio < eyeHeightRatioLimit && (topNode.x != 0 || topNode.y != 0) && topNode.x < src.cols - 10 && topNode.y > 5) {
						int size = abs((botNode.x - topNode.x) * (botNode.y - topNode.y));
						if (eyeVec.size() == 0) {
							eyeInfo temp;
							temp.botNode = botNode;
							temp.topNode = topNode;
							temp.cenNode = Point(centerX, centerY);
							temp.type = centerX < srcClone.cols / 2 ? 1 : 2;
							temp.size = size;
							eyeVec.push_back(temp);
							cout << "Push object" << endl;
						}
						else if (eyeVec.size() == 1) {
							eyeInfo temp;
							temp.botNode = botNode;
							temp.topNode = topNode;
							temp.cenNode = Point(centerX, centerY);
							temp.type = centerX < srcClone.cols / 2 ? 1 : 2;
							temp.size = size;
							//The distance between two eyes can not be less than a value
							int dis = sqrt(pow(eyeVec[0].cenNode.x - centerX, 2) + pow(eyeVec[0].cenNode.y - centerY, 2));
							cout << "Object Distance " << dis << endl;
							cout << "Object replaced" << endl;
							if (dis >= eyesDistance)
								eyeVec.push_back(temp);
							//else if (size > eyeVec[0].size)
							else if (centerY > eyeVec[0].cenNode.y)
								eyeVec[0] = temp;
						}
						else {
							int minIndex = 0, minValue = 99999;
							int dis = 9999;
							for (int a = 0; a < eyeVec.size(); a++) {
								cout << "xxx: " << centerX << " " << centerY << endl;
								cout << "222: " << eyeVec[a].cenNode.x << " " << eyeVec[a].cenNode.y << endl;
								dis = sqrt(pow(eyeVec[a].cenNode.x - centerX, 2) + pow(eyeVec[a].cenNode.y - centerY, 2));
								cout << "Eye dis: " << dis << endl;
								if (dis < eyeBrowDis && eyeVec[a].cenNode.y < centerY) {
									cout << "Eye Object Replaced " << i << " " << j << endl;
									eyeInfo temp;
									temp.botNode = botNode;
									temp.topNode = topNode;
									temp.cenNode = Point(centerX, centerY);
									temp.type = centerX < srcClone.cols / 2 ? 1 : 2;
									temp.size = size;
									eyeVec[a] = temp;
									break;
								}
								/*if (eyeVec[i].size < minValue) {
									minValue = eyeVec[i].size;
									minIndex = i;
								}*/
							}

							/*int dis = sqrt(pow(eyeVec[minIndex].cenNode.x - centerX, 2) + pow(eyeVec[minIndex].cenNode.y - centerY, 2));
							cout << "current dis2: " << dis << endl;

							if (size > minValue && dis < eyeBrowDis) {
								cout << "Object Replaced " << i << " " << j << endl;
								eyeInfo temp;
								temp.botNode = botNode;
								temp.topNode = topNode;
								temp.type = centerX < srcClone.cols / 2 ? 1 : 2;
								temp.size = size;
								eyeVec[minIndex] = temp;
							}*/
						}
					}
					else {
						cout << "Object abandoned!" << endl;
					}

					//rectangle(result, topNode, botNode, Scalar(0, 255, 0), 1, 1, 0);
					//eraseObject(orginal2, i, j, 0);
				}
				else {
					//copyObject(orginal2, dst, i, j);
				}
			}
		}
	}
	//Draw rectangle on eyes.
	for (int i = 0; i < eyeVec.size(); i++) {
		rectangle(result, eyeVec[i].topNode, eyeVec[i].botNode, Scalar(0, 255, 0), 1, 1, 0);
		Rect tempEye(eyeVec[i].topNode, eyeVec[i].botNode);
		//Mat eyeTemp = src(tempEye);
		Mat eyeTemp = srcImage(tempEye);
		imshow("eye1", srcImage);
		//-- Find Eye Centers
		Point leftPupil = findEyeCenter(srcClone3, tempEye, "Left Eye");
		//Point leftPupil = findEyeCenter(srcImage, tempEye, "Left Eye");
		cout << "pupil: " << leftPupil.x << " " << leftPupil.y << endl;
		leftPupil.x += tempEye.x;
		leftPupil.y += tempEye.y;
		eyeVec[i].pupil = leftPupil;
		//circle(result, leftPupil, 3, 1234);
		//imshow("eye1", result);
	}
	
	return result;
}

Mat FaceTools::getExactMouth(Mat &src, vector<mouthInfo> &mouVec, int threshold) {
	cout << "Get exact mouth start" << endl;
	Mat srcClone = src.clone();
	Mat srcClone2 = src.clone();
	Mat srcClone3 = src.clone();
	Mat result = src.clone();

	//imshow("mouth area", src);
	//Change the image from RGB to gray format and apply equalization
	Mat srcImage = src.clone();
	//Mat srcImage = src_gray.clone();
	//cvtColor(srcImage, srcImage, CV_RGB2GRAY);
	//equalizeHist(srcImage, srcImage);

	mouVec.clear();
	int judgeM[1000][1000];
	memset(judgeM, 0, sizeof(judgeM));
	int size = 0;
	if (srcClone.rows <= 0 || srcClone.cols <= 0) {
		cout << "Illegal size of face area." << endl;
		exit(1);
	}
	for (int i = 0; i < srcClone.rows; i++) {
		for (int j = 0; j < srcClone.cols; j++) {
			if (srcClone.at<uchar>(i, j) == 0){
				//cout << "Eye Pos: " << i << " " << j << endl;
				size = 0;
				this->getObjectSize(srcClone, 2, i, j, size, judgeM);
				if (size > 0)
					cout << "current MouthObject size: " << size << endl;
				//If object size is less than 100 * 100, just filter it.
				if (size > threshold) {
					Point topNode(j, i);
					Point botNode(0, 0);
					cout << "Pos: " << i << " " << j << endl;
					cout << "current MouthObject size: " << size << endl;
					this->scanObject(srcClone, 1, i, j, topNode);
					this->scanObject(srcClone2, 2, i, j, botNode);
					int centerX = (topNode.x + botNode.x) / 2;
					int centerY = (topNode.y + botNode.y) / 2;
					int height = abs(topNode.y - botNode.y);
					int width = abs(topNode.x - botNode.x);
					//rectangle(result, topNode, botNode, Scalar(0, 255, 0), 1, 1, 0);
					if (centerY >= mouthPosRange && centerY <= srcClone.rows - mouthPosRange && height <= 30) {
					//if (width >= 40 && height <= 30) {
						int size = abs((botNode.x - topNode.x) * (botNode.y - topNode.y));
						int length = abs(botNode.x - topNode.x);
						if (mouVec.size() == 0) {
							mouthInfo temp;
							temp.botNode = botNode;
							temp.topNode = topNode;
							temp.size = size;
							temp.length = abs(botNode.x - topNode.x);
							mouVec.push_back(temp);
							cout << "Push Mouth object" << endl;
						}
						else {
							int minIndex = 0, minValue = 99999;

							if (length > mouVec[0].length) {
								mouthInfo temp;
								temp.botNode = botNode;
								temp.topNode = topNode;
								temp.size = size;
								temp.length = abs(botNode.x - topNode.x);
								mouVec[0] = temp;
								cout << "Replace Mouth object" << endl;
							}
						}
					}

				}
				else {
					//cout << "Abandon mouth object" << endl;
					//copyObject(orginal2, dst, i, j);
				}
			}
		}
	}
	//Draw rectangle on eyes.
	for (int i = 0; i < mouVec.size(); i++) {
		rectangle(result, mouVec[i].topNode, mouVec[i].botNode, Scalar(0, 255, 0), 1, 1, 0);
	}

	return result;
}

Mat FaceTools::getNoseArea(Mat &src, vector<eyeInfo> &eyeVec, vector<mouthInfo> &mouVec, Point &border) {
	Mat srcClone = src.clone();
	cout << "Get nose Area start" << endl;

	if (eyeVec.size() == 2 && mouVec.size() == 1) {
		if (eyeVec[0].pupil.y > eyeVec[1].pupil.y)
			border.y = eyeVec[1].pupil.y;
		else
			border.y = eyeVec[0].pupil.y;
		if (eyeVec[0].pupil.x > eyeVec[1].pupil.x)
			border.x = getBoundValue(src, eyeVec[1].pupil.x - noseAreaRange, 2);
		else
			border.x = eyeVec[0].pupil.x - noseAreaRange;
		cout << "1111" << endl;
		srcClone = srcClone(eyeVec[0].pupil.y > eyeVec[1].pupil.y ? 
			Range(eyeVec[1].pupil.y, mouVec[0].centerNode.y) : Range(eyeVec[0].pupil.y, mouVec[0].centerNode.y), 
			eyeVec[0].pupil.x + noseAreaRange > eyeVec[1].pupil.x ?
			Range(getBoundValue(src, eyeVec[1].pupil.x - noseAreaRange, 2), getBoundValue(src, eyeVec[0].pupil.x + noseAreaRange, 2))
			: Range(getBoundValue(src, eyeVec[0].pupil.x - noseAreaRange, 2), getBoundValue(src, eyeVec[1].pupil.x + noseAreaRange, 2)));
		cout << "2222" << endl;
	}
	else if (eyeVec.size() == 1 && mouVec.size() == 1) {
		if (eyeVec[0].pupil.y > mouVec[0].centerNode.y)
			border.y = mouVec[0].centerNode.y;
		else
			border.y = eyeVec[0].pupil.y;
		if (eyeVec[0].pupil.x > mouVec[0].centerNode.x)
			border.x = getBoundValue(src, mouVec[0].centerNode.x - noseAreaRange, 2);
		else
			border.x = eyeVec[0].pupil.x;
		cout << "3333" << endl;
		srcClone = srcClone(eyeVec[0].pupil.y > mouVec[0].centerNode.y ?
			Range(mouVec[0].centerNode.y, eyeVec[0].pupil.y) : Range(eyeVec[0].pupil.y, mouVec[0].centerNode.y),
			eyeVec[0].pupil.x + noseAreaRange > mouVec[0].centerNode.x ?
			Range(getBoundValue(src, mouVec[0].centerNode.x - noseAreaRange, 2), eyeVec[0].pupil.x) 
			: Range(eyeVec[0].pupil.x, getBoundValue(src, mouVec[0].centerNode.x + noseAreaRange, 2)));
		cout << "4444" << endl;
	}

	return srcClone;
}

Mat FaceTools::getExactNose(Mat &src, vector<noseInfo> &noseVec, int threshold) {
	cout << "Get exact nose start" << endl;
	Mat srcClone = src.clone();
	Mat srcClone2 = src.clone();
	Mat result = src.clone();

	//imshow("mouth area", src);
	//Change the image from RGB to gray format and apply equalization
	Mat srcImage = src.clone();
	//Mat srcImage = src_gray.clone();
	//cvtColor(srcImage, srcImage, CV_RGB2GRAY);
	//equalizeHist(srcImage, srcImage);

	noseVec.clear();
	int judgeM[1000][1000];
	memset(judgeM, 0, sizeof(judgeM));
	int size = 0;
	if (srcClone.rows <= 0 || srcClone.cols <= 0) {
		cout << "Illegal size of face area." << endl;
		exit(1);
	}
	for (int i = 0; i < srcClone.rows; i++) {
		for (int j = 0; j < srcClone.cols; j++) {
			if (srcClone.at<uchar>(i, j) == 0){
				//cout << "Eye Pos: " << i << " " << j << endl;
				size = 0;
				this->getObjectSize(srcClone, 2, i, j, size, judgeM);
				if (size > 0)
					cout << "current NoseObject size: " << size << endl;
				//If object size is less than 100 * 100, just filter it.
				if (size > threshold && size < noseMaxSize) {
					Point topNode(j, i);
					Point botNode(0, 0);
					cout << "Pos: " << i << " " << j << endl;
					cout << "current NoseObject size: " << size << endl;
					this->scanObject(srcClone, 1, i, j, topNode);
					this->scanObject(srcClone2, 2, i, j, botNode);
					int centerX = (topNode.x + botNode.x) / 2;
					int centerY = (topNode.y + botNode.y) / 2;
					int height = abs(topNode.y - botNode.y);
					int width = abs(topNode.x - botNode.x);
					rectangle(result, topNode, botNode, Scalar(0, 255, 0), 1, 1, 0);
					if (centerY >= nosePosRange && centerY <= srcClone.rows - nosePosRange 
						&& centerX >= nosePosRange && centerX <= srcClone.cols - nosePosRange
						&& height <= 20 && width <= 20) {
						//if (width >= 40 && height <= 30) {
						int size = abs((botNode.x - topNode.x) * (botNode.y - topNode.y));
						int length = abs(botNode.x - topNode.x);
						if (noseVec.size() == 0) {
							noseInfo temp;
							temp.botNode = botNode;
							temp.topNode = topNode;
							temp.centerNode = Point(centerX, centerY);
							temp.size = size;
							temp.length = abs(botNode.x - topNode.x);
							noseVec.push_back(temp);
							cout << "Push Nose object" << endl;
						}
						else if (noseVec.size() == 1) {
							noseInfo temp;
							temp.botNode = botNode;
							temp.topNode = topNode;
							temp.centerNode = Point(centerX, centerY);
							temp.size = size;
							//The distance between two eyes can not be less than a value
							int dis = sqrt(pow(noseVec[0].centerNode.x - centerX, 2) + pow(noseVec[0].centerNode.y - centerY, 2));
							cout << "Object Distance " << dis << endl;
							if (dis <= noseDistance)
								noseVec.push_back(temp);
						}
						else {
							
						}
					}

				}
				else {
					//copyObject(orginal2, dst, i, j);
				}
			}
		}
	}
	//Draw rectangle on nose.
	for (int i = 0; i < noseVec.size(); i++) {
		rectangle(result, noseVec[i].topNode, noseVec[i].botNode, Scalar(0, 255, 0), 1, 1, 0);
	}

	return result;
}

Mat FaceTools::getExactNoseGradient(Mat &src, vector<noseInfo> &noseVec, int threshold) {
	cout << "Get exact nose by gradient start" << endl;
	Mat srcClone = src.clone();
	Mat srcClone2 = src.clone();
	Mat result = src.clone();
	Mat srcImage = src.clone();
	Point nosePoint(0, 0);
	cout << "Start Gaussian Blur" << endl;
	cvtColor(srcClone, srcClone, CV_BGR2GRAY);
	GaussianBlur(srcClone, srcClone, Size(9, 9), 2, 2);

	int judgeM[1000][1000];
	memset(judgeM, 0, sizeof(judgeM));
	int size = 0;
	if (srcClone.rows <= 0 || srcClone.cols <= 0) {
		cout << "Illegal size of face area." << endl;
		exit(1);
	}
	//Record the max magnitude in the mat
	double maxMag;
	Mat gradientX = computeMatGradient(srcClone);
	Mat gradientY = computeMatGradient(srcClone.t()).t();
	Mat out = matrixMagnitude(gradientX, gradientY, maxMag, 2);
	Mat peakOut = findPeakPoint(srcClone, out, maxMag / 2, 90, nosePoint);
	imshow("Nose Before", srcClone);
	imshow("Nose After", peakOut);
	//waitKey();

	noseInfo temp;
	temp.botNode = nosePoint;
	temp.topNode = nosePoint;
	temp.centerNode = nosePoint;
	temp.size = 0;
	temp.length = 0;
	noseVec.push_back(temp);
	cout << "Push Nose object" << endl;

	return result;
}

int FaceTools::assumeNose(vector<noseInfo> &noseVec, Point eyeP, Point mouthP) {
	if (noseVec.size() != 1)
		return 0;
	else {
		float a = 0, b = 0, exp = 0;
		int divM = eyeP.x - mouthP.x;
		if (divM == 0) {
			b = - eyeP.x;
			exp = noseVec[0].centerNode.x + b;
			if (exp > 0)
				moveNose(noseVec, 2);
			else
				moveNose(noseVec, 1);
		}
		else {
			a = (float)(eyeP.y - mouthP.y) / divM;
			b = (float)(mouthP.y - mouthP.x * a);
			exp = noseVec[0].centerNode.y - a * noseVec[0].centerNode.x - b;
			if ((a >= 0 && exp >= 0) || (a < 0 && exp < 0))
				moveNose(noseVec, 1);
			if ((a >= 0 && exp < 0) || (a < 0 && exp >= 0))
				moveNose(noseVec, 2);
		}
	}
	return 1;
}

int FaceTools::moveNose(vector<noseInfo> &noseVec, int type) {
	if (type == 1) {
		noseVec[0].botNode.x -= noseVectorX;
		noseVec[0].topNode.x -= noseVectorX;
		noseVec[0].centerNode.x -= noseVectorX;
		noseVec[0].botNode.y -= noseVectorY;
		noseVec[0].topNode.y -= noseVectorY;
		noseVec[0].centerNode.y -= noseVectorY;
	}
	else if (type == 2) {
		noseVec[0].botNode.x += noseVectorX;
		noseVec[0].topNode.x += noseVectorX;
		noseVec[0].centerNode.x += noseVectorX;
		noseVec[0].botNode.y -= noseVectorY;
		noseVec[0].topNode.y -= noseVectorY;
		noseVec[0].centerNode.y -= noseVectorY;
	}
	else
		return 0;
	return 1;
}

//Calculate the coefficient of inclined ratio by using formula:
//p = 
int FaceTools::calculateFace(Mat &src, Mat &eyeBin, vector<eyeInfo> &eyeVec, vector<mouthInfo> &mouVec, Point noseCenter) {
	if (src.rows <= 0 || src.cols <= 0) {
		cout << "Invalid source exception." << endl;
		return 0;
	}

	String msg, msg2, msg3, msg4, msg5;
	float gradient_face = 0, inclined_face = 0, eye_size_com = 0, eye_offset = 0, nose_offset = 0, dis_nose = 0, dis_eyes = 0;
	float concentration = 0;
	int angle_face, divided;
	Point eyeCenter;
	ostringstream os;

	//If both eyes and mouth are detected, draw a triangle
	if (eyeVec.size() == 2 && mouVec.size() == 1) {
		//Change the nodes' positions if necessary
		//changeMouPosition(mouVec, 20, 20);
		//Calculate head angle
		msg = "Face Angle: ";
		eyeCenter.x = (eyeVec[0].pupil.x + eyeVec[1].pupil.x) / 2;
		eyeCenter.y = (eyeVec[0].pupil.y + eyeVec[1].pupil.y) / 2;
		divided = eyeCenter.x - mouVec[0].centerNode.x;
		if (divided == 0) {
			angle_face = 0;
		}
		else {
			gradient_face = abs((eyeCenter.y - mouVec[0].centerNode.y) / (eyeCenter.x - mouVec[0].centerNode.x));
			angle_face = abs(90 - atan(gradient_face) / PI * 180);
		}
		os << angle_face;
		msg += os.str();

		//Calculate face angle
		/*cout << "Line: " << sqrt(pow((eyeVec[0].pupil.y - mouVec[0].centerNode.y), 2) + pow((eyeVec[0].pupil.x - mouVec[0].centerNode.x), 2)) << endl;
		cout << "Line2: " << sqrt(pow((eyeVec[1].pupil.y - mouVec[0].centerNode.y), 2) + pow((eyeVec[1].pupil.x - mouVec[0].centerNode.x), 2)) << endl;
		inclined_face = (sqrt(pow((eyeVec[0].pupil.y - mouVec[0].centerNode.y), 2) + pow((eyeVec[0].pupil.x - mouVec[0].centerNode.x), 2))) 
			/ (sqrt(pow((eyeVec[1].pupil.y - mouVec[0].centerNode.y), 2) + pow((eyeVec[1].pupil.x - mouVec[0].centerNode.x), 2)));

		cout << "Inclined Rate: " << inclined_face << endl;

		os.str("");
		msg2 = "Inclined Rate: ";
		os << inclined_face;
		msg2 += os.str();*/

		//Calculate the offset of the center of eyes
		int centerY = (eyeVec[0].cenNode.y + eyeVec[1].cenNode.y) / 2;
		int dis_l = 0, dis_r = 0, dis = src.cols;
		for (int i = 0; i < eyeBin.cols; i++) {
			if (eyeBin.at<uchar>(centerY, i) == 0)
				continue;
		}
		eye_offset = (float)abs(eyeVec[0].cenNode.x + eyeVec[1].cenNode.x - src.cols) / src.cols;
		cout << "The offset of eyes: " << eye_offset << endl;

		os.str("");
		msg3 = "Eyes Offset: ";
		os << eye_offset;
		msg3 += os.str();

		//Calculate the horizontal inclined ratio of the face according to nose
		int nose_div = eyeCenter.x - mouVec[0].centerNode.x;
		if (nose_div != 0) {
			float a = (float)(eyeCenter.y - mouVec[0].centerNode.y) / (eyeCenter.x - mouVec[0].centerNode.x);
			float b = (float)(eyeCenter.y - a * eyeCenter.x);
			dis_nose = (float)abs((a * noseCenter.x - noseCenter.y + b) / (sqrt(a * a + 1)));
		}
		else {
			dis_nose = (float)abs((noseCenter.x - eyeCenter.x));
		}
		dis_eyes = (float)sqrt(pow((eyeVec[0].pupil.y - eyeVec[1].pupil.y), 2) + pow((eyeVec[0].pupil.x - eyeVec[1].pupil.x), 2)) / 2;
		nose_offset = dis_nose / dis_eyes;

		cout << "The offset of nose: " << dis_nose << endl;

		os.str("");
		msg4 = "Nose Offset: ";
		os << nose_offset;
		msg4 += os.str();

		//Calculate the final ratio of concentration
		concentration = (float)angle_face / 90 + nose_offset;

		cout << "Distraction: " << concentration << endl;

		os.str("");
		msg5 = "Distraction: ";
		os << concentration;
		msg5 += os.str();
	}
	//If just one eye and mouth are detected, just link them
	else if (eyeVec.size() == 1 && mouVec.size() == 1) {

	}

	putText(src, msg, Point(10, 20), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));
	//putText(src, msg2, Point(10, 40), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));
	//putText(src, msg3, Point(10, 60), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));
	putText(src, msg4, Point(10, 40), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));
	putText(src, msg5, Point(10, 60), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));

	return 1;
}

int FaceTools::changeMouPosition(vector<mouthInfo> &mouVec, int x, int y) {
	if (mouVec.size() != 1)
		return 0;
	else {
		mouVec[0].centerNode.x += x;
		mouVec[0].centerNode.y += y;
	}
	return 1;
}

int FaceTools::findMass(Mat &src) {
	if (src.cols <= 0 || src.rows <= 0)
		return 0;
	else {
		int massTotalX = 0, massTotalY = 0, massNum = 0;
		int centerX = 0, centerY = 0;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.ptr<Vec3b>(i)[j] == Vec3b(100, 100, 100)){
					massTotalX += j;
					massTotalY += i;
					massNum++;
				}
			}
		}

		cout << "massNum: " << massNum << endl;
		cout << "mass x: " << massTotalX << " mass y: " << massTotalY << endl;
		centerX = massTotalX / massNum;
		centerY = massTotalY / massNum;
		cout << "mass x: " << centerX << " mass y: " << centerY << endl;

		circle(src, Point(centerX, centerY), 3, 1234);
	}
	return 1;
}

void FaceTools::drawFacialFeatures(Mat &src, Mat &faceBin, vector<eyeInfo> &eyeVec, vector<mouthInfo> &mouVec, Point &noseCenter) {
	cout << "Draw facial features start" << endl;
	vector<noseInfo> noseVec;
	Point eyeCenter, noseStart;
	Mat graySrc = src.clone();
	//Change coordinates of eyes from eye's area to face's area
	for (int i = 0; i < eyeVec.size(); i++) {
		eyeVec[i].pupil.x += src.cols * feaSearchColStartRatio;
		eyeVec[i].pupil.y += src.rows * eyeSearchRowStartRatio;
		circle(src, eyeVec[i].pupil, 3, 1234);
		//imshow("eye1", result);
	}
	//Change the coordinate of mouth from mouth's area to face's area
	for (int i = 0; i < mouVec.size(); i++) {
		mouVec[i].topNode.x += src.cols * feaSearchColStartRatio;
		mouVec[i].topNode.y += src.rows * mouthSearchRowStartRatio;
		mouVec[i].botNode.x += src.cols * feaSearchColStartRatio;
		mouVec[i].botNode.y += src.rows * mouthSearchRowStartRatio;
		mouVec[i].centerNode.x = (mouVec[i].topNode.x + mouVec[i].botNode.x) / 2;
		mouVec[i].centerNode.y = (mouVec[i].topNode.y + mouVec[i].botNode.y) / 2;
		//rectangle(result, mouVec[i].topNode, mouVec[i].botNode, Scalar(0, 255, 0), 1, 1, 0);
	}
	//Get the nose area by using positions of eyes and mouth
	Mat noseArea = getNoseArea(faceBin, eyeVec, mouVec, noseStart);
	Mat noseAreaRGB = getNoseArea(src, eyeVec, mouVec, noseStart);

	cvtColor(graySrc, graySrc, CV_BGR2GRAY);
	////equalizeHist(noseAreaRGB, noseAreaRGB);
	//imshow("222", noseAreaRGB);
	//Mat dst_x, dst_y, dst;
	//Sobel(noseAreaRGB, dst_x, noseAreaRGB.depth(), 1, 0);
	//Sobel(noseAreaRGB, dst_y, noseAreaRGB.depth(), 0, 1);
	//imshow("xxx", dst_x);
	//imshow("yyy", dst_y);

	//Get the nose detection result
	//Mat noseResult = getExactNose(noseArea, noseVec, noseMinSize);
	//imshow("Nose src", src);
	float variance = getVariance(graySrc, graySrc.rows * noseStartRatio, graySrc.rows * noseEndRatio);
	getExactNoseGradient(noseAreaRGB, noseVec, noseMinSize);
	//Mat faceBorder = getSobelBorder(noseAreaRGB);
	//imshow("Face Border", faceBorder);
	//waitKey();
	imshow("nose area2", noseArea);
	//imshow("Nose", noseResult);
	
	//Draw the nose
	if (noseVec.size() == 0) {
		cout << "No nose founded!!!" << endl;
		if (mouVec.size() > 0) {
			noseInfo temp;
			temp.botNode = Point(mouVec[0].botNode.x, mouVec[0].botNode.y - 40);
			temp.topNode = Point(mouVec[0].topNode.x, mouVec[0].topNode.y - 40);
			temp.size = 100;
			temp.centerNode = Point((mouVec[0].topNode.x + mouVec[0].botNode.x) / 2, (mouVec[0].topNode.y + mouVec[0].botNode.y - 80) / 2);
			temp.length = abs(temp.botNode.x - temp.topNode.x);
			noseVec.push_back(temp);
		}
	}
	else {
		cout << noseVec.size() << " noses founded!!!" << endl;
		for (int i = 0; i < noseVec.size(); i++) {
			noseVec[i].topNode.x += noseStart.x;
			noseVec[i].topNode.y += noseStart.y;
			noseVec[i].botNode.x += noseStart.x;
			noseVec[i].botNode.y += noseStart.y;
			/*noseVec[i].topNode.x += frame.cols * feaSearchColStartRatio;
			noseVec[i].topNode.y += frame.rows * noseSearchRowStartRatio;
			noseVec[i].botNode.x += frame.cols * feaSearchColStartRatio;
			noseVec[i].botNode.y += frame.rows * noseSearchRowStartRatio;*/
			noseVec[i].centerNode.x = (noseVec[i].topNode.x + noseVec[i].botNode.x) / 2;
			noseVec[i].centerNode.y = (noseVec[i].topNode.y + noseVec[i].botNode.y) / 2;
			//rectangle(result, mouVec[i].topNode, mouVec[i].botNode, Scalar(0, 255, 0), 1, 1, 0);
		}
	}
	//If both eyes and mouth are detected, draw a triangle
	//changeMouPosition(mouVec, -18, 8);
	if (eyeVec.size() == 2 && mouVec.size() == 1) {
		eyeCenter.x = (eyeVec[0].pupil.x + eyeVec[1].pupil.x) / 2;
		eyeCenter.y = (eyeVec[0].pupil.y + eyeVec[1].pupil.y) / 2;
		line(src, mouVec[0].centerNode, eyeCenter, Scalar(0, 255, 0), 1, 1, 0);
		line(src, eyeVec[0].pupil, eyeVec[1].pupil, Scalar(0, 255, 0), 1, 1, 0);
		line(src, mouVec[0].centerNode, eyeVec[0].pupil, Scalar(0, 255, 0), 1, 1, 0);
		line(src, mouVec[0].centerNode, eyeVec[1].pupil, Scalar(0, 255, 0), 1, 1, 0);
		if (noseVec.size() == 2) {
			noseCenter.x = (noseVec[0].centerNode.x + noseVec[1].centerNode.x) / 2;
			noseCenter.y = (noseVec[0].centerNode.y + noseVec[1].centerNode.y) / 2;
			line(src, noseCenter, eyeVec[0].pupil, Scalar(0, 0, 255), 1, 1, 0);
			line(src, noseCenter, eyeVec[1].pupil, Scalar(0, 0, 255), 1, 1, 0);
			line(src, noseCenter, mouVec[0].centerNode, Scalar(0, 0, 255), 1, 1, 0);
		}
		else {
			//assumeNose(noseVec, eyeCenter, mouVec[0].centerNode);
			noseCenter = noseVec[0].centerNode;
			line(src, noseCenter, eyeVec[0].pupil, Scalar(0, 0, 255), 1, 1, 0);
			line(src, noseCenter, eyeVec[1].pupil, Scalar(0, 0, 255), 1, 1, 0);
			line(src, noseCenter, mouVec[0].centerNode, Scalar(0, 0, 255), 1, 1, 0);
		}
	}
	//If just one eye and mouth are detected, just link them
	else if (eyeVec.size() == 1 && mouVec.size() == 1) {
		//line(result, eyeVec[0].pupil, eyeVec[1].pupil, Scalar(0, 255, 0), 1, 1, 0);
		line(src, mouVec[0].centerNode, eyeVec[0].pupil, Scalar(0, 255, 0), 1, 1, 0);
		if (noseVec.size() == 2) {
			noseCenter.x = (noseVec[0].centerNode.x + noseVec[1].centerNode.x) / 2;
			noseCenter.y = (noseVec[0].centerNode.y + noseVec[1].centerNode.y) / 2;
			line(src, noseCenter, eyeVec[0].pupil, Scalar(0, 0, 255), 1, 1, 0);
			line(src, noseCenter, mouVec[0].centerNode, Scalar(0, 0, 255), 1, 1, 0);
		}
		else {
			//assumeNose(noseVec, eyeCenter, mouVec[0].centerNode);
			noseCenter = noseVec[0].centerNode;
			line(src, noseCenter, eyeVec[0].pupil, Scalar(0, 0, 255), 1, 1, 0);
			line(src, noseCenter, mouVec[0].centerNode, Scalar(0, 0, 255), 1, 1, 0);
		}
	}
}

int FaceTools::getBoundValue(Mat src, int range, int type) {
	if (type == 1) {
		if (range >= src.rows - 1)
			return src.rows - 1;
		else if (range <= 0)
			return 0;
	}
	else if (type == 2) {
		if (range >= src.cols - 1)
			return src.cols - 1;
		else if (range <= 0)
			return 0;
	}
	return range;
}

Mat FaceTools::computeMatGradient(const cv::Mat &mat) {
	cv::Mat out(mat.rows, mat.cols, CV_64F);

	for (int y = 0; y < mat.rows; ++y) {
		const uchar *Mr = mat.ptr<uchar>(y);
		double *Or = out.ptr<double>(y);

		Or[0] = Mr[1] - Mr[0];
		for (int x = 1; x < mat.cols - 1; ++x) {
			Or[x] = (Mr[x + 1] - Mr[x - 1]) / 2.0;
		}
		Or[mat.cols - 1] = Mr[mat.cols - 1] - Mr[mat.cols - 2];
	}

	return out;
}

Mat FaceTools::matrixMagnitude(const Mat &matX, const Mat &matY, double &maxMag, int type) {
	cv::Mat mags(matX.rows, matX.cols, CV_64F);
	maxMag = 0;
	double startRatio, endRatio;
	if (type == 1) {
		startRatio = eyeStartRatio;
		endRatio = eyeEndRatio;
	}
	else if (type == 2) {
		startRatio = noseStartRatio;
		endRatio = noseEndRatio;
	}
	for (int y = matX.rows * startRatio; y < matX.rows * endRatio; ++y) {
		const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
		double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < matX.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			double magnitude = sqrt((gX * gX) + (gY * gY));
			Mr[x] = magnitude;
			maxMag = magnitude > maxMag ? magnitude : maxMag;
		}
	}
	cout << "Max magnitude: " << maxMag << endl;
	return mags;
}

Mat FaceTools::findPeakPoint(const Mat &src, const Mat &grad, double threshold, int grayThres, Point &nosePoint) {
	Mat result = src.clone();
	int count = 0, centerX = 0, centerY = 0;
	for (int i = 0; i < grad.rows; i++) {
		for (int j = 0; j < grad.cols; j++) {
			//if (grad.ptr<double>(i)[j] >= threshold && src.at<uchar>(i, j) - 0 <= grayThres){
			if (grad.ptr<double>(i)[j] >= threshold && src.at<uchar>(i, j) - 0 <= grayThres
				&& i > src.rows * noseStartRatio && i < src.rows * noseEndRatio){
				//cout << src.at<uchar>(i, j) - 0 << " " << grad.ptr<double>(i)[j] << endl;
				centerX += j;
				centerY += i;
				count++;
				circle(result, Point(j, i), 3, 1234);
			}
		}
	}

	cout << "Nose Pixel Num: " << count << endl;
	nosePoint.x = count > 0 ? centerX / count : src.cols / 2;
	nosePoint.y = count > 0 ? centerY / count : src.rows / 2;
	//circle(result, nosePoint, 3, 1234);
	return result;
}

Mat FaceTools::findPeakPointEyes(const Mat &src, const Mat &grad, double threshold, int grayThres, Point &nosePoint) {
	Mat result = src.clone();
	int count = 0, centerX = 0, centerY = 0;
	for (int i = 0; i < grad.rows; i++) {
		for (int j = 0; j < grad.cols; j++) {
			//if (grad.ptr<double>(i)[j] >= threshold && src.at<uchar>(i, j) - 0 <= grayThres){
			if (grad.ptr<double>(i)[j] >= threshold && src.at<uchar>(i, j) - 0 <= grayThres
				&& i > src.rows * eyeStartRatio && i < src.rows * eyeEndRatio){
				//cout << src.at<uchar>(i, j) - 0 << " " << grad.ptr<double>(i)[j] << endl;
				centerX += j;
				centerY += i;
				count++;
				circle(result, Point(j, i), 3, 1234);
			}
		}
	}

	cout << "Nose Pixel Num: " << count << endl;
	nosePoint.x = count > 0 ? centerX / count : src.cols / 2;
	nosePoint.y = count > 0 ? centerY / count : src.rows / 2;
	//circle(result, nosePoint, 3, 1234);
	return result;
}

int FaceTools::findFaceInDB(char dbPath[256]) {
	// Images variables
	char imgDir[256];
	char imgFile[64];
	char txtFile[64];
	FILE *ptFile;
	unsigned int xt1, yt1, xt2, yt2;

	// Label variables
	char name[64];
	char prop[64];
	unsigned int x, y, w, h;

	// Database installation directory
	strcpy(imgDir, dbPath);
	cout << imgDir << endl;

	// Create window and display results
	cvNamedWindow("Source Image", 1);

	//File
	Mat src;
	// Apply face detection on database images
	int numPers, numSer = 1;
	int i, pan, tilt = 0;
	int panIndex, tiltIndex;
	char*  panPlus;
	char* tiltPlus;
	for (numPers = 1; numPers <= 15; numPers++) {
		for (numSer = 1; numSer <= 2; numSer++) {
			for (i = 0; i < 93; i++) {
				panPlus = ""; tiltPlus = "";

				// Retrieve pan and tilt angles
				if (i == 0) {
					tilt = -90; pan = 0;
				}
				else if (i == 92) {
					tilt = 90; pan = 0;
				}
				else {
					pan = ((i - 1) % 13 - 6) * 15;
					tilt = ((i - 1) / 13 - 3) * 15;
					if (abs(tilt) == 45) tilt = tilt / abs(tilt) * 60;
				}

				// Add "+" before positive angles    
				if (pan >= 0)  panPlus = "+";
				if (tilt >= 0) tiltPlus = "+";

				// Build image file path and load image
				sprintf(imgFile, "%sPerson%02i/person%02i%i%02i%s%i%s%i.jpg",
					imgDir, numPers, numPers, numSer, i, tiltPlus, tilt, panPlus, pan);
				printf("Processing %s\n", imgFile);
				//image = cvLoadImage(imgFile, 1);
				Mat src = imread(imgFile);




				/********************
				*   DO                                            *
				*            YOUR                                 *
				*                        PROCESS                  *
				*                                     HERE        *
				******************/
				if (!src.empty()){
					Mat frame = src.clone(), finalresult = src.clone(), faceArea;
					Mat faceBorder = getSobelBorder(frame);

					cvtColor(frame, grayframe, CV_BGR2GRAY);
					equalizeHist(grayframe, testframe);
					Mat thres_lab = this->GetSkin(frame, testframe);
					Mat testframe2 = this->maskOnImage(thres_lab, frame);
					Mat testframe3, testframe4, eyeSkin;
					erode(testframe2, testframe3, Mat(5, 5, CV_8U), Point(-1, -1), 2);
					dilate(testframe3, testframe4, Mat(5, 5, CV_8U), Point(-1, -1), 2);
					Mat orginal2 = testframe4.clone();
					Mat testframe5 = testframe4.clone();
					for (int i = 0; i < testframe5.rows; i++) {
						for (int j = 0; j < testframe5.cols; j++) {
							testframe5.ptr<Vec3b>(i)[j] = Vec3b(100, 100, 100);
						}
					}
					this->processImage(testframe4, testframe5);
					//imshow("7777", testframe5);
					this->findFace(testframe5, frame, faceArea);
					//imwrite(outputpath + "TotalFaceAvg" + temp + ".jpg", norm_0_255(mean.reshape(1, db[0].rows)));
					this->findMass(testframe5);
					this->findFacialFeatures(faceArea, eyeSkin, faceArea);

					vector<int> compression_params;
					compression_params.push_back(CV_IMWRITE_PXM_BINARY);
					compression_params.push_back(1);

					imwrite(outputpath + "faceGet_Test.pgm", faceArea, compression_params);
					//imwrite(outputpath + "faceGet_Test.jpg", faceArea, CV_IMWRITE_PXM_BINARY);

					imshow("original", src);
					//imshow("gray222", grayframe);
					//imshow("gray", testframe);
					//imshow("face", orginal2);
					//imshow("operated", testframe5);
					//imshow("find", frame);
					imshow("result", faceArea);
					waitKey();
				}
				else{
					printf(" No input frame detected.");
					return 0;
				}
			}
		}
	}

	return 0;
}

float FaceTools::getVariance(const Mat &src, int start, int end) {
	float result = 0, expectation = 0;
	int total = abs(end - start) * src.cols;
	for (int i = start; i < end; i++) {
		for (int j = 0; j < src.cols; j++) {
			expectation += (float)(src.at<uchar>(i, j) - 0) / total;
		}
	}

	cout << "Expectation: " << expectation << endl;
	for (int i = start; i < end; i++) {
		for (int j = 0; j < src.cols; j++) {
			result += (float)(src.at<uchar>(i, j) - 0 - expectation) *
				(src.at<uchar>(i, j) - 0 - expectation) / total;
		}
	}

	result = sqrt(result);
	cout << "Variance: " << result << endl;
	return result;
}

Mat FaceTools::getHoughCircles(const Mat &src, int minRadius, int maxRadius) {
	Mat grayFrame = src.clone();
	cvtColor(grayFrame, grayFrame, CV_BGR2GRAY);
	//GaussianBlur(grayFrame, grayFrame, Size(9, 9), 2, 2);
	//Use Canny operator to do border detection
	Canny(grayFrame, grayFrame, cannyLowThres, cannyHighThres, 3);

	vector<Vec3f> circles;
	HoughCircles(grayFrame, circles, CV_HOUGH_GRADIENT, 1.5, 10, 50, 25, minRadius, maxRadius);

	//Draw Circle
	cout << "circle size : " << circles.size() << endl;
	imshow("Gray Face Formal", grayFrame);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		//Draw center of circle
		circle(grayFrame, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		//Draw outline of circle
		circle(grayFrame, center, radius, Scalar(155, 50, 255), 3, 8, 0);
	}
	imshow("Gray Face", grayFrame);

	return grayFrame;
}

Mat FaceTools::getContoursByCplus(const Mat &src, int mode, double minarea, double whRatio) {
	Mat srcClone = src.clone(), dst, canny_output;
	if (!srcClone.data)
	{
		std::cout << "read data error!" << std::endl;
		return dst;
	}
	blur(srcClone, srcClone, Size(3, 3));


	//the pram. for findContours,  
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// Detect edges using canny
	//Canny(srcClone, canny_output, 80, 255, 3);
	Canny(srcClone, canny_output, cannyLowThres, cannyHighThres, 3);
	//imshow("canny", canny_output);
	// Find contours
	if (mode == 2) {
		//findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		//CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE  

		double maxarea = 0;
		int maxAreaIdx = 0;

		for (int i = 0; i < contours.size(); i++)
		{

			double tmparea = fabs(contourArea(contours[i]));
			if (tmparea > maxarea)
			{
				maxarea = tmparea;
				maxAreaIdx = i;
				continue;
			}

			if (tmparea < minarea)
			{
				//Delete those area whose size smaller than minarea
				contours.erase(contours.begin() + i);
				std::wcout << "delete a small area" << std::endl;
				continue;
			}
			//Calculate the contour's width, height
			Rect aRect = boundingRect(contours[i]);
			if ((aRect.width / aRect.height) < whRatio)
			{
				//Delete those contour whose ratio of width and height is smaller than specified value
				contours.erase(contours.begin() + i);
				std::wcout << "delete a unnomalRatio area" << std::endl;
				continue;
			}
		}
		// Draw contours
		dst = Mat::zeros(canny_output.size(), CV_8UC3);
		RNG rng;
		for (int i = 0; i < contours.size(); i++)
		{
			//Random Color
			Scalar color = Scalar(rng.uniform(0, 0), rng.uniform(255, 255), rng.uniform(0, 0));
			drawContours(dst, contours, i, color, 1, 8, hierarchy, 0, Point());
		}
		// Create Window  
		char* source_window = "countors";
		namedWindow(source_window, CV_WINDOW_NORMAL);
		imshow(source_window, dst);
		waitKey(0);

		return dst;
	}
	else if (mode == 0) {
		//string inputFile = "E:\\faceTemplate\\HeadPoseImageDatabase\\Person14\\person14170+30-30.jpg"; // Create Memory 
		CvMemStorage *storage = cvCreateMemStorage(0);; //Load image from disk 
		//IplImage *img = cvLoadImage(inputFile.c_str(), 0);
		CvSeq *seq = 0;
		CvSeq *pConInner = NULL;
		double dConArea;

		IplImage *tmp = &IplImage(canny_output);
		//Create contour chains
		cvFindContours(tmp, storage, &seq, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_CODE, cvPoint(0, 0));
		//Create approximated Freeman chains
		seq = cvApproxChains(seq, storage, CV_CHAIN_APPROX_SIMPLE, 0, 0, 0);

		cvDrawContours(tmp, seq, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 2, CV_FILLED, 8, cvPoint(0, 0));
		int wai = 0;
		int nei = 0;
		//Outer contours
		for (; seq != NULL; seq = seq->h_next)
		{
			wai++;
			//Inner contours
			/*for (pConInner = seq->v_next; seq != NULL; seq = seq->h_next)
			{
				nei++;
				dConArea = fabs(cvContourArea(pConInner, CV_WHOLE_SEQ));
				printf("%f\n", dConArea);
			}*/
			CvRect rect = cvBoundingRect(seq, 0);
			//cvRectangle(tmp, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height), CV_RGB(255, 255, 255), 1, 8, 0);
		}

		cout << "wai: " << wai << endl;

		Mat tempMat(tmp, 0);

		//vector<Vec3f> circles;
		//HoughCircles(tempMat, circles, CV_HOUGH_GRADIENT, 1.5, 10, 50, 25, 1, 10);

		////Draw Circle
		//cout << "circle size : " << circles.size() << endl;
		//for (size_t i = 0; i < circles.size(); i++)
		//{
		//	Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		//	int radius = cvRound(circles[i][2]);
		//	//Draw center of circle
		//	circle(tempMat, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		//	//Draw outline of circle
		//	circle(tempMat, center, radius, Scalar(155, 50, 255), 3, 8, 0);
		//}

		imshow("Contours222", tempMat);
		cvReleaseMemStorage(&storage);
		storage = NULL;
		dst = tempMat;
		
		// Create Window  
		/*char* source_window = "countors";
		namedWindow(source_window, CV_WINDOW_NORMAL);
		imshow(source_window, dst);*/
		//waitKey(0);

		return dst;
	}
}

Mat FaceTools::getExactEyesGray(Mat &src, int threshold) {
	Mat srcClone = src.clone();

	//int count = 0, centerX = 0, centerY = 0;
	//for (int i = 0; i < grad.rows; i++) {
	//	for (int j = 0; j < grad.cols; j++) {
	//		//if (grad.ptr<double>(i)[j] >= threshold && src.at<uchar>(i, j) - 0 <= grayThres){
	//		if (grad.ptr<double>(i)[j] >= threshold && src.at<uchar>(i, j) - 0 <= grayThres
	//			&& i > src.rows * noseStartRatio && i < src.rows * noseEndRatio){
	//			//cout << src.at<uchar>(i, j) - 0 << " " << grad.ptr<double>(i)[j] << endl;
	//			centerX += j;
	//			centerY += i;
	//			count++;
	//			circle(result, Point(j, i), 3, 1234);
	//		}
	//	}
	//}

	//cout << "Nose Pixel Num: " << count << endl;
	//nosePoint.x = count > 0 ? centerX / count : src.cols / 2;
	//nosePoint.y = count > 0 ? centerY / count : src.rows / 2;
	////circle(result, nosePoint, 3, 1234);
	//imshow("Eye gray", result);

	double maxMag;
	Point nosePoint;
	Mat gradientX = computeMatGradient(srcClone);
	Mat gradientY = computeMatGradient(srcClone.t()).t();
	Mat out = matrixMagnitude(gradientX, gradientY, maxMag, 1);
	Mat result = findPeakPointEyes(srcClone, out, maxMag / 4, 100, nosePoint);
	imshow("Eye gray", result);

	return result;
}

Mat FaceTools::detectCornerPoints(Mat &src, Mat &dst, Point startP, vector<Point>&cornerVec) {
	Mat srcClone = src.clone();
	cornerVec.clear();

	//Added 20150517 start
	vector<Point2f> corners;
	double qualityLevel = 0.01;
	double minDistance = 15;
	int blockSize = 3;
	int maxCorners = 50;
	int maxCornersThresh = 100;
	bool useHarrisDetector = false;
	double k = 0.04;

	Mat detectSrcCopy = src.clone();
	Mat refineSrcCopy = src.clone();
	Mat srcGray;
	char* detectWindow = "detection";
	char* refineWindow = "refinement";
	cvtColor(srcClone, srcGray, CV_RGB2GRAY);

	goodFeaturesToTrack(srcGray,
		corners,
		maxCorners,
		qualityLevel,
		minDistance,
		Mat(),
		blockSize,
		useHarrisDetector,
		k);

	cout << "*  detected corners : " << corners.size() << endl;
	cout << "** max corners: " << maxCorners << endl;

	int r = 3;
	cout << "-- Before refinement: " << endl;
	for (int i = 0; i < corners.size(); i++)
	{
		circle(detectSrcCopy, corners[i], r, Scalar(255, 0, 255), -1, 8, 0);
		cout << "	[" << i << "]  (" << corners[i].x << "," << corners[i].y << ")" << endl;
	}
	//namedWindow(detectWindow, CV_WINDOW_AUTOSIZE);
	//imshow(detectWindow, detectSrcCopy);

	Size winSize = Size(5, 5);
	Size zeroZone = Size(-1, -1);
	TermCriteria criteria = TermCriteria(
		CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,
		40, //maxCount=40
		0.001);	//epsilon=0.001
	cornerSubPix(srcGray, corners, winSize, zeroZone, criteria);

	cout << "-- After refinement: " << endl;
	for (int i = 0; i < corners.size(); i++)
	{
		circle(refineSrcCopy, corners[i], r, Scalar(255, 0, 255), -1, 8, 0);
		cout << "	[" << i << "]  (" << corners[i].x << "," << corners[i].y << ")" << endl;
		//This part will draw the corner points on the original image from the camera.
		Point temp(corners[i].x + startP.x, corners[i].y + startP.y);
		circle(dst, temp, r, Scalar(255, 0, 255), -1, 8, 0);
		cornerVec.push_back(temp);
	}
	namedWindow(refineWindow, CV_WINDOW_AUTOSIZE);
	imshow(refineWindow, refineSrcCopy);
	//Added 20150517 end

	return srcClone;
}

Mat FaceTools::refineBinaryByCorner(Mat src, Mat &dst, vector<Point>&cornerVec, int threshold) {
	Mat srcClone = src.clone();
	//Erase the dst
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			dst.at<uchar>(i, j) = 255;
		}
	}

	int judgeM[1000][1000];
	memset(judgeM, 0, sizeof(judgeM));
	int size = 0;
	if (srcClone.rows <= 0 || srcClone.cols <= 0) {
		cout << "Illegal size of face area." << endl;
		exit(1);
	}
	for (int i = 0; i < srcClone.rows; i++) {
		for (int j = 0; j < srcClone.cols; j++) {
			if (srcClone.at<uchar>(i, j) == 0){
				//cout << "Eye Pos: " << i << " " << j << endl;
				int find = 0;
				isHaveCorner(srcClone, cornerVec, 255, judgeM, find, threshold, i, j);
				//cout << "Find at " << i << " " << j << " : " << find << endl;
				if (0 == find)
					eraseObject(srcClone, i, j, 1, 255);
				else if (find > threshold)
					copyObjectBin(srcClone, dst, i, j);
			}
		}
	}

	imshow("After Refine", dst);
	//waitKey();

	return srcClone;
}

int FaceTools::isHaveCorner(Mat src, vector<Point>&cornerVec, uchar a, int(&judge)[1000][1000], int &find, int threshold, int x, int y) {
	/*if (x < 0 || x >= src.rows - 1)
		return 0;
	if (y < 0 || y >= src.cols - 1)
		return 0;
	if (src.at<uchar>(x, y) == a) {
		return 0;
	}
	if (judge[x][y] == 1) {
		return 0;
	}

	judge[x][y] = 1;
	for (int i = 0; i < cornerVec.size(); i++) {
		if (x == cornerVec[i].y && y == cornerVec[i].x)
			find ++;
	}

	if (find > threshold)
		return 1;

	isHaveCorner(src, cornerVec, a, judge, find, threshold, x, y - 1);
	isHaveCorner(src, cornerVec, a, judge, find, threshold, x, y + 1);
	isHaveCorner(src, cornerVec, a, judge, find, threshold, x - 1, y);
	isHaveCorner(src, cornerVec, a, judge, find, threshold, x + 1, y);*/

	if (x < 0 || x >= src.rows - 1)
		return 0;
	if (y < 0 || y >= src.cols - 1)
		return 0;

	int i = x, j = y;
	queue<binaryPoint>que;
	binaryPoint start;
	start.i = x, start.j = y;
	que.push(start);
	judge[i][j] = 1;
	int *cornerList = new int[cornerVec.size()];
	memset(cornerList, 0, cornerVec.size() * sizeof(int));

	while (!que.empty()) {
		binaryPoint cur = que.front();
		que.pop();
		//cout << "Current: " << cur.i << " " << cur.j << endl;
		for (int i = 0; i < 4; i++) {
			binaryPoint temp = cur + movements[i];
			//cout << "Temp: " << temp.i << " " << temp.j << endl;
			if (temp.i >= 0 && temp.i < src.rows && temp.j >= 0 && temp.j < src.cols) {
				if (judge[temp.i][temp.j] == 0 && src.at<uchar>(temp.i, temp.j) != a) {
					que.push(temp);
					judge[temp.i][temp.j] = 1;
					for (int z = 0; z < cornerVec.size(); z++) {
						//float dis = (float)sqrt((float)pow(temp.i - cornerVec[z].y, 2) + (float)pow(temp.j - cornerVec[z].x, 2));
						int dis = sqrt(pow(temp.i - cornerVec[z].y, 2) + pow(temp.j - cornerVec[z].x, 2));
						if (dis < 3.0 && cornerList[z] == 0) {
							cornerList[z] = 1;
							find++;
						}
						if (find > threshold)
							return 1;
					}
				}
			}
		}
	}

	return 1;
}