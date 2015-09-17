#include <cstdio>  
#include <iostream>  
#include <cmath>

#include <opencv2\opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/video/background_segm.hpp>  


//Threshold들 초기값 정리
#define Y_THRESHOLD 90
#define Y_THRESHOLD_HAT 245

#define C_THRESHOLD 120
#define C_THRESHOLD_HAT 245

#define T_THRESHOLD 10
#define T_THRESHOLD_HAT 245

//Texture Block 사이즈
#define BLOCKSIZE_X 1
#define BLOCKSIZE_Y 1

//Morphology 초기값
#define MORPHSIZE 3
#define MORPHSIZE2 2


using namespace std;
using namespace cv;


int keyboard;

//function declarations
void processVideo(char* videoFilename);
void processImages(char* firstFrameFilename);

void YmapCreator(Mat* Y_orig_Image, Mat* Y_Background_Image, Mat* Ymap_Image);
void CmapCreator(Mat* Cr_orig_Image, Mat* Cr_Background_Image, Mat* Cb_orig_Image, Mat* Cb_Background_Image, Mat* Cmap_Image);
void TmapCreator(Mat* Inten_Orig_Image, Mat* Inten_Background_Image, Mat* Tmap_Image);
void R_uv(double* AutoCorrelation_R, Mat* Original_Image, int u, int v, int x_coor, int y_coor, int blocksize_x, int blocksize_y);
void OR_MapCreator(Mat* OR_Map_Image, Mat* Y_Map, Mat* C_Map, Mat* T_Map);
void initializer(void);
void HistogramCreator(Mat* Output_Image, Mat* Input_Image);

//광역변수(Threshold들)

int y_threshold, y_threshold_hat;
int c_threshold, c_threshold_hat;
int t_threshold, t_threshold_hat;
int morph_size;
int morph_size2;


int main(int argc, char* argv[])
{
	//check for the input parameter correctness
	/*
	if (argc != 3) {
		cerr << "Incorrect input list" << endl;
		cerr << "exiting..." << endl;
		return EXIT_FAILURE;
	}
	//create GUI windows
	namedWindow("Frame");
	namedWindow("FG Mask MOG 2");
	namedWindow("Masked foreground");
	//create Background Subtractor objects
	pMOG2 = createBackgroundSubtractorMOG2(100, 64, false); //MOG2 approach

	if (strcmp(argv[1], "-vid") == 0) {
		//input data coming from a video
		processVideo(argv[2]);
	}
	else if (strcmp(argv[1], "-img") == 0) {
		//input data coming from a sequence of images
		processImages(argv[2]);
	}
	else {
		//error in reading input parameters
		cerr << "Please, check the input parameters." << endl;
		cerr << "Exiting..." << endl;
		return EXIT_FAILURE;
	}
	//destroy GUI windows
	destroyAllWindows();
	*/

	
	namedWindow("Original");
	//namedWindow("Background");
	//namedWindow("Y");
	namedWindow("Ydiff");
	//namedWindow("Ymap");
	//namedWindow("Cb");
	//namedWindow("Cr");
	//namedWindow("Cmap");
	namedWindow("Cdiff");
	//namedWindow("Tmap");
	namedWindow("Tdiff");
	namedWindow("Inten");
	//namedWindow("IntenB");

	namedWindow("OR_Map");
	

//윈도우 만들기 END

	initializer();

	char videoFilename[] = "test.avi";

	Mat OrigImage;
	Mat BackImage = imread("back.bmp");
	Mat YCrCbImage;
	Mat YCrCbImage_Back;
	Mat Intensity, Intensity_back;

	VideoCapture capture(videoFilename);
	
	if (!capture.isOpened())
	{
		//error in opening the video input
		cerr << "Unable to open video file: " << videoFilename << endl;
		exit(EXIT_FAILURE);
	}
	//read input data. ESC or 'q' for quitting
	while ((char)keyboard != 'q' && (char)keyboard != 27)
	{
		//read the current frame
		if (!capture.read(OrigImage)) {
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			exit(EXIT_FAILURE);
		}


		cvtColor(OrigImage, YCrCbImage, CV_RGB2YCrCb, 0);
		cvtColor(BackImage, YCrCbImage_Back, CV_RGB2YCrCb, 0);
		cvtColor(OrigImage, Intensity, CV_RGB2GRAY, 0);
		cvtColor(BackImage, Intensity_back, CV_RGB2GRAY, 0);

		Mat YCrCb_Split[3];
		Mat YCrCb_Back_Split[3];

		split(YCrCbImage, YCrCb_Split);
		split(YCrCbImage_Back, YCrCb_Back_Split);

		/*
		Mat Hist_Y, Hist_C, Hist_I;

		HistogramCreator(&Hist_I, &Intensity);
		HistogramCreator(&Hist_Y, &YCrCb_Split[0]);
		HistogramCreator(&Hist_C, &YCrCb_Split[1]);

		imshow("HistI", Hist_I);
		imshow("HistY", Hist_Y);
		imshow("HistC", Hist_C);
		*/

		Mat Ymap, Cmap, Tmap, ORmap;


		//cout << "Ymap Start" << endl;
		YmapCreator(&YCrCb_Split[0], &YCrCb_Back_Split[0], &Ymap);
		//cout << "Ymap End" << endl;

		//cout << "Cmap Start" << endl;
		CmapCreator(&YCrCb_Split[1], &YCrCb_Back_Split[1], &YCrCb_Split[2], &YCrCb_Back_Split[2], &Cmap);
		//cout << "Cmap End" << endl;

		//cout << "Tmap Start" << endl;
		TmapCreator(&Intensity, &Intensity_back, &Tmap);
		//cout << "Tmap End" << endl;

		//cout << "ORmap Start" << endl;
		OR_MapCreator(&ORmap, &Ymap, &Cmap, &Tmap);
		//cout << "ORmap End" << endl;



		//morphology START
		Mat element = getStructuringElement(MORPH_CROSS, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
		Mat element2 = getStructuringElement(MORPH_CROSS, Size(2 * morph_size2 + 1, 2 * morph_size2 + 1), Point(morph_size2, morph_size2));



		morphologyEx(Cmap, Cmap, MORPH_CLOSE, element2);
		morphologyEx(Cmap, Cmap, MORPH_OPEN, element2);

		morphologyEx(Ymap, Ymap, MORPH_CLOSE, element);
		morphologyEx(Ymap, Ymap, MORPH_OPEN, element);

		morphologyEx(Tmap, Tmap, MORPH_CLOSE, element);
		morphologyEx(Tmap, Tmap, MORPH_OPEN, element);

		morphologyEx(ORmap, ORmap, MORPH_OPEN, element);
		morphologyEx(ORmap, ORmap, MORPH_CLOSE, element);
		//morphology END


		//Image Show
		imshow("Original", OrigImage);
		//imshow("Background", OrigImage);
		//imshow("Y", YCrCb_Split[0]);
		//imshow("Cr", YCrCb_Split[1]);
		//imshow("Cb", YCrCb_Split[2]);
		//imshow("Ymap", Ymap);
		//imshow("Cmap", Cmap);
		//imshow("Tmap", Tmap);
		imshow("OR_Map", ORmap);

		keyboard = waitKey(30);
	}
	//delete capture object
	capture.release();

	return EXIT_SUCCESS;
}

void initializer(void)
{
	c_threshold = C_THRESHOLD;
	c_threshold_hat = C_THRESHOLD_HAT;

	y_threshold = Y_THRESHOLD;
	y_threshold_hat = Y_THRESHOLD_HAT;

	t_threshold = T_THRESHOLD;
	t_threshold_hat = T_THRESHOLD_HAT;

	morph_size = MORPHSIZE;
	morph_size2 = MORPHSIZE2;

}

void HistogramCreator(Mat* Output_Image, Mat* Input_Image)
{
	(*Output_Image) = Mat(512, 512, CV_8UC1);

	int histogram_table[256] = { 0 };
	int max_freq = 0;

	int Rows = Input_Image->rows;
	int Cols = Input_Image->cols;

	


	//histogram table 만들기
	for (int x = 0; x < Rows; x++)
	{
		for (int y = 0; y < Cols; y++)
		{
			if (Input_Image->data[x * Cols + y] < 256)
				histogram_table[Input_Image->data[x * Cols + y]]++;

			//최고값 기록
			
			if (histogram_table[Input_Image->data[x * Cols + y]] > max_freq)
				max_freq = histogram_table[Input_Image->data[x * Cols + y]];	
		}
	}

	double ratio = 512.0 / (double)max_freq;


	//histogram image 생성
	for (int x = 0; x < 512; x++)
	{
		for (int y = 0; y < 512; y++)
		{
			if (y < (double)histogram_table[x/2]*ratio)
				Output_Image->data[(511 - y) * 512 + x] = 0;
			else
				Output_Image->data[(511 - y) * 512 + x] = 255;
		}
	}



}

void OR_MapCreator(Mat* OR_Map_Image, Mat* Y_Map, Mat* C_Map, Mat* T_Map)
{
	int Rows = Y_Map->rows;
	int Cols = Y_Map->cols;

	(*OR_Map_Image) = Mat(Rows, Cols, CV_8UC1);

	for (int x = 0; x < Rows; x++)
	{
		for (int y = 0; y < Cols; y++)
		{
			if (Y_Map->data[x*Cols + y] || C_Map->data[x*Cols + y] || T_Map->data[x*Cols + y])
				OR_Map_Image->data[x*Cols + y] = 255;
			else
				OR_Map_Image->data[x*Cols + y] = 0;
		}
	}



}


void YmapCreator(Mat* Y_orig_Image, Mat* Y_Background_Image, Mat* Ymap_Image)
{
	

	int Rows = Y_orig_Image->rows;
	int Cols = Y_orig_Image->cols;

	Mat Difference_Y(Rows, Cols, CV_8UC1);
	(*Ymap_Image) = Mat(Rows, Cols, CV_8UC1);

	for (int x = 0; x < Rows; x++)
	{
		for (int y = 0; y < Cols; y++)
		{
			Difference_Y.data[x * Cols + y] = (Y_orig_Image->data[x * Cols + y] - Y_Background_Image->data[x * Cols + y]);

			if (Difference_Y.data[x * Cols + y] <= 0) //차이가 0이하면 무조건 0으로 처리(정의)
				Difference_Y.data[x * Cols + y] = 0;

			if (Difference_Y.data[x * Cols + y] >= y_threshold_hat) //차이가 0이하면 무조건 0으로 처리(정의)
				Difference_Y.data[x * Cols + y] = 0;


			if (Difference_Y.data[x * Cols + y] > y_threshold)
				Ymap_Image->data[x * Cols + y] = 255;
			else
				Ymap_Image->data[x * Cols + y] = 0;

		}
	}

	imshow("Ydiff", Difference_Y);

	Mat Histo;

	HistogramCreator(&Histo, &Difference_Y);
	
	imshow("Y diff Histogram", Histo);

}

void CmapCreator(Mat* Cr_orig_Image, Mat* Cr_Background_Image, Mat* Cb_orig_Image, Mat* Cb_Background_Image, Mat* Cmap_Image)
{
	int Rows = Cr_orig_Image->rows;
	int Cols = Cr_orig_Image->cols;

	Mat Difference_C(Rows, Cols, CV_8UC1);
	(*Cmap_Image) = Mat(Rows, Cols, CV_8UC1);

	for (int x = 0; x < Rows; x++)
	{
		for (int y = 0; y < Cols; y++)
		{
			Difference_C.data[x * Cols + y] = pow(Cr_orig_Image->data[x * Cols + y] - Cr_Background_Image->data[x * Cols + y], 2) + pow(Cb_orig_Image->data[x * Cols + y] - Cb_Background_Image->data[x * Cols + y], 2);

			if (Difference_C.data[x * Cols + y] <= 0) //차이가 0이하면 무조건 0으로 처리(정의)
				Difference_C.data[x * Cols + y] = 0;


			if (Difference_C.data[x * Cols + y] > c_threshold)
				Cmap_Image->data[x * Cols + y] = 255;
			else
				Cmap_Image->data[x * Cols + y] = 0;

		}
	}


	imshow("Cdiff", Difference_C);

	Mat Histo;

	HistogramCreator(&Histo, &Difference_C);

	imshow("C diff Histogram", Histo);

}

void TmapCreator(Mat* Inten_Orig_Image, Mat* Inten_Background_Image, Mat* Tmap_Image)
{
	int Rows = Inten_Orig_Image->rows;
	int Cols = Inten_Orig_Image->cols;

	Mat Difference_T(Rows, Cols, CV_8UC1);
	(*Tmap_Image) = Mat(Rows, Cols, CV_8UC1);

	double R_orig = 0;
	double R_back = 0;

	double denominator = 1.0 / ((2 * BLOCKSIZE_X + 1)*(2 * BLOCKSIZE_Y + 1));

	double differ_T = 0;

	for (int y = 0; y < Rows; y++)
	{
		cout << "Processing " << 100.0*(double)y/(double)Rows << "percent     " << '\r';

		for (int x = 0; x < Cols; x++)
		{
			Difference_T.data[y*Cols + x] = 255;
			differ_T = 0;

			
			for (int i = 0; i < 2 * BLOCKSIZE_X; i++)
			{
				for (int j = 0; j < 2 * BLOCKSIZE_Y; j++)
				{
					R_uv(&R_orig, Inten_Orig_Image, i, j, x, y, BLOCKSIZE_X, BLOCKSIZE_Y);
					R_uv(&R_back, Inten_Background_Image, i, j, x, y, BLOCKSIZE_X, BLOCKSIZE_Y);

					differ_T += pow((R_orig - R_back), 2);
				}
			}

			differ_T *= 50000*denominator;

			if ((differ_T) > 255)
				Difference_T.data[y*Cols + x] = 255;
			else
				Difference_T.data[y*Cols + x] = differ_T;


			if (differ_T > t_threshold)
				Tmap_Image->data[y*Cols + x] = 255;
			else
				Tmap_Image->data[y*Cols + x] = 0;
		}
	}

	//imshow("Tdiff", Difference_T);
	//imshow("Inten", (*Inten_Orig_Image));
	//imshow("IntenB", (*Inten_Background_Image));


	Mat Histo;

	HistogramCreator(&Histo, &Difference_T);

	imshow("T diff Histogram", Histo);

}

void R_uv(double* AutoCorrelation_R, Mat* Original_Image, int u, int v, int x_coor, int y_coor, int blocksize_x, int blocksize_y) //blocksize는 각각 M,N
{
	int Rowsize = Original_Image->rows;
	int Colsize = Original_Image->cols;
	int M = blocksize_x;
	int N = blocksize_y;
	int M2plus1 = 2 * M + 1;
	int N2plus1 = 2 * N + 1;

	// m = 0 , n=0일때는  절대 좌표는 (x,y), 즉 m,n자리에는 x+m, y+n이 들어가면 된다.


	
			//R1 = {(2M+1)(2N+1)}/{(2M+1 - u)(2N+1 - v)}
			double R1 = ((M2plus1)*(N2plus1)) / ((M2plus1 - u)*(N2plus1 - v));

			//R2_Den = sigma( sigma(p(m,n)^2 , 0 , 2N) , 0, 2M)
			double R2_den = 0;

			for (int i = 0; i < 2 * M; i++)
			{
				for (int j = 0; j < 2 * N; j++)
				{
					//m = x+i, n = y+j
					if ((x_coor + i) < Colsize && (y_coor + j) < Rowsize) //원본 이미지의 보더를 넘어가는 것을 방지
						R2_den += pow(Original_Image->data[(y_coor + j) * Colsize + (x_coor + i)], 2);
				}
			}

			//R2_Num

			double R2_num = 0;
			for (int i = 0; i < 2 * M - u; i++)
			{
				for (int j = 0; j < 2 * N - v; j++)
				{
					//m = x+i, n = y+j
					if ((x_coor + i + u) < Colsize && (y_coor + j + v) < Rowsize) //원본 이미지의 보더를 넘어가는 것을 방지
						R2_num += Original_Image->data[(y_coor + j) * Colsize + (x_coor + i)] * Original_Image->data[(y_coor + j + v)* Colsize + (x_coor + i + u)];
				}
			}

			(*AutoCorrelation_R) = R1 * R2_num / R2_den;




}


/*
void processVideo(char* videoFilename) {
	//create the capture object
	int morph_size = 5;
	int morph_size2 = 2;
	Mat element = getStructuringElement(MORPH_CROSS, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	Mat element2 = getStructuringElement(MORPH_CROSS, Size(2 * morph_size2 + 1, 2 * morph_size2 + 1), Point(morph_size2, morph_size2));

	

	VideoCapture capture(videoFilename);
	if (!capture.isOpened()){
		//error in opening the video input
		cerr << "Unable to open video file: " << videoFilename << endl;
		exit(EXIT_FAILURE);
	}
	//read input data. ESC or 'q' for quitting
	while ((char)keyboard != 'q' && (char)keyboard != 27){
		//read the current frame
		if (!capture.read(frame)) {
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			exit(EXIT_FAILURE);
		}

		

		//update the background model
		pMOG2->apply(frame, fgMaskMOG2);
		//get the frame number and write it on the current frame
		stringstream ss;
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		ss << capture.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//show the current frame and the fg morphologyEx

		
		morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, element2);
		morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_CLOSE, element2);


		Mat foregroundmasked(frame.rows,frame.cols, CV_8UC3, Scalar(0,0,0));
		frame.copyTo(foregroundmasked, fgMaskMOG2);


		imshow("Frame", frame);
		imshow("FG Mask MOG 2", fgMaskMOG2);
		imshow("Masked foreground", foregroundmasked);
		//get the input from the keyboard
		keyboard = waitKey(30);
	}
	//delete capture object
	capture.release();
}

void processImages(char* fistFrameFilename) {
	//read the first file of the sequence
	frame = imread(fistFrameFilename);
	if (!frame.data){
		//error in opening the first image
		cerr << "Unable to open first image frame: " << fistFrameFilename << endl;
		exit(EXIT_FAILURE);
	}
	//current image filename
	string fn(fistFrameFilename);
	//read input data. ESC or 'q' for quitting
	while ((char)keyboard != 'q' && (char)keyboard != 27){
		//update the background model
		pMOG2->apply(frame, fgMaskMOG2);
		//get the frame number and write it on the current frame
		size_t index = fn.find_last_of("/");
		if (index == string::npos) {
			index = fn.find_last_of("\\");
		}
		size_t index2 = fn.find_last_of(".");
		string prefix = fn.substr(0, index + 1);
		string suffix = fn.substr(index2);
		string frameNumberString = fn.substr(index + 1, index2 - index - 1);
		istringstream iss(frameNumberString);
		int frameNumber = 0;
		iss >> frameNumber;
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//show the current frame and the fg masks
		imshow("Frame", frame);
		imshow("FG Mask MOG 2", fgMaskMOG2);
		//get the input from the keyboard
		keyboard = waitKey(30);
		//search for the next image in the sequence
		ostringstream oss;
		oss << (frameNumber + 1);
		string nextFrameNumberString = oss.str();
		string nextFrameFilename = prefix + nextFrameNumberString + suffix;
		//read the next frame
		frame = imread(nextFrameFilename);
		if (!frame.data){
			//error in opening the next image in the sequence
			cerr << "Unable to open image frame: " << nextFrameFilename << endl;
			exit(EXIT_FAILURE);
		}
		//update the path of the current frame
		fn.assign(nextFrameFilename);
	}
}

*/