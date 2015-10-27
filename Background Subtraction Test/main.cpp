#include "BackgroundSubtraction.h"
#include "CL\opencl.h"

#define NUM_DATA 100

#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     abort();                                                                   \
      } while (0)

#define CL_CHECK_ERR(_expr)                                                     \
   ({                                                                           \
     cl_int _err = CL_INVALID_VALUE;                                            \
     typeof(_expr) _ret = _expr;                                                \
     if (_err != CL_SUCCESS) {                                                  \
       fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       abort();                                                                 \
	      }                                                                          \
     _ret;                                                                      \
      })


using namespace std;
using namespace cv;


int keyboard;

int frame_no;



//광역변수(Threshold들)

int y_threshold, y_threshold_hat;
int c_threshold, c_threshold_hat;
int t_threshold, t_threshold_hat;
int morph_size;
int morph_size2;

int Threshold_offet;

//광역변수
int Rows, Cols;
Mat YCrCbImage;
Mat YCrCbImage_Back;
Mat Intensity, Intensity_back;
Mat YCrCb_Split[3];
Mat YCrCb_Back_Split[3];



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

	

	//OPENCL 실험용 코드
	cl_uint platformIdCount = 0;
	clGetPlatformIDs(0, nullptr, &platformIdCount);

	std::vector<cl_platform_id> platformIds(platformIdCount);
	clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

	/*
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);
	std::vector<cl_device_id> deviceIds(deviceIdCount);
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);
	*/
	cl_platform_id platforms[100];
	cl_uint platforms_n = 0;
	CL_CHECK(clGetPlatformIDs(100, platforms, &platforms_n));
	
	printf("=== %d OpenCL platform(s) found: ===\n", platforms_n);
	for (int i = 0; i<platforms_n; i++)
	{
		char buffer[10240];
		printf("  -- %d --\n", i);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 10240, buffer, NULL));
		printf("  PROFILE = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 10240, buffer, NULL));
		printf("  VERSION = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 10240, buffer, NULL));
		printf("  NAME = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL));
		printf("  VENDOR = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL));
		printf("  EXTENSIONS = %s\n", buffer);
	}

	

	

//윈도우 만들기 END
	frame_no = 0;

	char videoFilename[] = "[mix][mix]test.avi"; //입력 영상 파일

	Mat OrigImage;
	Mat BackImage = imread("back.bmp"); 

	//3개의 Difference Image
	Mat Tdiff, Ydiff, Cdiff;

	//9개의 Map들
	Mat Tmap_Low, Tmap_Med, Tmap_High;
	Mat Ymap_Low, Ymap_Med, Ymap_High;
	Mat Cmap_Low, Cmap_Med, Cmap_High;

	//3개의 ORmap
	Mat ORmap_Low, ORmap_Med, ORmap_High;

	//3개의 BoundaryMap
	Mat Boundary_Low, Boundary_Med, Boundary_High;

	//2개의 Gradient Map
	Mat Gradient_fore_x, Gradient_fore_y, Gradient_fore_normalized;
	Mat Gradient_back_x, Gradient_back_y, Gradient_back_normalized;

	Mat Gradient_diff;


	//Edge map
	Mat Edge_fore, Edge_back;
	Mat Edgemap;


	TriThreshold Thresh;


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
		
		if (Tdiff.data == NULL) //초기화가 안되어있다면 초기화단계를 거친다.
		{
			//최초의 한번만 확인을 거친다
			Rows = OrigImage.rows;
			Cols = OrigImage.cols;

			if (BackImage.rows != Rows || BackImage.cols != Cols) //배경과 크기의 일치를 확인한다
			{
				cerr << "Frame Size Mismatch" << endl;
				cerr << "Exiting..." << endl;
				exit(EXIT_FAILURE);
			}

			Tdiff = Mat(Rows, Cols, CV_8UC1);
			Ydiff = Mat(Rows, Cols, CV_8UC1);
			Cdiff = Mat(Rows, Cols, CV_8UC1);

			Cmap_Low = Mat(Rows, Cols, CV_8UC1);
			Cmap_Med = Mat(Rows, Cols, CV_8UC1);
			Cmap_High = Mat(Rows, Cols, CV_8UC1);
			Ymap_Low = Mat(Rows, Cols, CV_8UC1);
			Ymap_Med = Mat(Rows, Cols, CV_8UC1);
			Ymap_High = Mat(Rows, Cols, CV_8UC1);
			Tmap_Low = Mat(Rows, Cols, CV_8UC1);
			Tmap_Med = Mat(Rows, Cols, CV_8UC1);
			Tmap_High = Mat(Rows, Cols, CV_8UC1);

			ORmap_Low = Mat(Rows, Cols, CV_8UC1);
			ORmap_Med = Mat(Rows, Cols, CV_8UC1);
			ORmap_High = Mat(Rows, Cols, CV_8UC1);

			Boundary_Low = Mat(Rows, Cols, CV_8UC1);
			Boundary_Med = Mat(Rows, Cols, CV_8UC1);
			Boundary_High = Mat(Rows, Cols, CV_8UC1);

			Edge_fore = Mat(Rows, Cols, CV_8UC1);
			Edge_back = Mat(Rows, Cols, CV_8UC1);

			Edgemap = Mat(Rows, Cols, CV_8UC1);

			Gradient_fore_x = Mat(Rows, Cols, CV_16SC1);
			Gradient_fore_y = Mat(Rows, Cols, CV_16SC1);

			Gradient_back_x = Mat(Rows, Cols, CV_16SC1);
			Gradient_back_y = Mat(Rows, Cols, CV_16SC1);

			Gradient_diff = Mat(Rows, Cols, CV_16SC1);
		}

		OrigImage = imread("foretest.bmp");


		//YCbCr 생성
		cvtColor(OrigImage, YCrCbImage, CV_RGB2YCrCb, 0);
		cvtColor(BackImage, YCrCbImage_Back, CV_RGB2YCrCb, 0);
		split(YCrCbImage, YCrCb_Split);
		split(YCrCbImage_Back, YCrCb_Back_Split);

		//Differential 이미지 3가지(T, Y, C) 생성
		DiffentialImageCalcuation(&Tdiff, &Ydiff, &Cdiff, &OrigImage, &BackImage);

		//Threshold
		TriThresholdImageCreate(&Tmap_Low, &Tmap_Med, &Tmap_High, &Tdiff);
		TriThresholdImageCreate(&Ymap_Low, &Ymap_Med, &Ymap_High, &Ydiff);
		TriThresholdImageCreate(&Cmap_Low, &Cmap_Med, &Cmap_High, &Cdiff);

		//ORmap
		OR_MapCreator_n(&ORmap_Low, &Tmap_Low, &Ymap_Low, &Cmap_Low);
		OR_MapCreator_n(&ORmap_Med, &Tmap_Med, &Ymap_Med, &Cmap_Med);
		OR_MapCreator_n(&ORmap_High, &Tmap_High, &Ymap_High, &Cmap_High);


		//Canny를 이용하여 Edge 검출
		CannyEdgeDetector(&Edge_fore, &YCrCb_Split[0]);
		//CannyEdgeDetector(&Edge_back, &YCrCb_Back_Split[0]);

		GradientMap(&Gradient_fore_x, &Gradient_fore_y, &YCrCb_Split[0]);
		GradientMap(&Gradient_back_x, &Gradient_back_y, &YCrCb_Back_Split[0]);

		GradientDifference(&Gradient_diff, &Gradient_fore_x, &Gradient_fore_y, &Gradient_back_x, &Gradient_back_y);
		Mat Graddiff8uc1;

		Gradient_diff.convertTo(Graddiff8uc1, CV_8UC1);


		//ThresholdImageCreate(&Gradient_diff, &Gradient_diff);
		ThresholdImageCreate(&Graddiff8uc1, &Graddiff8uc1);

		//morphology START
		Mat element2 = getStructuringElement(MORPH_CROSS, Size(2 * 2 + 1, 2 * 2 + 1), Point(2, 2));


		morphologyEx(Graddiff8uc1, Graddiff8uc1, MORPH_CLOSE, element2);
		//morphologyEx(Graddiff8uc1, Graddiff8uc1, MORPH_OPEN, element2);


		ForegroundEdgeMap(&Edgemap, &Edge_fore, &Graddiff8uc1);

		//Boundary
		BoundaryMap(&Boundary_Low, &ORmap_Low);
		BoundaryMap(&Boundary_Med, &ORmap_Med);
		BoundaryMap(&Boundary_High, &ORmap_High);
		

		//Connected Component Test
		Mat ConnectLabel_Low, ConnectLabel_Med, ConnectLabel_High;

		connectedComponents(Boundary_Low, ConnectLabel_Low, 8, CV_16U);
		connectedComponents(Boundary_Med, ConnectLabel_Med, 8, CV_16U);
		connectedComponents(Boundary_High, ConnectLabel_High, 8, CV_16U);

		imshow("Connect Low", ConnectLabel_Low);
		imshow("Connect Med", ConnectLabel_Med);
		imshow("Connect High", ConnectLabel_High);


		//morphologyEx(Ymap, Ymap, MORPH_CLOSE, element);
		//morphologyEx(Ymap, Ymap, MORPH_OPEN, element);

		//morphologyEx(Tmap, Tmap, MORPH_CLOSE, element);
		//morphologyEx(Tmap, Tmap, MORPH_OPEN, element);

		//morphologyEx(ORmap, ORmap, MORPH_OPEN, element);
		//morphologyEx(ORmap, ORmap, MORPH_CLOSE, element);

		//morphology END


		//Image Show
	
		

		//imshow("Edge fore", Edge_fore);
		//imshow("Edge back", Edge_back);

		imshow("Edge Map", Edgemap);

		//imshow("Gradient fore x", Gradient_fore_x);
		//imshow("Gradient fore y", Gradient_fore_y);

		//imshow("Gradient back x", Gradient_back_x);
		//imshow("Gradient back y", Gradient_back_y);

		imshow("Gradient diff", Graddiff8uc1);

		//imshow("Tdiff", Tdiff);
		//imshow("Ydiff", Ydiff);
		//imshow("Cdiff", Cdiff);

		imshow("OR Low", ORmap_Low);
		imshow("OR Med", ORmap_Med);
		imshow("OR High", ORmap_High);

		imshow("Boundary Low", Boundary_Low);
		imshow("Boundary Med", Boundary_Med);
		imshow("Boundary High", Boundary_High);

		imwrite("Tmap_Low.bmp", Tmap_Low);
		imwrite("Tmap_Med.bmp", Tmap_Med);
		imwrite("Tmap_High.bmp", Tmap_High);

		imwrite("Ymap_Low.bmp", Ymap_Low);
		imwrite("Ymap_Med.bmp", Ymap_Med);
		imwrite("Ymap_High.bmp", Ymap_High);

		imwrite("Cmap_Low.bmp", Cmap_Low);
		imwrite("Cmap_Med.bmp", Cmap_Med);
		imwrite("Cmap_High.bmp", Cmap_High);

		imwrite("OR_Low.bmp", ORmap_Low);
		imwrite("OR_Med.bmp", ORmap_Med);
		imwrite("OR_High.bmp", ORmap_High);

	
		imwrite("Boundary Low.bmp", Boundary_Low);
		imwrite("Boundary Med.bmp", Boundary_Med);
		imwrite("Boundary High.bmp", Boundary_High);

		imwrite("Edge map.bmp", Edgemap);

		keyboard = waitKey(30);

		system("pause");

		cout << frame_no++ << endl;
	}
	//delete capture object
	capture.release();

	return EXIT_SUCCESS;
}


void BoundaryMap(Mat* BoundaryMap, Mat* InputMap)
{
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));

	Mat ErodedMap; 

	morphologyEx((*InputMap), ErodedMap, MORPH_ERODE, element);

	int rows = InputMap->rows;
	int cols = InputMap->cols;

	(*BoundaryMap) = Mat(rows, cols, CV_8UC1);

	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			if ((int)InputMap->data[y*cols + x] - (int)ErodedMap.data[y*cols + x] != 0)
				BoundaryMap->data[y*cols + x] = 255;
			else 
				BoundaryMap->data[y*cols + x] = 0;
		}
	}


}

/*


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


void YmapCreator(Mat* Y_orig_Image, Mat* Y_Background_Image, Mat* Ymap_Image_Low, Mat* Ymap_Image_Med, Mat* Ymap_Image_High)
{
	

	int Rows = Y_orig_Image->rows;
	int Cols = Y_orig_Image->cols;

	Mat Difference_Y(Rows, Cols, CV_8UC1);
	(*Ymap_Image_Low) = Mat(Rows, Cols, CV_8UC1);
	(*Ymap_Image_Med) = Mat(Rows, Cols, CV_8UC1);
	(*Ymap_Image_High) = Mat(Rows, Cols, CV_8UC1);

	for (int x = 0; x < Rows; x++)
	{
		for (int y = 0; y < Cols; y++)
		{
			Difference_Y.data[x * Cols + y] = (Y_orig_Image->data[x * Cols + y] - Y_Background_Image->data[x * Cols + y]);

			if (Difference_Y.data[x * Cols + y] <= 0) //차이가 0이하면 무조건 0으로 처리(정의)
				Difference_Y.data[x * Cols + y] = 0;

			if (Difference_Y.data[x * Cols + y] >= y_threshold_hat) //차이가 0이하면 무조건 0으로 처리(정의)
				Difference_Y.data[x * Cols + y] = 0;
		}
	}

	Mat Histo;
	int Histogram_table[256] = { 0 };

	HistogramTableCreator(Histogram_table, &Difference_Y);
	HistogramImageCreator(&Histo, Histogram_table);

	cout << "Y threshold: ";
	TriThreshold Thresh = TriangleAlgorithm(Histogram_table);


	imshow("Ydiff", Difference_Y);	
	imshow("Y diff Histogram", Histo);

	imwrite("Ydiff.jpg", Difference_Y);
	imwrite("Y diff Histogram.jpg", Histo);

	for (int x = 0; x < Rows; x++)
	{
		for (int y = 0; y < Cols; y++)
		{
			if (Difference_Y.data[x * Cols + y] > Thresh.Low)
				Ymap_Image_Low->data[x * Cols + y] = 255;
			else
				Ymap_Image_Low->data[x * Cols + y] = 0;


			if (Difference_Y.data[x * Cols + y] > Thresh.Med)
				Ymap_Image_Med->data[x * Cols + y] = 255;
			else
				Ymap_Image_Med->data[x * Cols + y] = 0;


			if (Difference_Y.data[x * Cols + y] > Thresh.High)
				Ymap_Image_High->data[x * Cols + y] = 255;
			else
				Ymap_Image_High->data[x * Cols + y] = 0;
		}
	}


}

void CmapCreator(Mat* Cr_orig_Image, Mat* Cr_Background_Image, Mat* Cb_orig_Image, Mat* Cb_Background_Image, Mat* Cmap_Image_Low, Mat* Cmap_Image_Med, Mat* Cmap_Image_High)
{
	int Rows = Cr_orig_Image->rows;
	int Cols = Cr_orig_Image->cols;

	Mat Difference_C(Rows, Cols, CV_8UC1);
	(*Cmap_Image_Low) = Mat(Rows, Cols, CV_8UC1);
	(*Cmap_Image_Med) = Mat(Rows, Cols, CV_8UC1);
	(*Cmap_Image_High) = Mat(Rows, Cols, CV_8UC1);

	for (int x = 0; x < Rows; x++)
	{
		for (int y = 0; y < Cols; y++)
		{
			Difference_C.data[x * Cols + y] = (uchar)sqrt(pow(Cr_orig_Image->data[x * Cols + y] - Cr_Background_Image->data[x * Cols + y], 2) + pow(Cb_orig_Image->data[x * Cols + y] - Cb_Background_Image->data[x * Cols + y], 2) );

			if (Difference_C.data[x * Cols + y] <= 0) //차이가 0이하면 무조건 0으로 처리(정의)
				Difference_C.data[x * Cols + y] = 0;
		}
	}




	Mat Histo;
	int Histogram_table[256] = { 0 };

	HistogramTableCreator(Histogram_table, &Difference_C);
	HistogramReductionSmoother(Histogram_table, 256, DELTA_MAX);
	HistogramImageCreator(&Histo, Histogram_table);

	cout << "C threshold: ";
	TriThreshold Thresh = TriangleAlgorithm(Histogram_table);

	imshow("C diff Histogram", Histo);
	imshow("Cdiff", Difference_C);

	imwrite("C diff Histogram.jpg", Histo);
	imwrite("Cdiff.jpg", Difference_C);

	Thresh.Low += C_THRESHOLD_OFFSET;
	Thresh.Med += C_THRESHOLD_OFFSET;
	Thresh.High += C_THRESHOLD_OFFSET;

	for (int x = 0; x < Rows; x++)
	{
		for (int y = 0; y < Cols; y++)
		{
			if (Difference_C.data[x * Cols + y] > Thresh.Low)
				Cmap_Image_Low->data[x * Cols + y] = 255;
			else
				Cmap_Image_Low->data[x * Cols + y] = 0;


			if (Difference_C.data[x * Cols + y] > Thresh.Med)
				Cmap_Image_Med->data[x * Cols + y] = 255;
			else
				Cmap_Image_Med->data[x * Cols + y] = 0;


			if (Difference_C.data[x * Cols + y] > Thresh.High)
				Cmap_Image_High->data[x * Cols + y] = 255;
			else
				Cmap_Image_High->data[x * Cols + y] = 0;
		}
	}

}

void TmapCreator(Mat* Inten_Orig_Image, Mat* Inten_Background_Image, Mat* Tmap_Image_Low, Mat* Tmap_Image_Med, Mat* Tmap_Image_High)
{
	int Rows = Inten_Orig_Image->rows;
	int Cols = Inten_Orig_Image->cols;

	Mat Difference_T(Rows, Cols, CV_8UC1);
	(*Tmap_Image_Low) = Mat(Rows, Cols, CV_8UC1);
	(*Tmap_Image_Med) = Mat(Rows, Cols, CV_8UC1);
	(*Tmap_Image_High) = Mat(Rows, Cols, CV_8UC1);

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
				Difference_T.data[y*Cols + x] = (uchar)differ_T;



		}
	}

	imshow("Tdiff", Difference_T);
	//imshow("Inten", (*Inten_Orig_Image));
	//imshow("IntenB", (*Inten_Background_Image));


	Mat Histo;
	int Histogram_table[256] = { 0 };

	HistogramTableCreator(Histogram_table, &Difference_T);
	HistogramImageCreator(&Histo, Histogram_table);


	cout << "T threshold: ";
	TriThreshold Thresh = TriangleAlgorithm(Histogram_table);



	imshow("T diff Histogram", Histo);

	imwrite("Tdiff.jpg", Difference_T);
	imwrite("T diff Histogram.jpg", Histo);

	for (int y = 0; y < Rows; y++)
	{
		for (int x = 0; x < Cols; x++)
		{
			if (Difference_T.data[y * Cols + x] > Thresh.Low)
				Tmap_Image_Low->data[y * Cols + x] = 255;
			else
				Tmap_Image_Low->data[y * Cols + x] = 0;


			if (Difference_T.data[y * Cols + x] > Thresh.Med)
				Tmap_Image_Med->data[y * Cols + x] = 255;
			else
				Tmap_Image_Med->data[y * Cols + x] = 0;


			if (Difference_T.data[y * Cols + x] > Thresh.High)
				Tmap_Image_High->data[y * Cols + x] = 255;
			else
				Tmap_Image_High->data[y * Cols + x] = 0;
		}
	}

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

*/

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