#include "BackgroundSubtraction.h"

using namespace cv;


void CannyEdgeDetector(Mat* EdgeImage, Mat* InputImage)
{
	int lowthreshold = 50;
	int ratio = 3;
	int kernelsize = 3;

	blur(*InputImage, *EdgeImage, Size(3, 3));
	
	Canny(*EdgeImage, *EdgeImage, lowthreshold, ratio*lowthreshold, kernelsize);


}

void ForegroundEdgeMap(Mat* EdgeMap_fore, Mat* Edge_I, Mat* Grad_diff)
{
	for (int x = 0; x < Cols; x++)
	{
		for (int y = 0; y < Rows; y++)
		{
			if ((Edge_I->data[y*Cols + x] == 255) && (Grad_diff->data[y*Cols + x] == 255))
				EdgeMap_fore->data[y*Cols + x] = 255;
			else
				EdgeMap_fore->data[y*Cols + x] = 0;
		}
	}


}
