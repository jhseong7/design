#include "BackgroundSubtraction.h"

using namespace cv;


void CannyEdgeDetector(Mat* EdgeImage, Mat* InputImage)
{
	int lowthreshold = 100;
	int ratio = 3;
	int kernelsize = 3;

	blur(*InputImage, *EdgeImage, Size(3, 3));
	
	Canny(*EdgeImage, *EdgeImage, lowthreshold, ratio*lowthreshold, kernelsize);


}
