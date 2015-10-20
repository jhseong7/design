#include "BackgroundSubtraction.h"

#define LOW_OFFSET 5
#define MED_OFFSET 0
#define HIGH_OFFSET 0

using namespace cv;

void ThresholdImageCreate(Mat* Threshold_Image, Mat* InputImage)
{
	int histogramtable[256] = { 0 };
	TriThreshold threshold;
	Mat Histogram;

	HistogramTableCreator(histogramtable, InputImage);

	HistogramReductionSmoother(histogramtable, 256, DELTA_MAX);


	threshold = TriangleAlgorithm(histogramtable);

	//3개의 Threshold 이미지 생성
	Thresholder(Threshold_Image, InputImage, threshold.Med);


}

void TriThresholdImageCreate(Mat* Threshold_Low, Mat* Threshold_Mid, Mat* Threshold_High, Mat* InputImage)
{
	int histogramtable[256] = { 0 };
	TriThreshold threshold;
	Mat Histogram;
	
	HistogramTableCreator(histogramtable, InputImage);
	
	HistogramReductionSmoother(histogramtable, 256, DELTA_MAX);


	threshold = TriangleAlgorithm(histogramtable);

	//3개의 Threshold 이미지 생성
	Thresholder(Threshold_Low, InputImage, threshold.Low + LOW_OFFSET);
	Thresholder(Threshold_Mid, InputImage, threshold.Med + MED_OFFSET);
	Thresholder(Threshold_High, InputImage, threshold.High + HIGH_OFFSET);

}

void Thresholder(Mat* ThresholdImage, Mat* OrigImage, int threshold)
{

	for (int x = 0; x < Cols; x++)
	{
		for (int y = 0; y < Rows; y++)
		{
			if (OrigImage->data[y*Cols + x] >= threshold)
				ThresholdImage->data[y*Cols + x] = 255;
			else
				ThresholdImage->data[y*Cols + x] = 0;
		}
	}
}