#include "BackgroundSubtraction.h"

#define ALPHA 0.0
#define BETA 1.0
#define S_THRESH 5
#define H_THRESH 10

#define min_fast(a,b) ( (a>b) ? b : a)

using namespace cv;

void ShadowMapCreator(Mat* Shadow_Map, Mat* Input_Image, Mat* Background_Image)
{
	
	Mat HSV_Image, HSV_Background;
	Mat HSV_Split[3], HSV_Background_Split[3];

	cvtColor(*Background_Image, HSV_Background, CV_RGB2HSV, 0);
	split(HSV_Background, HSV_Background_Split);

	cvtColor(*Input_Image, HSV_Image, CV_RGB2HSV, 0);
	split(HSV_Image, HSV_Split);
	
	//0: Hue (angle, 0~360), 1: Saturation(0~1), 2: Value(0~1)

	bool HueCondition = false;
	bool SatCondition = false;
	bool ValCondition = false;
	
	double HueCon_data = 0.0;
	double SatCon_data = 0.0;
	double ValCon_data = 0.0;

	for (int x = 0; x < Cols; x++)
	{
		for (int y = 0; y < Rows; y++)
		{
			HueCondition = false;
			SatCondition = false;
			ValCondition = false;

			//condition calculation

			//Val
			ValCon_data = (double)HSV_Split[2].data[y*Cols + x] / (double)HSV_Background_Split[2].data[y*Cols + x];

			if (ValCon_data >= ALPHA && ValCon_data <= BETA)
				ValCondition = true;

			//Saturation
			SatCon_data = abs(HSV_Split[1].data[y*Cols + x] - HSV_Background_Split[1].data[y*Cols + x]);

			if (SatCon_data <= S_THRESH)
				SatCondition = true;

			//Hue
			HueCon_data = min_fast(HSV_Split[0].data[y*Cols + x] - HSV_Background_Split[0].data[y*Cols + x], 360 - (HSV_Split[0].data[y*Cols + x] - HSV_Background_Split[0].data[y*Cols + x]) );

			if (HueCon_data <= H_THRESH)
				HueCondition = true;


			//Shadow Map
			if (HueCondition && SatCondition && ValCondition)
				Shadow_Map->data[y*Cols + x] = 255;
			else
				Shadow_Map->data[y*Cols + x] = 0;


		}
	}


}