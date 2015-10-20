#include "BackgroundSubtraction.h"

using namespace cv;

void IBS_PairDetection(Mat* BoundaryLabel_Low, Mat* BoundaryLabel_High, int* PairTable, size_t TableSize)
{
	/*
	PairTable은 256이나 512단계로 미리 설정
	들어오자마자 0으로 초기화한다. (이전 loop의 잔재 제거
	Boundary Low로 들어온 라벨에 대응되는 Boundary High의 라벨 번호를 입력하여 Lookup table 형태로 구성
	이 table은 나중에 Boundary Selection에서 사용
	*/


	//1. Table의 초기화
	for (int i = 0; i < TableSize; i++)
		PairTable[i] = 0;


	//2. Orientation Matching
	
	for (int x = 0; x < Cols; x++)
	{
		for (int y = 0; y < Rows; y++)
		{


		}
		
	}
}

void Velocity()
{

}