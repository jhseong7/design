#include "BackgroundSubtraction.h"

using namespace cv;

void IBS_PairDetection(Mat* BoundaryLabel_Low, Mat* BoundaryLabel_High, int* PairTable, size_t TableSize)
{
	/*
	PairTable�� 256�̳� 512�ܰ�� �̸� ����
	�����ڸ��� 0���� �ʱ�ȭ�Ѵ�. (���� loop�� ���� ����
	Boundary Low�� ���� �󺧿� �����Ǵ� Boundary High�� �� ��ȣ�� �Է��Ͽ� Lookup table ���·� ����
	�� table�� ���߿� Boundary Selection���� ���
	*/


	//1. Table�� �ʱ�ȭ
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