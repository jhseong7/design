#include "BackgroundSubtraction.h"

using namespace cv;

typedef struct{
	int x;
	int y;
} Vector2d;


int VectorConversionCoordinate_to_Int(Vector2d RawVector);
Vector2d VectorConversionInt_to_Coordinate(int IntVector);



Vector2d NormalVector(Vector2d Pkm1, Vector2d Pk, Vector2d Pkp1)
{
	Vector2d Line1, Line2;
	Vector2d NormalVector;

	Line2.x = Pkp1.x - Pk.x;
	Line2.y = Pkp1.y - Pk.y;
	
	Line1.x = Pk.x - Pkm1.x;
	Line1.y = Pk.y - Pkm1.y;

	int Line1_int, Line2_int;

	Line1_int = VectorConversionCoordinate_to_Int(Line1);
	Line2_int = VectorConversionCoordinate_to_Int(Line2);

	//Modulo([E(k)+E(k+1)] div 2 - 2, 8)

	int Normal_int = (((Line1_int + Line2_int) / 2) - 2) % 8;

	NormalVector = VectorConversionInt_to_Coordinate(Normal_int);

	return NormalVector;
}


int VectorConversionCoordinate_to_Int(Vector2d RawVector)
{
	/*
	1 | 0 | 7
	2 | * | 6
	3 | 4 | 5
	
	*/


	if ((RawVector.x == 0) && (RawVector.y == 1))
		return 0;
	if ((RawVector.x == 1) && (RawVector.y == 1))
		return 7;
	if ((RawVector.x == 1) && (RawVector.y == 0))
		return 6;
	if ((RawVector.x == 1) && (RawVector.y == -1))
		return 5;
	if ((RawVector.x == 0) && (RawVector.y == -1))
		return 4;
	if ((RawVector.x == -1) && (RawVector.y == -1))
		return 3;
	if ((RawVector.x == -1) && (RawVector.y == 0))
		return 2;
	if ((RawVector.x == -1) && (RawVector.y == 1))
		return 1;

}

Vector2d VectorConversionInt_to_Coordinate(int IntVector)
{
	Vector2d ConvertedVector;

	if (IntVector == 0)
		ConvertedVector = { 0, 1 };
	else if (IntVector == 1)
		ConvertedVector = { 1, 1 };
	else if (IntVector == 2)
		ConvertedVector = { 1, 0 };
	else if (IntVector == 3)
		ConvertedVector = { 1, -1 };
	else if (IntVector == 4)
		ConvertedVector = { 0, -1 };
	else if (IntVector == 5)
		ConvertedVector = { -1, -1 };
	else if (IntVector == 6)
		ConvertedVector = { -1, 0 };
	else if (IntVector == 7)
		ConvertedVector = { -1, 1 };

	
	return ConvertedVector;
}