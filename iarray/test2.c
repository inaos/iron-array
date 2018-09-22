#include <string.h>
#include <stdlib.h>
#include <stdio.h>

typedef float(*calc)(float a, float b, float c);


static float calc0(float a, float b, float c)
{
	return a + b + c;
}

static float calc1(float a, float b, float c)
{
	return a*b*c;
}

int main(int argc, char **argv)
{

	float sum = 0;
	calc c = calc0;

	for (int i = 0; i < 2000000000; ++i) {
		sum += c((float)i, 2, 3);
	}

	printf("sum: %f\n", sum);

	return 0;
}