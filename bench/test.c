
#include <stdlib.h>
#include <stdio.h>

typedef float (*calc)(float a, float b, float c);


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
	int test = atoi(argv[1]);

	calc c;
	float sum = 0;

	for (int i = 0; i < 2000000000; ++i) {
		switch (test) {
		case 0:
			c = calc0;
			break;
		case 1:
			c = calc1;
			break;
		}
		sum += c((float)i, 2, 3);
	}

	printf("sum: %f\n", sum);

	return 0;
}