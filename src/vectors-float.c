//
// Created by Francesc Alted on 04/10/2018.
//

/*
  Example program demonstrating how to execute an expression with super-chunks as operands.

  To compile this program:

  $ gcc -O3 vectors-float.c -o vectors-float -lblosc

  To run:

  $ ./vectors
  ...

*/

#include <stdio.h>
#include "iarray.h"

#define KB (1024.)
#define MB (1024 * KB)
#define GB (1024 * MB)

#define NCHUNKS  50
#define CHUNKSIZE (200 * 1000)  // fits well in modern L3 caches
#define NELEM (NCHUNKS * CHUNKSIZE)  // multiple of CHUNKSIZE for now
#define NTHREADS  4

// Fill X values in regular array
int fill_x(float* x)
{
	float incx = 10.f/NELEM;

	/* Fill even values between 0 and 10 */
	for (int i = 0; i<NELEM; i++) {
		x[i] = incx*i;
	}
	return 0;
}

// Compute and fill X values in a buffer
void fill_buffer(float* x, int nchunk)
{
	float incx = 10.f/NELEM;

	for (int i = 0; i<CHUNKSIZE; i++) {
		x[i] = incx*(nchunk*CHUNKSIZE+i);
	}
}

void fill_sc_x(blosc2_schunk* sc_x, const size_t isize)
{
	float buffer_x[CHUNKSIZE];

	/* Fill with even values between 0 and 10 */
	for (int nchunk = 0; nchunk<NCHUNKS; nchunk++) {
		fill_buffer(buffer_x, nchunk);
		blosc2_schunk_append_buffer(sc_x, buffer_x, isize);
	}
}

float poly(const float x)
{
	return (x - 1.35f) * (x - 4.45f) * (x - 8.5f);
}

// Compute and fill Y values in regular array
void compute_y(const float* x, float* y)
{
	for (int i = 0; i<NELEM; i++) {
		y[i] = poly(x[i]);
	}
}

// Compute and fill Y values in a buffer
void fill_buffer_y(const float* x, float* y)
{
	for (int i = 0; i<CHUNKSIZE; i++) {
		y[i] = poly(x[i]);
	}
}

int main(int argc, char** argv)
{
	printf("Blosc version info: %s (%s)\n",
			BLOSC_VERSION_STRING, BLOSC_VERSION_DATE);

	blosc_init();

	const size_t isize = CHUNKSIZE*sizeof(float);
	float buffer_x[CHUNKSIZE];
	float buffer_y[CHUNKSIZE];
	int dsize;
	blosc2_cparams cparams = BLOSC_CPARAMS_DEFAULTS;
	blosc2_dparams dparams = BLOSC_DPARAMS_DEFAULTS;
	blosc2_schunk *sc_x, *sc_y;
	int nchunk;
	blosc_timestamp_t last, current;
	double ttotal;

	/* Create a super-chunk container for input (X values) */
	cparams.typesize = sizeof(float);
	cparams.compcode = BLOSC_LZ4;
	cparams.clevel = 9;
	cparams.blocksize = 16 * (int)KB;
	cparams.nthreads = NTHREADS;
	dparams.nthreads = NTHREADS;

	// Fill the plain x operand
	static float x[NELEM];
	blosc_set_timestamp(&last);
	fill_x(x);
	blosc_set_timestamp(&current);
	ttotal = blosc_elapsed_secs(last, current);
	printf("Time for filling X values: %.3g s, %.1f MB/s\n",
			ttotal, sizeof(x)/(ttotal*MB));

	// Create and fill a super-chunk for the x operand
	sc_x = blosc2_new_schunk(cparams, dparams, NULL);
	blosc_set_timestamp(&last);
	fill_sc_x(sc_x, isize);
	blosc_set_timestamp(&current);
	ttotal = blosc_elapsed_secs(last, current);
	printf("Time for filling X values (compressed): %.3g s, %.1f MB/s\n",
			ttotal, (sc_x->nbytes/(ttotal*MB)));
	printf("Compression for X values: %.1f MB -> %.1f MB (%.1fx)\n",
			(sc_x->nbytes/MB), (sc_x->cbytes/MB),
			((double) sc_x->nbytes/sc_x->cbytes));

	// Compute the plain y vector
	static float y[NELEM];
	blosc_set_timestamp(&last);
	compute_y(x, y);
	blosc_set_timestamp(&current);
	ttotal = blosc_elapsed_secs(last, current);
	printf("Time for computing and filling Y values: %.3g s, %.1f MB/s\n",
			ttotal, sizeof(y)/(ttotal*MB));
	// To prevent the optimizer going too smart and removing 'dead' code
	int retcode = y[0] > y[1];

	// Create a super-chunk container and compute y values
	sc_y = blosc2_new_schunk(cparams, dparams, NULL);
	blosc_set_timestamp(&last);
	for (nchunk = 0; nchunk<sc_x->nchunks; nchunk++) {
		dsize = blosc2_schunk_decompress_chunk(sc_x, nchunk, buffer_x, isize);
		if (dsize<0) {
			printf("Decompression error.  Error code: %d\n", dsize);
			return dsize;
		}
		fill_buffer_y(buffer_x, buffer_y);
		blosc2_schunk_append_buffer(sc_y, buffer_y, isize);
	}
	blosc_set_timestamp(&current);
	ttotal = blosc_elapsed_secs(last, current);
	printf("Time for computing and filling Y values (compressed): %.3g s, %.1f MB/s\n",
			ttotal, sc_y->nbytes/(ttotal*MB));
	printf("Compression for Y values: %.1f MB -> %.1f MB (%.1fx)\n",
			(sc_y->nbytes/MB), (sc_y->cbytes/MB),
			(1.*sc_y->nbytes)/sc_y->cbytes);
	dsize = blosc2_schunk_decompress_chunk(sc_y, 0, buffer_y, isize);
	printf("first value of Y: %f\n", buffer_y[0]);
	dsize = blosc2_schunk_decompress_chunk(sc_y, sc_y->nchunks - 1, buffer_y, isize);
	printf("last value of Y: %f\n", buffer_y[CHUNKSIZE - 1]);

	// Check IronArray performance
	// First for chunk evaluator
	iarray_variable_t vars[] = {{"x", sc_x}, {"y", sc_y}};
	blosc2_schunk *sc_out = blosc2_new_schunk(cparams, dparams, NULL);
	iarray_variable_t out = {"out", sc_out};

	int err;
	blosc_set_timestamp(&last);
	//iarray_eval("x + y", vars, 2, out, IARRAY_DATA_TYPE_FLOAT, &err);
	iarray_eval_chunk("(x - 1.35) * (x - 4.45) * (x - 8.5)", vars, 1, out, IARRAY_DATA_TYPE_FLOAT, &err);
	blosc_set_timestamp(&current);
	ttotal = blosc_elapsed_secs(last, current);
	printf("\n");
	printf("Time for computing and filling OUT values using iarray (chunk eval):  %.3g s, %.1f MB/s\n",
			ttotal, sc_out->nbytes / (ttotal * MB));
	printf("Compression for OUT values: %.1f MB -> %.1f MB (%.1fx)\n",
			(sc_out->nbytes/MB), (sc_out->cbytes/MB),
			(1.*sc_out->nbytes)/sc_out->cbytes);
//	dsize = blosc2_schunk_decompress_chunk(sc_out, 0, buffer_y, isize);
//	printf("first value of OUT: %f\n", buffer_y[0]);
//	dsize = blosc2_schunk_decompress_chunk(sc_out, sc_out->nchunks - 1, buffer_y, isize);
//	printf("last value of OUT: %f\n", buffer_y[CHUNKSIZE - 1]);

	// Then for block evaluator
	blosc2_free_schunk(sc_out);
	sc_out = blosc2_new_schunk(cparams, dparams, NULL);
	iarray_variable_t out2 = {"out", sc_out};
	blosc_set_timestamp(&last);
	//iarray_eval("x + y", vars, 2, out, IARRAY_DATA_TYPE_FLOAT, &err);
	iarray_eval_block("(x - 1.35) * (x - 4.45) * (x - 8.5)", vars, 1, out2, IARRAY_DATA_TYPE_FLOAT, &err);
	blosc_set_timestamp(&current);
	ttotal = blosc_elapsed_secs(last, current);
	printf("\n");
	printf("Time for computing and filling OUT values using iarray (block eval):  %.3g s, %.1f MB/s\n",
			ttotal, sc_out->nbytes / (ttotal * MB));
	printf("Compression for OUT values: %.1f MB -> %.1f MB (%.1fx)\n",
			(sc_out->nbytes/MB), (sc_out->cbytes/MB),
			(1.*sc_out->nbytes)/sc_out->cbytes);
//	dsize = blosc2_schunk_decompress_chunk(sc_out, 0, buffer_y, isize);
//	printf("first value of OUT: %f\n", buffer_y[0]);
//	dsize = blosc2_schunk_decompress_chunk(sc_out, sc_out->nchunks - 1, buffer_y, isize);
//	printf("last value of OUT: %f\n", buffer_y[CHUNKSIZE - 1]);


	// Free resources
	blosc2_free_schunk(sc_x);
	blosc2_free_schunk(sc_y);
	blosc2_free_schunk(sc_out);

	blosc_destroy();

	return retcode;
}
