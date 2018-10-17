//
// Created by Francesc Alted on 15/10/2018.
//

#include "test_common.h"
#include "blosc.h"
#include <libiarray/iarray.h>

#define NCHUNKS  10
#define NITEMS_CHUNK (20 * 1000)
#define NELEM (NCHUNKS * NITEMS_CHUNK)  // multiple of NITEMS_CHUNKS for now
#define NTHREADS 1

/* Global vars */
int tests_run = 0;
blosc2_schunk *sc_x, *sc_y, *sc_out;
int nbytes, cbytes;
int clevel = 9;
iarray_context_t *iactx = NULL;

double x[NELEM];
double y[NELEM];
double buffer_x[NITEMS_CHUNK];
double buffer_y[NITEMS_CHUNK];

// Fill X values in regular array
int fill_x(double* x)
{
	double incx = 10./NELEM;

	/* Fill even values between 0 and 10 */
	for (int i = 0; i<NELEM; i++) {
		x[i] = incx*i;
	}
	return 0;
}

double poly(const double x)
{
	return (x - 1.35) * (x - 4.45) * (x - 8.5);
}

// Compute and fill Y values in regular array
void compute_y(const double* x, double* y)
{
	for (int i = 0; i<NELEM; i++) {
		y[i] = poly(x[i]);
	}
}

// Compute and fill X values in a buffer
void fill_buffer(double* x, int nchunk)
{
	double incx = 10./NELEM;

	for (int i=0; i<NITEMS_CHUNK; i++) {
		x[i] = incx*(nchunk * NITEMS_CHUNK + i);
	}
}

void fill_sc_x(blosc2_schunk* sc_x, const size_t isize)
{
	double buffer_x[NITEMS_CHUNK];

	/* Fill with even values between 0 and 10 */
	for (int nchunk = 0; nchunk<NCHUNKS; nchunk++) {
		fill_buffer(buffer_x, nchunk);
		blosc2_schunk_append_buffer(sc_x, buffer_x, isize);
	}
}

// Compute and fill Y values in a buffer
void fill_buffer_y(const double* x, double* y)
{
	for (int i = 0; i<NITEMS_CHUNK; i++) {
		y[i] = poly(x[i]);
	}
}

static char *test_eval_chunk1() {
	//iarray_variable_t vars[] = {{"x", sc_x}, {"y", sc_y}};
	//iarray_variable_t out = {"out", sc_out};

    iarray_expression_t *e;
    iarray_expr_new(iactx, &e);
    iarray_container_t *c_x;
    iarray_from_sc(iactx, sc_x, IARRAY_DATA_TYPE_DOUBLE, &c_x);
    iarray_expr_bind(e, "x", c_x);

	iarray_expr_compile(e, "(x - 1.35) * (x - 4.45) * (x - 8.5)");
    iarray_eval(iactx, e, sc_out, 0, NULL);
	// Check that we are getting the same results than through manual computation
	if (!test_schunks_equal_double(sc_y, sc_out)) {
		return "Super-chunks are not equal";
	}

	iarray_expr_free(iactx, &e);

	return 0;
}

static char *all_tests() {
	mu_run_test(test_eval_chunk1);

	return 0;
}

int main(int argc, char **argv) {
	char *result;
	const size_t isize = NITEMS_CHUNK * sizeof(double);

	printf("STARTING TESTS for %s", argv[0]);

	ina_app_init(argc, argv, NULL);

	blosc_init();

	// Fill the plain x operand
	fill_x(x);
	// Compute the plain y vector
	compute_y(x, y);

	/* Create a super-chunk container for input (X values) */
	blosc2_cparams cparams = BLOSC_CPARAMS_DEFAULTS;
	blosc2_dparams dparams = BLOSC_DPARAMS_DEFAULTS;
	cparams.typesize = sizeof(double);
	cparams.compcode = BLOSC_LZ4;
	cparams.clevel = 9;
	cparams.filters[0] = BLOSC_TRUNC_PREC;
	cparams.filters_meta[0] = 23;  // treat doubles as floats
	cparams.blocksize = 16 * (int)KB;  // 16 KB seems optimal for evaluating expressions
	cparams.nthreads = NTHREADS;
	dparams.nthreads = NTHREADS;

    iarray_config_t cfg = {.max_num_threads = 1, .flags = 0, .cparams = &cparams, .dparams = &dparams};
    iarray_ctx_new(&cfg, &iactx);

	sc_x = blosc2_new_schunk(cparams, dparams, NULL);
	fill_sc_x(sc_x, isize);

	/* Create a super-chunk container for output (Y values) */
	sc_y = blosc2_new_schunk(cparams, dparams, NULL);
	for (int nchunk = 0; nchunk < sc_x->nchunks; nchunk++) {
		int dsize = blosc2_schunk_decompress_chunk(sc_x, nchunk, buffer_x, isize);
		if (dsize < 0) {
			printf("Decompression error.  Error code: %d\n", dsize);
			return dsize;
		}
		fill_buffer_y(buffer_x, buffer_y);
		blosc2_schunk_append_buffer(sc_y, buffer_y, isize);
	}

	/* Create a super-chunk container for eval output (OUT values) */
	sc_out = blosc2_new_schunk(cparams, dparams, NULL);

	/* Run all the suite */
	result = all_tests();
	if (result != 0) {
		printf(" (%s)\n", result);
	}
	else {
		printf(" ALL TESTS PASSED");
	}
	printf("\tTests run: %d\n", tests_run);


	return result != 0;
}
