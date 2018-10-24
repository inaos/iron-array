//
// Created by Francesc Alted on 15/10/2018.
//

#include "test_common.h"
#include "blosc.h"
#include <libiarray/iarray.h>

#define NCHUNKS  10
#define NITEMS_CHUNK (20 * 1000)
#define NELEM (((NCHUNKS - 1) * NITEMS_CHUNK) + 10)
#define NTHREADS 1

/* Global vars */
int tests_run = 0;
blosc2_schunk *sc_x, *sc_y, *sc_out;
int nbytes, cbytes;
int clevel = 9;

double buffer_x[NITEMS_CHUNK];
double buffer_y[NITEMS_CHUNK];

// Compute and fill X values in a buffer
void fill_buffer(double* x, int nchunk, int nitems)
{
    /* Fill with even values between 0 and 10 */
    double incx = 10./NELEM;

    for (int i = 0; i<nitems; i++) {
        x[i] = incx*(nchunk*NITEMS_CHUNK+i);
    }
}

void fill_sc_x(blosc2_schunk* sc_x)
{
    for (int nchunk = 0; nchunk<NCHUNKS; nchunk++) {
        int nitems = (nchunk < NCHUNKS - 1) ? NITEMS_CHUNK : NELEM - nchunk * NITEMS_CHUNK;
        fill_buffer(buffer_x, nchunk, nitems);
        blosc2_schunk_append_buffer(sc_x, buffer_x, nitems * sizeof(double));
    }
}

double poly(const double x)
{
    return (x-1.35)*(x-4.45)*(x-8.5);
}

// Compute and fill Y values in a buffer
void fill_buffer_y(const double* x, double* y, int nitems)
{
    for (int i = 0; i<nitems; i++) {
        y[i] = poly(x[i]);
    }
}

static char* test_eval_chunk1()
{
    iarray_context_t *iactx;
    iarray_config_t cfg = {.max_num_threads = NTHREADS, .flags = IARRAY_EXPR_EVAL_CHUNK};
    iarray_ctx_new(&cfg, &iactx);
    iarray_expression_t* e;
    iarray_expr_new(iactx, &e);
    iarray_container_t* c_x;
    iarray_from_sc(iactx, sc_x, IARRAY_DATA_TYPE_DOUBLE, &c_x);
    iarray_expr_bind(e, "x", c_x);

    iarray_expr_compile(e, "(x - 1.35) * (x - 4.45) * (x - 8.5)");
    iarray_eval(iactx, e, sc_out, 0, NULL);
    // Check that we are getting the same results than through manual computation
    if (!test_schunks_equal_double(sc_y, sc_out)) {
        return "Super-chunks are not equal";
    }

    iarray_expr_free(iactx, &e);
    iarray_ctx_free(&iactx);

    return 0;
}

static char* test_eval_block1()
{
    iarray_context_t *iactx;
    iarray_config_t cfg = {.max_num_threads = NTHREADS, .flags = IARRAY_EXPR_EVAL_BLOCK};
    iarray_ctx_new(&cfg, &iactx);
    iarray_expression_t* e;
    iarray_expr_new(iactx, &e);
    iarray_container_t* c_x;
    iarray_from_sc(iactx, sc_x, IARRAY_DATA_TYPE_DOUBLE, &c_x);
    iarray_expr_bind(e, "x", c_x);

    iarray_expr_compile(e, "(x - 1.35) * (x - 4.45) * (x - 8.5)");
    iarray_eval(iactx, e, sc_out, 0, NULL);
    // Check that we are getting the same results than through manual computation
    if (!test_schunks_equal_double(sc_y, sc_out)) {
        return "Super-chunks are not equal";
    }

    iarray_expr_free(iactx, &e);
    iarray_ctx_free(&iactx);

    return 0;
}

static char* all_tests()
{
    mu_run_test(test_eval_chunk1);
    mu_run_test(test_eval_block1);

    return 0;
}

int main(int argc, char** argv)
{
    char* result;
    const size_t isize = NITEMS_CHUNK*sizeof(double);

    printf("STARTING TESTS for %s", argv[0]);

    ina_app_init(argc, argv, NULL);

    blosc_init();

    /* Create a super-chunk container for input (X values) */
    blosc2_cparams cparams = BLOSC_CPARAMS_DEFAULTS;
    blosc2_dparams dparams = BLOSC_DPARAMS_DEFAULTS;
    cparams.typesize = sizeof(double);
    cparams.compcode = BLOSC_LZ4;
    cparams.clevel = 9;
    cparams.filters[0] = BLOSC_TRUNC_PREC;
    cparams.filters_meta[0] = 23;  // treat doubles as floats
    cparams.blocksize = 16*(int) KB;  // 16 KB seems optimal for evaluating expressions
    cparams.nthreads = NTHREADS;
    dparams.nthreads = NTHREADS;

    sc_x = blosc2_new_schunk(cparams, dparams, NULL);
    fill_sc_x(sc_x);

    /* Create a super-chunk container for output (Y values) */
    sc_y = blosc2_new_schunk(cparams, dparams, NULL);
    for (int nchunk = 0; nchunk<sc_x->nchunks; nchunk++) {
        int dsize = blosc2_schunk_decompress_chunk(sc_x, nchunk, buffer_x, isize);
        if (dsize<0) {
            printf("Decompression error.  Error code: %d\n", dsize);
            return dsize;
        }
        int nitems = (nchunk < NCHUNKS - 1) ? NITEMS_CHUNK : NELEM - nchunk * NITEMS_CHUNK;
        fill_buffer_y(buffer_x, buffer_y, nitems);
        blosc2_schunk_append_buffer(sc_y, buffer_y, nitems * sizeof(double));
    }

    /* Create a super-chunk container for eval output (OUT values) */
    sc_out = blosc2_new_schunk(cparams, dparams, NULL);

    /* Run all the suite */
    result = all_tests();
    if (result!=0) {
        printf(" (%s)\n", result);
    } else {
        printf(" ALL TESTS PASSED");
    }
    printf("\tTests run: %d\n", tests_run);

    return result!=0;
}
