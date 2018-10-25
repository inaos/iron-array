//
// Created by Francesc Alted on 25/09/2018.
//

/*
  Example program demonstrating how to execute an expression with super-chunks as operands.

  To compile this program:

  $ gcc -O3 vectors.c -o vectors -lblosc

  To run:

  $ ./vectors
  ...

*/

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <libiarray/iarray.h>

#define KB (1024.)
#define MB (1024 * KB)
#define GB (1024 * MB)

#define NCHUNKS  100
#define NITEMS_CHUNK (200 * 1000)  // fits well in modern L3 caches
#define NELEM (NCHUNKS * NITEMS_CHUNK)  // multiple of NITEMS_CHUNKS for now
#define NTHREADS 1

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
    static double buffer_x[NITEMS_CHUNK];

    /* Fill with even values between 0 and 10 */
    for (int nchunk = 0; nchunk<NCHUNKS; nchunk++) {
        fill_buffer(buffer_x, nchunk);
        blosc2_schunk_append_buffer(sc_x, buffer_x, isize);
    }
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

// Compute and fill Y values in a buffer
void fill_buffer_y(const double* x, double* y)
{
    for (int i = 0; i<NITEMS_CHUNK; i++) {
        y[i] = poly(x[i]);
    }
}

// Check that two super-chunks with the same partitions are equal
bool test_schunks_equal(blosc2_schunk* sc1, blosc2_schunk* sc2) {
    size_t chunksize = (size_t)sc1->chunksize;
    int nitems_in_chunk = (int)chunksize / sc1->typesize;
    double *buffer_sc1 = malloc(chunksize);
    double *buffer_sc2 = malloc(chunksize);
    for (int nchunk=0; nchunk < sc1->nchunks; nchunk++) {
        int dsize = blosc2_schunk_decompress_chunk(sc1, nchunk, buffer_sc1, chunksize);
        dsize = blosc2_schunk_decompress_chunk(sc2, nchunk, buffer_sc2, chunksize);
        for (int nelem=0; nelem < nitems_in_chunk; nelem++) {
            double vdiff = fabs(buffer_sc1[nelem] - buffer_sc2[nelem]);
            if (vdiff > 1e-6) {
                printf("Values differ in (%d nchunk, %d nelem) (diff: %f)\n", nchunk, nelem, vdiff);
                free(buffer_sc1);
                free(buffer_sc2);
                return false;
            }
        }
    }
    free(buffer_sc1);
    free(buffer_sc2);
    return true;
}

int main(int argc, char** argv)
{
    printf("Blosc version info: %s (%s)\n",
            BLOSC_VERSION_STRING, BLOSC_VERSION_DATE);

    ina_app_init(argc, argv, NULL);

    blosc_init();

    const size_t isize = NITEMS_CHUNK * sizeof(double);
    static double buffer_x [NITEMS_CHUNK];
    static double buffer_y[NITEMS_CHUNK];
    blosc2_cparams cparams = BLOSC_CPARAMS_DEFAULTS;
    blosc2_dparams dparams = BLOSC_DPARAMS_DEFAULTS;
    blosc2_schunk *sc_x, *sc_y;
    blosc_timestamp_t last, current;
    double ttotal;

    /* Create a super-chunk container for input (X values) */
    cparams.typesize = sizeof(double);
    cparams.compcode = BLOSC_LZ4;
    cparams.clevel = 9;
    cparams.filters[0] = BLOSC_TRUNC_PREC;
    cparams.filters_meta[0] = 23;  // treat doubles as floats
    cparams.blocksize = 16 * (int)KB;  // 16 KB seems optimal for evaluating expressions
    cparams.nthreads = NTHREADS;
    dparams.nthreads = NTHREADS;

    // Fill the plain x operand
    static double x[NELEM];
    blosc_set_timestamp(&last);
    fill_x(x);
    blosc_set_timestamp(&current);
//	ttotal = blosc_elapsed_secs(last, current);
//	printf("Time for filling X values: %.3g s, %.1f MB/s\n",
//			ttotal, sizeof(x)/(ttotal*MB));

    // Create and fill a super-chunk for the x operand
    sc_x = blosc2_new_schunk(cparams, dparams, NULL);
    blosc_set_timestamp(&last);
    fill_sc_x(sc_x, isize);
    blosc_set_timestamp(&current);
//	ttotal = blosc_elapsed_secs(last, current);
//	printf("Time for filling X values (compressed): %.3g s, %.1f MB/s\n",
//			ttotal, (sc_x->nbytes/(ttotal*MB)));
//	printf("Compression for X values: %.1f MB -> %.1f MB (%.1fx)\n",
//			(sc_x->nbytes/MB), (sc_x->cbytes/MB),
//			((double) sc_x->nbytes/sc_x->cbytes));

    // Compute the plain y vector
    static double y[NELEM];
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
    for (int nchunk = 0; nchunk < sc_x->nchunks; nchunk++) {
        int dsize = blosc2_schunk_decompress_chunk(sc_x, nchunk, buffer_x, isize);
        if (dsize < 0) {
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

    // Check IronArray performance
    // First for the chunk evaluator
    iarray_context_t *iactx;
    iarray_config_t cfg = {.max_num_threads = 1, .flags = IARRAY_EXPR_EVAL_CHUNK, .cparams = &cparams, .dparams = &dparams};
    iarray_ctx_new(&cfg, &iactx);

    blosc2_schunk *sc_out = blosc2_new_schunk(cparams, dparams, NULL);

    iarray_expression_t *e;
    iarray_expr_new(iactx, &e);
    iarray_container_t *c_x, *c_y;
    iarray_from_sc(iactx, sc_x, IARRAY_DATA_TYPE_DOUBLE, &c_x);
    iarray_from_sc(iactx, sc_y, IARRAY_DATA_TYPE_DOUBLE, &c_y);
    iarray_expr_bind(e, "x", c_x);
    //iarray_expr_bind(e, "y", c_y);
    iarray_expr_compile(e, "(x - 1.35) * (x - 4.45) * (x - 8.5)");

    blosc_set_timestamp(&last);
    iarray_eval(iactx, e, sc_out, 0, NULL);
    blosc_set_timestamp(&current);
    ttotal = blosc_elapsed_secs(last, current);
    printf("\n");
    printf("Time for computing and filling OUT values using iarray (chunk eval):  %.3g s, %.1f MB/s\n",
            ttotal, sc_out->nbytes / (ttotal * MB));
    printf("Compression for OUT values: %.1f MB -> %.1f MB (%.1fx)\n",
            (sc_out->nbytes/MB), (sc_out->cbytes/MB),
            (1.*sc_out->nbytes)/sc_out->cbytes);

    // Check that we are getting the same results than through manual computation
    if (!test_schunks_equal(sc_y, sc_out)) {
        return -1;
    }

    iarray_expr_free(iactx, &e);
    iarray_ctx_free(&iactx);

    // Then for the block evaluator
    iarray_config_t cfg2 = {.max_num_threads = 1, .flags = IARRAY_EXPR_EVAL_BLOCK, .cparams = &cparams, .dparams = &dparams};
    iarray_ctx_new(&cfg2, &iactx);

    blosc2_schunk *sc_out2 = blosc2_new_schunk(cparams, dparams, NULL);

    iarray_expr_new(iactx, &e);
    iarray_from_sc(iactx, sc_x, IARRAY_DATA_TYPE_DOUBLE, &c_x);
    iarray_from_sc(iactx, sc_y, IARRAY_DATA_TYPE_DOUBLE, &c_y);
    iarray_expr_bind(e, "x", c_x);
    //iarray_expr_bind(e, "y", c_y);
    iarray_expr_compile(e, "(x - 1.35) * (x - 4.45) * (x - 8.5)");

    blosc_set_timestamp(&last);
    iarray_eval(iactx, e, sc_out2, 0, NULL);
    blosc_set_timestamp(&current);
    ttotal = blosc_elapsed_secs(last, current);
    printf("\n");
    printf("Time for computing and filling OUT values using iarray (block eval):  %.3g s, %.1f MB/s\n",
            ttotal, sc_out2->nbytes / (ttotal * MB));
    printf("Compression for OUT values: %.1f MB -> %.1f MB (%.1fx)\n",
            (sc_out2->nbytes/MB), (sc_out2->cbytes/MB),
            (1.*sc_out2->nbytes)/sc_out2->cbytes);

    // Check that we are getting the same results than through manual computation
    if (!test_schunks_equal(sc_y, sc_out2)) {
        return -1;
    }

    iarray_expr_free(iactx, &e);
    iarray_ctx_free(&iactx);

    // Free resources
    blosc2_free_schunk(sc_x);
    blosc2_free_schunk(sc_y);
    blosc2_free_schunk(sc_out);
    blosc2_free_schunk(sc_out2);

    blosc_destroy();

    return retcode;
}
