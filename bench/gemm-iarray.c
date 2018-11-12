//
// Created by Francesc Alted on 25/09/2018.
//

/*
  Example program demonstrating how to execute an expression with super-chunks as operands.
  This is the version for using frames (either in-memory or on-disk) backing the super-chunks.

  To run:

  $ ./gemm-iarray memory
  ...
  $ ./gemm-iarray disk
  ...

*/

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <libiarray/iarray.h>
#include <mkl.h>

#define KB (1024.)
#define MB (1024 * KB)
#define GB (1024 * MB)

#define N (1000)   // array size is (N * N)
#define P (1000)   // partition size
#define NELEM (N * N)
#define NTHREADS 1


// Simple matrix-matrix multiplication for square matrices
int simple_matmul(size_t n, double const *a, double const *b, double *c)
{
    size_t i, j, k;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            double t = 0.0;
            for (k = 0; k < n; ++k)
                t += a[i*n+k] * b[k*n+j];
            c[i*n+j] = t;
        }
    }
    return 0;
}


// Check that the values of a super-chunk are equal to a C matrix
bool test_mat_equal(double *c1, double *c2) {
    for (int nelem=0; nelem < NELEM; nelem++) {
        double vdiff = fabs((c1[nelem] - c2[nelem]) / c1[nelem]);
        if (vdiff > 1e-6) {
            printf("%f, %f\n", c1[nelem], c2[nelem]);
            printf("Values differ in (%d nelem) (diff: %f)\n", nelem, vdiff);
            return false;
        }
    }
    return true;
}


int main(int argc, char** argv)
{
    long n = N;
    long nelem = NELEM;

    printf("Blosc version info: %s (%s)\n",
            BLOSC_VERSION_STRING, BLOSC_VERSION_DATE);

    ina_app_init(argc, argv, NULL);

    bool diskframes = false;
    if (argc > 1) {
        if (*argv[1] == 'd') {
            diskframes = true;
            printf("Storage for iarray matrices: *disk*\n");
        }
    }
    if (!diskframes) {
        printf("Storage for iarray matrices: *memory*\n");
    }

    if (argc > 2) {
        n = strtol(argv[2], NULL, 10);
        nelem = n * n;
    }
    printf("Measuring time for multiplying matrices of (%ld, %ld), with a partition of (%d, %d)\n", n, n, P, P);
    printf("Working set for the 4 uncompressed matrices: %.1f MB\n", n * n * sizeof(double) * 4 / MB);

    blosc_init();

    blosc2_cparams cparams = BLOSC_CPARAMS_DEFAULTS;
    blosc2_dparams dparams = BLOSC_DPARAMS_DEFAULTS;
    blosc_timestamp_t last, current;
    double ttotal;

    // Fill the plain C buffers for x, y matrices
    double *mat_x = malloc(nelem * sizeof(double));
    double *mat_y = malloc(nelem * sizeof(double));
    double *mat_out = malloc(nelem * sizeof(double));
    blosc_set_timestamp(&last);
    double incx = 10. / nelem;
    for (int i = 0; i < nelem; i++) {
        mat_x[i] = i * incx;
        mat_y[i] = i * incx;
    }
    blosc_set_timestamp(&current);
    ttotal = blosc_elapsed_secs(last, current);
//    printf("Time for filling X and Y matrices: %.3g s, %.1f MB/s\n",
//           ttotal, (sizeof(mat_x) + sizeof(mat_y)) / (ttotal * MB));

    // Compute matrix-matrix multiplication
    blosc_set_timestamp(&last);
    //simple_matmul(n, mat_x, mat_y, mat_out);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                1.0, mat_x, n, mat_y, n, 1.0, mat_out, n);

    blosc_set_timestamp(&current);
    ttotal = blosc_elapsed_secs(last, current);
    printf("Time for multiplying two matrices (pure C): %.3g s, %.1f GFlop/s, %.1f MB/s\n",
        ttotal, ((double)n * (double)n * (double)n * 2) / (ttotal * 1e9),
        (n * n * 3) / (ttotal * MB));

    /* Create a super-chunk container for input (X values) */
    cparams.typesize = sizeof(double);
    cparams.compcode = BLOSC_LZ4;
    cparams.clevel = 9;
    cparams.filters[0] = BLOSC_TRUNC_PREC;
    cparams.filters_meta[0] = 23;  // treat doubles as floats
    cparams.blocksize = 16 * (int)KB;  // 16 KB seems optimal for evaluating expressions
    cparams.nthreads = NTHREADS;
    dparams.nthreads = NTHREADS;

    // Create Caterva arrays out of C buffers
    caterva_pparams pparams;
    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        pparams.shape[i] = 1;
        pparams.cshape[i] = 1;
    }
    pparams.shape[CATERVA_MAXDIM - 1] = n;  // FIXME: 1's at the beginning should be removed
    pparams.shape[CATERVA_MAXDIM - 2] = n;  // FIXME: 1's at the beginning should be removed
    pparams.cshape[CATERVA_MAXDIM - 1] = P;  // FIXME: 1's at the beginning should be removed
    pparams.cshape[CATERVA_MAXDIM - 2] = P;  // FIXME: 1's at the beginning should be removed
    pparams.ndims = 2;

    blosc2_frame frame_x = BLOSC_EMPTY_FRAME;
    if (diskframes) frame_x.fname = "x.b2frame";
    caterva_array *cta_x = caterva_new_array(cparams, dparams, &frame_x, pparams);
    blosc2_frame frame_y = BLOSC_EMPTY_FRAME;
    if (diskframes) frame_y.fname = "y.b2frame";
    caterva_array *cta_y = caterva_new_array(cparams, dparams, &frame_y, pparams);

    blosc_set_timestamp(&last);
    caterva_from_buffer(cta_x, mat_x);
    caterva_from_buffer(cta_y, mat_y);
    blosc_set_timestamp(&current);
	ttotal = blosc_elapsed_secs(last, current);
//	printf("Time for filling X values (compressed): %.3g s, %.1f MB/s\n",
//			ttotal, (cta_x->sc->nbytes * 2) / (ttotal * MB));
	printf("Compression for values in matrix X: %.1f MB -> %.1f MB (%.1fx)\n",
			(cta_x->sc->nbytes/MB), (cta_x->sc->cbytes/MB),
			((double) cta_x->sc->nbytes/cta_x->sc->cbytes));

	// Check that operands are the same
    caterva_to_buffer(cta_x, mat_x);
    caterva_to_buffer(cta_y, mat_y);
    if (!test_mat_equal(mat_x, mat_y)) {
        return -1;
    }
    free(mat_x);
    free(mat_y);

    // Check IronArray performance
    iarray_context_t *iactx;
    iarray_config_t cfg = {.max_num_threads = 1, .flags = IARRAY_EXPR_EVAL_CHUNK,
                           .cparams = &cparams, .dparams = &dparams, .pparams = &pparams};
    iarray_ctx_new(&cfg, &iactx);

    /* Create a super-chunk backed by an in-memory frame */
    blosc2_frame frame_out = BLOSC_EMPTY_FRAME;
    if (diskframes) frame_out.fname = "out2.b2frame";
    caterva_array *cta_out = caterva_new_array(cparams, dparams, &frame_out, pparams);

    iarray_container_t *iac_x, *iac_y, *iac_out;
    iarray_from_ctarray(iactx, cta_x, IARRAY_DATA_TYPE_DOUBLE, &iac_x);
    iarray_from_ctarray(iactx, cta_y, IARRAY_DATA_TYPE_DOUBLE, &iac_y);
    iarray_from_ctarray(iactx, cta_out, IARRAY_DATA_TYPE_DOUBLE, &iac_out);

    blosc_set_timestamp(&last);
    ina_rc_t errcode = iarray_gemm(iac_x, iac_y, iac_out);
    if (errcode < 0) {
        printf("Error in iarray_gemm()\n");
        return -1;
    }
    blosc_set_timestamp(&current);
    ttotal = blosc_elapsed_secs(last, current);
    blosc2_schunk *sc_out = cta_out->sc;
    //printf("\n");
    printf("Time for multiplying two matrices (iarray):  %.3g s, %.1f GFlop/s, %.1f MB/s\n",
            ttotal, ((double)n * (double)n * (double)n * 2) / (ttotal * 1e9),
           (n * n * 3) / (ttotal * MB));
    printf("Compression for values in matrix OUT: %.1f MB -> %.1f MB (%.1fx)\n",
            (sc_out->nbytes/MB), (sc_out->cbytes/MB),
            (1.*sc_out->nbytes) / sc_out->cbytes);

    // Check that we are getting the same results than through manual computation
    double *mat_out2 = malloc(nelem * sizeof(double));
    caterva_to_buffer(cta_out, mat_out2);
    if (!test_mat_equal(mat_out, mat_out2)) {
        return -1;
    }
    free(mat_out);

    // Free resources
    caterva_free_array(cta_x);
    caterva_free_array(cta_y);
    caterva_free_array(cta_out);

    blosc_destroy();

    return 0;
}
