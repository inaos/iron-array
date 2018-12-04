/*
 * Copyright INAOS GmbH, Thalwil, 2018.
 * Copyright Francesc Alted, 2018.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of INAOS GmbH
 * and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <libiarray/iarray.h>
#include <iarray_private.h>

#define P (100)  /* partition size */  
#define NELEM_BYTES(nelem) (nelem*sizeof(double))
#define NTHREADS 1

/* Simple matrix-matrix multiplication for square matrices */
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


/* Check that the values of a super-chunk are equal to a C matrix */
int test_mat_equal(int nelems, double *c1, double *c2) {
    for (int nelem=0; nelem < nelems; nelem++) {
        double vdiff = fabs((c1[nelem] - c2[nelem]) / c1[nelem]);
        if (vdiff > 1e-6) {
            printf("%f, %f\n", c1[nelem], c2[nelem]);
            printf("Values differ in (%d nelem) (diff: %f)\n", nelem, vdiff);
            return 0;
        }
    }
    return 1;
}

static double *mat_x = NULL;
static double *mat_y = NULL;
static double *mat_out = NULL;

static void ina_cleanup_handler(int error, int *exitcode)
{
    iarray_destroy();
}

int main(int argc, char** argv)
{
    ina_stopwatch_t *w = NULL;
    iarray_context_t *ctx = NULL;
    const char *mat_x_name = NULL;
    const char *mat_y_name = NULL;
    const char *mat_out_name = NULL;
    int n = 0;
    int nelem = 0;

    INA_OPTS(opt,
        INA_OPT_FLAG("p", "persistence", "Use persistent containers"),
        INA_OPT_INT("n", "elements", 1000, "Number of Elements")
    );

    if (!INA_SUCCEED(ina_app_init(argc, argv, opt))) {
        return EXIT_FAILURE;
    }
    ina_set_cleanup_handler(ina_cleanup_handler);

    if (INA_SUCCEED(ina_opt_isset("p"))) {
        mat_x_name = "mat_x";
        mat_y_name = "mat_y";
        mat_out_name = "mat_out";
    }
    else {
        printf("Storage for iarray matrices: *memory*\n");
    }
    INA_MUST_SUCCEED(ina_opt_get_int("n", &n));
    nelem = n * n;
    
    printf("Measuring time for multiplying matrices of (%ld, %ld), with a partition of (%d, %d)\n", n, n, P, P);
    printf("Working set for the 4 uncompressed matrices: %.1f MB\n", nelem * sizeof(double) * 4 / (double)_IARRAY_SIZE_MB);

    INA_MUST_SUCCEED(iarray_init());

    iarray_config_t config = IARRAY_CONFIG_DEFAULTS;
    config.compression_codec = IARRAY_COMPRESSION_LZ4;
    config.compression_level = 5;
    config.max_num_threads = NTHREADS;
    config.flags = IARRAY_EXPR_EVAL_CHUNK;

    INA_MUST_SUCCEED(iarray_context_new(&config, &ctx));

    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    mat_x = (double*)ina_mem_alloc((sizeof(double)*nelem));
    mat_y = (double*)ina_mem_alloc((sizeof(double)*nelem));
    mat_out = (double*)ina_mem_alloc((sizeof(double)*nelem));

    INA_STOPWATCH_START(w);
    double incx = 10. / nelem;
    for (int i = 0; i < nelem; i++) {
        mat_x[i] = i * incx;
        mat_y[i] = i * incx;
    }
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time for filling X and Y matrices: %.3g s, %.1f MB/s\n",
        elapsed_sec, (sizeof(mat_x) + sizeof(mat_y)) / (elapsed_sec * _IARRAY_SIZE_MB));

    /* Compute naive matrix-matrix multiplication */
    INA_STOPWATCH_START(w);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
        1.0, mat_x, n, mat_y, n, 1.0, mat_out, n);
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time for multiplying two matrices (pure C): %.3g s, %.1f MB/s\n",
        elapsed_sec, (sizeof(mat_x) * 3) / (elapsed_sec * _IARRAY_SIZE_MB));

    iarray_dtshape_t shape;
    shape.ndim = 2;
    shape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    shape.shape[0] = n;
    shape.shape[1] = n;
    shape.partshape[0] = P;
    shape.partshape[1] = P;

    iarray_container_t *con_x;
    iarray_container_t *con_y;

    INA_STOPWATCH_START(w);
    INA_MUST_SUCCEED(iarray_from_buffer(ctx, &shape, mat_x, n, mat_x_name, 0, &con_x));
    INA_MUST_SUCCEED(iarray_from_buffer(ctx, &shape, mat_y, n, mat_y_name, 0, &con_y));
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    size_t nbytes = 0;
    size_t cbytes = 0;
    double nbytes_mb = 0;
    double cbytes_mb = 0;
    iarray_container_info(con_x, &nbytes, &cbytes);
    printf("Time for filling X and Y iarray-containers: %.3g s, %.1f MB/s\n",
        elapsed_sec, (nbytes * 2) / (elapsed_sec * _IARRAY_SIZE_MB));
    nbytes_mb = (nbytes / _IARRAY_SIZE_MB);
    cbytes_mb = (cbytes / _IARRAY_SIZE_MB);
    printf("Compression for X iarray-container: %.1f MB -> %.1f MB (%.1fx)\n",
        nbytes_mb, cbytes_mb, ((double)nbytes / cbytes));

    INA_MUST_SUCCEED(iarray_to_buffer(ctx, con_x, mat_x, NELEM_BYTES(nelem)));
    INA_MUST_SUCCEED(iarray_to_buffer(ctx, con_y, mat_y, NELEM_BYTES(nelem)));
    if (!test_mat_equal(nelem, mat_x, mat_y)) {
        return EXIT_FAILURE; /* FIXME: error handling */
    }

    iarray_container_t *con_out;
    iarray_container_new(ctx, &shape, mat_out_name, 0, &con_out);

    INA_STOPWATCH_START(w);
    iarray_matmul(con_x, con_y, con_out, IARRAY_OPERATION_GENERAL); /* FIXME: error handling */
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    iarray_container_info(con_out, &nbytes, &cbytes);
    printf("\n");
    printf("Time for multiplying two matrices (iarray):  %.3g s, %.1f MB/s\n",
        elapsed_sec, (nbytes * 3) / (elapsed_sec * _IARRAY_SIZE_MB));
    nbytes_mb = (nbytes / _IARRAY_SIZE_MB);
    cbytes_mb = (cbytes / _IARRAY_SIZE_MB);
    printf("Compression for OUT values: %.1f MB -> %.1f MB (%.1fx)\n",
            nbytes_mb, cbytes_mb, (1.*nbytes) / cbytes);

    /* Check that we are getting the same results than through manual computation */
    ina_mem_set(mat_out, 0, NELEM_BYTES(nelem));
    iarray_to_buffer(ctx, con_out, mat_out, NELEM_BYTES(nelem));
    if (!test_mat_equal(nelem, mat_out, mat_out)) {
        return EXIT_FAILURE; /* FIXME: error-handling */
    }

    iarray_container_free(ctx, &con_x);
    iarray_container_free(ctx, &con_y);
    iarray_container_free(ctx, &con_out);

    iarray_context_free(&ctx);

    ina_mem_free(mat_x);
    ina_mem_free(mat_y);
    ina_mem_free(mat_out);

    INA_STOPWATCH_FREE(&w);

    return EXIT_SUCCESS;
}
