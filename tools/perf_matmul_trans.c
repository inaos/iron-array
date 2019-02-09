/*
 * Copyright INAOS GmbH, Thalwil, 2018.
 * Copyright Francesc Alted, 2018.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of INAOS GmbH
 * and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 */

#include <libiarray/iarray.h>
#include <iarray_private.h>

#define NELEM_BYTES(nelem) (nelem * sizeof(double))
#define NTHREADS 1

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
static double *mat_res = NULL;

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

    uint64_t nbytes = 0;
    uint64_t cbytes = 0;
    double nbytes_mb = 0;
    double cbytes_mb = 0;

    uint64_t xshape[] = {3230, 4056};
    uint64_t xpshape[] = {300, 675};
    uint64_t xbshape[] = {800, 400};
    int xtrans = 1;
    int xflag = CblasNoTrans;

    uint64_t xsize = xshape[0] * xshape[1];

    uint64_t yshape[] = {3712, 3230};
    uint64_t ypshape[] = {478, 300};
    uint64_t ybshape [] = {400, 600};
    int ytrans = 1;
    int yflag = CblasNoTrans;

    uint64_t ysize = yshape[0] * yshape[1];

    uint64_t oshape[] = {xshape[1], yshape[0]};
    uint64_t opshape[] = {xbshape[0], ybshape[1]};

    uint64_t osize = oshape[0] * oshape[1];

    uint64_t flops = (2 * xshape[1] - 1) * xshape[0] * yshape[1];

    INA_OPTS(opt,
        INA_OPT_FLAG("p", "persistence", "Use persistent containers"),
        INA_OPT_FLAG("r", "remove", "Remove the previous persistent containers (only valid w/ -p)")
    );

    if (!INA_SUCCEED(ina_app_init(argc, argv, opt))) {
        return EXIT_FAILURE;
    }
    ina_set_cleanup_handler(ina_cleanup_handler);

    if (INA_SUCCEED(ina_opt_isset("p"))) {
        mat_x_name = "mat_x_t.b2frame";
        mat_y_name = "mat_y_t.b2frame";
        mat_out_name = "mat_out_t.b2frame";
        if (INA_SUCCEED(ina_opt_isset("r"))) {
            remove(mat_x_name);
            remove(mat_y_name);
            remove(mat_out_name);
            printf("Storage for iarray matrices: *memory*\n");
        } else {
            printf("Storage for iarray matrices: *disk*\n");
        }
    } else {
        printf("Storage for iarray matrices: *memory*\n");
    }

    iarray_store_properties_t mat_x_prop = {.id = mat_x_name};
    iarray_store_properties_t mat_y_prop = {.id = mat_y_name};
    iarray_store_properties_t mat_out_prop = {.id = mat_out_name};

    printf("\n");
    printf("Measuring time for multiplying matrices X and Y\n");

    printf("\n");
    printf("Matrix X has a shape of (%lld, %lld) with a partition of (%lld, %lld) \n",
           xshape[0], xshape[1], xpshape[0], xpshape[1]);
    printf("Matrix Y has a shape of (%lld, %lld) with a partition of (%lld, %lld) \n",
           yshape[0], yshape[1], ypshape[0], ypshape[1]);

    printf("\n");
    printf("Working set for the 4 uncompressed matrices: %.1f MB\n", (xsize + ysize + osize * 2) * sizeof(double) / (double)_IARRAY_SIZE_MB);

    INA_MUST_SUCCEED(iarray_init());

    iarray_config_t config = IARRAY_CONFIG_DEFAULTS;
    config.compression_codec = IARRAY_COMPRESSION_LZ4;
    config.compression_level = 5;
    config.max_num_threads = NTHREADS;
    config.eval_flags = IARRAY_EXPR_EVAL_CHUNK;

    INA_MUST_SUCCEED(iarray_context_new(&config, &ctx));

    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    iarray_container_t *con_x;
    iarray_container_t *con_y;

    int flags = INA_SUCCEED(ina_opt_isset("p"))? IARRAY_CONTAINER_PERSIST : 0;

    bool allocated = false;

    mat_x = (double *) ina_mem_alloc((sizeof(double) * xsize));
    mat_y = (double *) ina_mem_alloc((sizeof(double) * ysize));

    printf("\n");
    if (INA_SUCCEED(ina_opt_isset("p")) && _iarray_file_exists(mat_x_prop.id) && _iarray_file_exists(mat_y_prop.id)) {
        INA_STOPWATCH_START(w);
        INA_MUST_SUCCEED(iarray_from_file(ctx, &mat_x_prop, &con_x));
        INA_MUST_SUCCEED(iarray_from_file(ctx, &mat_y_prop, &con_y));
        INA_STOPWATCH_STOP(w);
        INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
        printf("Time for *opening* X and Y values: %.3g s, %.1f GB/s\n",
               elapsed_sec, NELEM_BYTES(xsize + ysize) / (elapsed_sec * _IARRAY_SIZE_GB));
    } else {

        iarray_dtshape_t xdtshape;
        xdtshape.ndim = 2;
        xdtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
        for (int i = 0; i < xdtshape.ndim; ++i) {
            xdtshape.shape[i] = xshape[i];
            xdtshape.pshape[i] = xpshape[i];
        }

        iarray_dtshape_t ydtshape;
        ydtshape.ndim = 2;
        ydtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
        for (int i = 0; i < ydtshape.ndim; ++i) {
            ydtshape.shape[i] = yshape[i];
            ydtshape.pshape[i] = ypshape[i];
        }


        INA_STOPWATCH_START(w);
        INA_MUST_SUCCEED(iarray_linspace(ctx, &xdtshape, xsize, 0, 10, &mat_x_prop, flags, &con_x));
        INA_MUST_SUCCEED(iarray_linspace(ctx, &ydtshape, ysize, 0, 10, &mat_y_prop, flags, &con_y));
        INA_STOPWATCH_STOP(w);
        INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

        iarray_container_info(con_x, &nbytes, &cbytes);
        printf("Time for filling X and Y iarray-containers: %.3g s, %.1f MB/s\n",
               elapsed_sec, NELEM_BYTES(xsize + ysize) / (elapsed_sec * _IARRAY_SIZE_MB));
        nbytes_mb = ((double) nbytes / _IARRAY_SIZE_MB);
        cbytes_mb = ((double) cbytes / _IARRAY_SIZE_MB);
        printf("Compression for X iarray-container: %.1f MB -> %.1f MB (%.1fx)\n",
               nbytes_mb, cbytes_mb, ((double) nbytes / cbytes));
    }

    INA_MUST_SUCCEED(iarray_to_buffer(ctx, con_x, mat_x, NELEM_BYTES(xsize)));
    INA_MUST_SUCCEED(iarray_to_buffer(ctx, con_y, mat_y, NELEM_BYTES(ysize)));

    if (xtrans == 1) {
        xflag = CblasTrans;
        INA_MUST_SUCCEED(iarray_linalg_transpose(ctx, con_x));
    }

    if (ytrans == 1) {
        yflag = CblasTrans;
        INA_MUST_SUCCEED(iarray_linalg_transpose(ctx, con_y));
    }

    mat_out = (double *) ina_mem_alloc((sizeof(double) * osize));
    mat_res = (double *) ina_mem_alloc((sizeof(double) * osize));


    int M = (int) con_x->dtshape->shape[0];
    int K = (int) con_x->dtshape->shape[1];
    int N = (int) con_y->dtshape->shape[1];

    int ldx = K;
    if (xtrans == 1) {
        ldx = M;
    }
    int ldy = N;
    if (ytrans == 1) {
        ldy = K;
    }
    int ldr = N;
    /* Compute naive matrix-matrix multiplication */
    INA_STOPWATCH_START(w);
    cblas_dgemm(CblasRowMajor, xflag, yflag, M, N, K, 1.0, mat_x, ldx, mat_y, ldy, 0.0, mat_res, ldr);
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("\n");
    printf("Time for multiplying two matrices (pure C): %.3g s, %.1f GFLOPs\n",
        elapsed_sec, flops / (elapsed_sec * 10e9));


    iarray_dtshape_t outdtshape;
    outdtshape.ndim = 2;
    outdtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    for (int i = 0; i < outdtshape.ndim; ++i) {
        outdtshape.shape[i] = oshape[i];
        outdtshape.pshape[i] = opshape[i];
    }

    iarray_container_t *con_out;
    iarray_container_new(ctx, &outdtshape, &mat_out_prop, 0, &con_out);

    INA_STOPWATCH_START(w);
    iarray_linalg_matmul(ctx, con_x, con_y, con_out, xbshape, ybshape, IARRAY_OPERATOR_GENERAL); /* FIXME: error handling */
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    iarray_container_info(con_out, &nbytes, &cbytes);
    printf("\n");
    printf("Time for multiplying two matrices (iarray):  %.3g s, %.1f GFLOPs\n",
        elapsed_sec, flops / (elapsed_sec * 10e9));

    nbytes_mb = ((double) nbytes / _IARRAY_SIZE_MB);
    cbytes_mb = ((double) cbytes / _IARRAY_SIZE_MB);
    printf("Compression for OUT values: %.1f MB -> %.1f MB (%.1fx)\n",
            nbytes_mb, cbytes_mb, (1.*nbytes) / cbytes);

    /* Check that we are getting the same results than through manual computation */
    ina_mem_set(mat_out, 0, NELEM_BYTES(osize));
    iarray_to_buffer(ctx, con_out, mat_out, NELEM_BYTES(osize));

    if (!test_mat_equal((int) osize, mat_res, mat_out)) {
        return EXIT_FAILURE; /* FIXME: error-handling */
    } else {
        printf("\nThe multiplication has been done correctly!\n");
    }

    iarray_container_free(ctx, &con_x);
    iarray_container_free(ctx, &con_y);
    iarray_container_free(ctx, &con_out);

    iarray_context_free(&ctx);

    ina_mem_free(mat_x);
    ina_mem_free(mat_y);
    ina_mem_free(mat_out);
    ina_mem_free(mat_res);

    INA_STOPWATCH_FREE(&w);

    return EXIT_SUCCESS;
}
