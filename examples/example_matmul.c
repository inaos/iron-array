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

int main(int argc, char **argv)
{
    iarray_init();
    ina_stopwatch_t *w = NULL;
    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    int n_threads = 2;
    int8_t ndim = 2;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int64_t shape_x[] = {2000, 1000};
    int64_t shape_y[] = {1000, 1500};
    int64_t shape_z[] = {2000, 1500};

    int64_t size_x = 2000 * 1000;
    int64_t size_y = 1000 * 1500;
    int64_t size_z = 2000 * 1500;


    int64_t pshape_x[] = {0, 0};
    int64_t pshape_y[] = {100, 200};
    int64_t pshape_z[] = {0, 0};

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.max_num_threads = n_threads;
    iarray_context_t *ctx;
    iarray_context_new(&cfg, &ctx);

    iarray_dtshape_t dtshape_x;
    dtshape_x.ndim = ndim;
    dtshape_x.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_x.shape[i] = shape_x[i];
        dtshape_x.pshape[i] = pshape_x[i];
    }
    iarray_container_t *c_x;
    iarray_linspace(ctx, &dtshape_x, size_x, 0, 1, NULL, 0, &c_x);

    iarray_dtshape_t dtshape_y;
    dtshape_y.ndim = ndim;
    dtshape_y.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_y.shape[i] = shape_y[i];
        dtshape_y.pshape[i] = pshape_y[i];
    }

    iarray_container_t *c_y;
    iarray_linspace(ctx, &dtshape_y, size_y, 0, 1, NULL, 0, &c_y);

    iarray_dtshape_t dtshape_z;
    dtshape_z.ndim = ndim;
    dtshape_z.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_z.shape[i] = shape_z[i];
        dtshape_z.pshape[i] = pshape_z[i];
    }

    iarray_container_t *c_z;
    iarray_container_new(ctx, &dtshape_z, NULL, 0, &c_z);
    mkl_set_num_threads(n_threads);


    double *b_x = (double *) malloc(size_x * sizeof(double));
    double *b_y = (double *) malloc(size_y * sizeof(double));
    double *b_z = (double *) malloc(size_z * sizeof(double));
    double *b_res = (double *) malloc(size_z * sizeof(double));

    iarray_to_buffer(ctx, c_x, b_x, size_x * sizeof(double));
    iarray_to_buffer(ctx, c_y, b_y, size_y * sizeof(double));


    INA_STOPWATCH_START(w);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) shape_x[0], (int) shape_y[1], (int) shape_x[1],
                1.0, b_x, (int) shape_x[1], b_y, (int) shape_y[1], 0.0, b_z, (int) shape_y[1]);
    INA_STOPWATCH_STOP(w);

    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time mkl (C): %.4f\n", elapsed_sec);

    // int64_t bshape_x[] = {2000, 1000};
    // int64_t bshape_y[] = {1000, 1500};
    // If using the block shapes below, the iarray_linalg_matmul() does not work well
    int64_t bshape_x[2];
    int64_t bshape_y[2];
    if (INA_FAILED(iarray_matmul_advice(ctx, c_x, c_y, c_z, bshape_x, bshape_y, 0, 0))) {
        printf("Error in getting advice for matmul: %s\n", ina_err_strerror(ina_err_get_rc()));
        exit(1);
    }
    printf("bshape_x: (%lld, %lld)\n", bshape_x[0], bshape_x[1]);
    printf("bshape_y: (%lld, %lld)\n", bshape_y[0], bshape_y[1]);

    INA_STOPWATCH_START(w);
    if (INA_FAILED(iarray_linalg_matmul(ctx, c_x, c_y ,c_z, bshape_x, bshape_y, IARRAY_OPERATOR_GENERAL))) {
        printf("Error in linalg_matmul: %s\n", ina_err_strerror(ina_err_get_rc()));
        exit(1);
    }
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time iarray: %.4f\n", elapsed_sec);

    iarray_to_buffer(ctx, c_z, b_res, size_z * sizeof(double));

    for (int i = 0; i < size_z; ++i) {
        if (fabs((b_res[i] - b_z[i]) / b_res[i]) > 1e-8) {
            printf("%f - %f = %f\n", b_res[i], b_z[i], b_res[i] - b_z[i]);
            printf("Error in element %d\n", i);
            return INA_ERROR(INA_ERR_ERROR);
        }
    }

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_z);
    free(b_x);
    free(b_y);
    free(b_z);
    free(b_res);
    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);
    iarray_destroy();
    return EXIT_SUCCESS;
}
