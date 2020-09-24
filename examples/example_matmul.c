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
#include <mkl.h>


int main(void)
{
    iarray_init();
    ina_stopwatch_t *w = NULL;
    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    int n_threads = 1;
    int8_t ndim = 2;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int64_t shape_x[] = {2000, 1000};
    int64_t shape_y[] = {1000, 1500};
    int64_t shape_z[] = {2000, 1500};

    int64_t size_x = 2000 * 1000;
    int64_t size_y = 1000 * 1500;
    int64_t size_z = 2000 * 1500;

    int64_t cshape_x[] = {200, 200};
    int64_t cshape_y[] = {200, 200};
    int64_t cshape_z[] = {200, 200};

    int64_t blockshape_x[2] = {200, 200};
    int64_t blockshape_y[2] = {200, 200};

    int64_t bshape_x[] = {20, 15};
    int64_t bshape_y[] = {31, 17};
    int64_t bshape_z[] = {45, 77};

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.max_num_threads = n_threads;
    iarray_context_t *ctx;
    IARRAY_FAIL_IF_ERROR(iarray_context_new(&cfg, &ctx));

    iarray_dtshape_t dtshape_x;
    dtshape_x.ndim = ndim;
    dtshape_x.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_x.shape[i] = shape_x[i];
    }

    iarray_storage_t store_x;
    store_x.backend = IARRAY_STORAGE_BLOSC;
    store_x.enforce_frame = false;
    store_x.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        store_x.chunkshape[i] = cshape_x[i];
        store_x.blockshape[i] = bshape_x[i];
    }
    iarray_container_t *c_x;
    IARRAY_FAIL_IF_ERROR(iarray_linspace(ctx, &dtshape_x, size_x, 0, 1, &store_x, 0, &c_x));

    iarray_dtshape_t dtshape_y;
    dtshape_y.ndim = ndim;
    dtshape_y.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_y.shape[i] = shape_y[i];
    }
    iarray_storage_t store_y;
    store_y.backend = IARRAY_STORAGE_BLOSC;
    store_y.enforce_frame = false;
    store_y.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        store_y.chunkshape[i] = cshape_y[i];
        store_y.blockshape[i] = bshape_y[i];
    }
    iarray_container_t *c_y;
    IARRAY_FAIL_IF_ERROR(iarray_linspace(ctx, &dtshape_y, size_y, 0, 1, &store_y, 0, &c_y));

    iarray_dtshape_t dtshape_z;
    dtshape_z.ndim = ndim;
    dtshape_z.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_z.shape[i] = shape_z[i];
    }
    iarray_storage_t store_z;
    store_z.backend = IARRAY_STORAGE_BLOSC;
    store_z.enforce_frame = false;
    store_z.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        store_z.chunkshape[i] = cshape_z[i];
        store_z.blockshape[i] = bshape_z[i];
    }
    iarray_container_t *c_z;
    IARRAY_FAIL_IF_ERROR(iarray_container_new(ctx, &dtshape_z, &store_z, 0, &c_z));
    mkl_set_num_threads(n_threads);


    double *b_x = (double *) malloc(size_x * sizeof(double));
    double *b_y = (double *) malloc(size_y * sizeof(double));
    double *b_z = (double *) malloc(size_z * sizeof(double));
    double *b_res = (double *) malloc(size_z * sizeof(double));

    IARRAY_FAIL_IF_ERROR(iarray_to_buffer(ctx, c_x, b_x, size_x * sizeof(double)));
    IARRAY_FAIL_IF_ERROR(iarray_to_buffer(ctx, c_y, b_y, size_y * sizeof(double)));

    INA_STOPWATCH_START(w);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) shape_x[0], (int) shape_y[1], (int) shape_x[1],
                1.0, b_x, (int) shape_x[1], b_y, (int) shape_y[1], 0.0, b_z, (int) shape_y[1]);
    INA_STOPWATCH_STOP(w);
    IARRAY_FAIL_IF_ERROR(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time mkl (C): %.4f\n", elapsed_sec);

    INA_STOPWATCH_START(w);
    if (INA_FAILED(iarray_linalg_matmul(ctx, c_x, c_y ,c_z, blockshape_x, blockshape_y, IARRAY_OPERATOR_GENERAL))) {
        fprintf(stderr, "Error in linalg_matmul: %s\n", ina_err_strerror(ina_err_get_rc()));
        goto fail;
    }
    INA_STOPWATCH_STOP(w);
    IARRAY_FAIL_IF_ERROR(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time iarray: %.4f\n", elapsed_sec);

    IARRAY_FAIL_IF_ERROR(iarray_to_buffer(ctx, c_z, b_res, size_z * sizeof(double)));

    for (int64_t i = 0; i < size_z; ++i) {
        if (fabs((b_res[i] - b_z[i]) / b_res[i]) > 1e-8) {
            fprintf(stderr, "%f - %f = %f\n", b_res[i], b_z[i], b_res[i] - b_z[i]);
            fprintf(stderr, "Error in element %" PRId64 "\n", i);
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

    return INA_SUCCESS;
    fail:
    return ina_err_get_rc();
}
