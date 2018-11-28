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

#include <tests/iarray_test.h>

static ina_rc_t test_slice(iarray_context_t *ctx, iarray_container_t **c_out, iarray_container_t *c_x, size_t* start, size_t *stop) {
    INA_TEST_ASSERT_SUCCEED(iarray_slice(ctx, c_x, start, stop, NULL, 0, c_out));

    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_gemm(iarray_context_t *ctx, iarray_data_type_t dtype, size_t type_size, int ndim,
                                     int *shape, int *pshape, size_t *start, size_t *stop) {
    void *buffer_x;
    void *buffer_y;
    void *buffer_r;
    size_t buffer_x_len;
    size_t buffer_y_len;
    size_t buffer_r_len;
    double tol;

    buffer_x_len = 1;
    for (int i = 0; i < ndim; ++i) {
        buffer_x_len *= shape[i];
    }
    buffer_x = ina_mem_alloc(buffer_x_len * type_size);

    if (type_size == sizeof(float)) {
        tol = 1e-06;
        ffill_buf((float *) buffer_x, buffer_x_len);

    } else {
        tol = 1e-14;
        dfill_buf((double *) buffer_x, buffer_x_len);
    }

    iarray_dtshape_t xshape;

    xshape.dtype = dtype;
    xshape.ndim = ndim;
    for (int j = 0; j < xshape.ndim; ++j) {
        xshape.shape[j] = shape[j];
        xshape.partshape[j] = pshape[j];
    }


    iarray_container_t *c_x;
    iarray_container_t *c_out;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xshape, buffer_x, buffer_x_len * type_size, NULL, 0, &c_x));

    INA_TEST_ASSERT_SUCCEED(test_slice(ctx, &c_out, c_x, start, stop));

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_out);

    ina_mem_free(buffer_x);

    return INA_SUCCESS;
}

INA_TEST_DATA(slice) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(slice) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(slice) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(slice, double_data) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int ndim = 3;
    int shape[] = {123, 156, 234};
    int pshape[] = {13, 12, 17};
    size_t start[] = {45, 2, 103};
    size_t stop[] = {102, 66, 199};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_gemm(data->ctx, dtype, type_size, ndim, shape, pshape, start, stop));
}