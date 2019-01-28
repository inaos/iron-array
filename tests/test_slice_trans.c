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

#include <tests/iarray_test.h>

static ina_rc_t test_slice(iarray_context_t *ctx, iarray_container_t *c_x, int64_t * start, int64_t *stop,
    iarray_dtshape_t dtshape, iarray_store_properties_t *stores, int flags, iarray_container_t **c_out) {
    INA_TEST_ASSERT_SUCCEED(iarray_slice(ctx, c_x, start, stop, &dtshape, stores, flags, c_out));
    INA_TEST_ASSERT_SUCCEED(iarray_squeeze(ctx, *c_out));

    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_slice(iarray_context_t *ctx, iarray_data_type_t dtype, size_t type_size, uint8_t ndim,
                                      const uint64_t *shape, const uint64_t *pshape, const uint64_t *pshape_dest,
                                      int64_t *start, int64_t *stop, const void *result) {
    void *buffer_x;
    size_t buffer_x_len;

    buffer_x_len = 1;
    for (int i = 0; i < ndim; ++i) {
        buffer_x_len *= shape[i];
    }
    buffer_x = ina_mem_alloc(buffer_x_len * type_size);

    if (type_size == sizeof(float)) {
        ffill_buf((float *) buffer_x, buffer_x_len);

    } else {
        dfill_buf((double *) buffer_x, buffer_x_len);
    }

    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int j = 0; j < xdtshape.ndim; ++j) {
        xdtshape.shape[j] = shape[j];
        xdtshape.pshape[j] = pshape[j];
    }

    iarray_dtshape_t outdtshape;

    outdtshape.dtype = dtype;
    outdtshape.ndim = ndim;
    for (int j = 0; j < xdtshape.ndim; ++j) {
        int64_t st = (start[j] + shape[j]) % shape[j];
        int64_t sp = (stop[j] + shape[j] - 1) % shape[j] + 1;
        outdtshape.shape[j] = (uint64_t) sp - st;
        outdtshape.pshape[j] = pshape_dest[j];
    }

    iarray_container_t *c_x;
    iarray_container_t *c_out;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buffer_x, buffer_x_len * type_size, NULL, 0, &c_x));

    iarray_linalg_transpose(ctx, c_x);

    INA_TEST_ASSERT_SUCCEED(test_slice(ctx, c_x, start, stop, outdtshape, NULL, 0, &c_out));

    uint64_t bufdes_size = 1;

    for (int k = 0; k < ndim; ++k) {
        int64_t st = (start[k] + shape[k]) % shape[k];
        int64_t sp = (stop[k] + shape[k] - 1) % shape[k] + 1;
        bufdes_size *= (uint64_t) sp - st;;
    }

    uint8_t *bufdes;

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        bufdes = ina_mem_alloc(bufdes_size * sizeof(double));
        iarray_to_buffer(ctx, c_out, bufdes, bufdes_size * sizeof(double));
        for (uint64_t l = 0; l < bufdes_size; ++l) {
            printf("%f - %f\n", ((double *) bufdes)[l], ((double *) result)[l]);
            //INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], ((double *) result)[l]);
        }
    } else {
        bufdes = ina_mem_alloc(bufdes_size * sizeof(float));
        iarray_to_buffer(ctx, c_out, bufdes, bufdes_size * sizeof(float));
        for (uint64_t l = 0; l < bufdes_size; ++l) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], ((float *) result)[l]);
        }
    }

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_out);

    ina_mem_free(buffer_x);

    return INA_SUCCESS;
}

INA_TEST_DATA(slice_trans) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(slice_trans) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(slice_trans) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(slice_trans, double_data_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    const uint64_t ndim = 2;
    uint64_t shape[] = {10, 10};
    uint64_t pshape[] = {3, 4};
    int64_t start[] = {2, 1};
    int64_t stop[] = {7, 3};
    uint64_t pshape_dest[] = {2, 2};

    double result[] = {12, 22, 13, 23, 14, 24, 15, 25, 16, 26};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, pshape, pshape_dest,
        start, stop, result));
}