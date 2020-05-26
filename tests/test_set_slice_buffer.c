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
#include <src/iarray_private.h>


static ina_rc_t test_set_slice_buffer(iarray_context_t *ctx,
                                      iarray_container_t *c_x,
                                      int64_t *start,
                                      int64_t *stop,
                                      void *buffer,
                                      int64_t buflen)
{

    INA_TEST_ASSERT_SUCCEED(iarray_set_slice_buffer(ctx, c_x, start, stop, buffer, buflen));
    INA_TEST_ASSERT_SUCCEED(iarray_get_slice_buffer(ctx, c_x, start, stop, buffer, buflen));

    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_set_slice(iarray_context_t *ctx,
                                          iarray_data_type_t dtype,
                                          int64_t type_size,
                                          int8_t ndim,
                                          const int64_t *shape,
                                          const int64_t *pshape,
                                          int64_t *start,
                                          int64_t *stop,
                                          int transposed) {
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
        if (pshape)
            xdtshape.pshape[j] = pshape[j];
    }

    iarray_storage_t xstore;
    xstore.backend = pshape? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    xstore.enforce_frame = false;
    xstore.filename = NULL;

    int64_t bufdes_size = 1;

    for (int k = 0; k < ndim; ++k) {
        int64_t st = (start[k] + shape[k]) % shape[k];
        int64_t sp = (stop[k] + shape[k] - 1) % shape[k] + 1;
        bufdes_size *= (int64_t) sp - st;
    }

    int64_t buflen = bufdes_size * type_size;

    uint8_t *bufdes = ina_mem_alloc(bufdes_size * type_size);

    for (int i = 0; i < bufdes_size; ++i) {
        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            ((double *) bufdes)[i] = (double) i;
        } else {
            ((float *) bufdes)[i] = (float) i;
        }
    }

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buffer_x, buffer_x_len * type_size, &xstore, 0, &c_x));

    if (transposed == 1) {
        iarray_linalg_transpose(ctx, c_x);
    }

    INA_TEST_ASSERT_SUCCEED(test_set_slice_buffer(ctx, c_x, start, stop, bufdes, buflen));

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        for (int64_t l = 0; l < bufdes_size; ++l) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], (double) l);
        }
    } else {
        for (int64_t l = 0; l < bufdes_size; ++l) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], (float) l);
        }
    }

    iarray_container_free(ctx, &c_x);

    ina_mem_free(buffer_x);
    ina_mem_free(bufdes);

    return INA_SUCCESS;
}

INA_TEST_DATA(set_slice_buffer) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(set_slice_buffer) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(set_slice_buffer) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(set_slice_buffer, 2_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t *pshape = NULL;
    int64_t start[] = {21, 17};
    int64_t stop[] = {-21, 55};
    bool transposed = true;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                      start, stop, transposed));
}

INA_TEST_FIXTURE(set_slice_buffer, 2_d_t) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t *pshape = NULL;
    int64_t start[] = {0, 0};
    int64_t stop[] = {-5, 5};
    bool transposed = true;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                      start, stop, transposed));
}

INA_TEST_FIXTURE(set_slice_buffer, 2_f_t) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 2;
    int64_t shape[] = {20, 14};
    int64_t *pshape = NULL;
    int64_t start[] = {3, 1};
    int64_t stop[] = {-2, 5};
    bool transposed = true;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                      start, stop, transposed));
}

INA_TEST_FIXTURE(set_slice_buffer, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 3;
    int64_t shape[] = {100, 123, 234};
    int64_t *pshape = NULL;
    int64_t start[] = {23, 31, 22};
    int64_t stop[] = {54, 78, 76};
    bool transposed = false;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                      start, stop, transposed));
}

INA_TEST_FIXTURE(set_slice_buffer, 4_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 4;
    int64_t shape[] = {60, 80, 80, 15};
    int64_t *pshape = NULL;
    int64_t start[] = {23, 31, 22, 1};
    int64_t stop[] = {54, 78, 76, 2};
    bool transposed = false;


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                      start, stop, transposed));
}

INA_TEST_FIXTURE(set_slice_buffer, 5_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 5;
    int64_t shape[] = {10, 12, 32, 14, 14};
    int64_t *pshape = NULL;
    int64_t start[] = {1, 2, 4, 5, 6};
    int64_t stop[] = {8, 9, 11, 12, 13};
    bool transposed = false;


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                      start, stop, transposed));
}

INA_TEST_FIXTURE(set_slice_buffer, 6_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 6;
    int64_t shape[] = {8, 7, 6, 7, 8, 5};
    int64_t *pshape = NULL;
    int64_t start[] = {1, 2, 3, 4, 5, 2};
    int64_t stop[] = {3, 4, 4, 7, 7, 4};
    bool transposed = false;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                      start, stop, transposed));
}

INA_TEST_FIXTURE(set_slice_buffer, 7_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 7;
    int64_t shape[] = {5, 7, 6, 4, 8, 6, 5};
    int64_t *pshape = NULL;
    int64_t start[] = {1, 2, 1, 2, 0, 2, 1};
    int64_t stop[] = {5, 4, 4, 3, 6, 3, 4};
    bool transposed = false;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                      start, stop, transposed));
}
