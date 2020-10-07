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

static ina_rc_t test_slice(iarray_context_t *ctx, iarray_container_t *c_x, int64_t *start,
                           int64_t *stop, iarray_storage_t *stores,
                           int flags, iarray_container_t **c_out) {
    INA_TEST_ASSERT_SUCCEED(iarray_get_slice(ctx, c_x, start, stop, true, stores, flags, c_out));
    INA_TEST_ASSERT_SUCCEED(iarray_squeeze(ctx, *c_out));

    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_slice(iarray_context_t *ctx, iarray_data_type_t dtype, int32_t type_size, int8_t ndim,
                                      const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                                      int64_t *start, int64_t *stop) {
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

    }

    iarray_storage_t xstore;
    xstore.backend = cshape? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    xstore.enforce_frame = false;
    xstore.filename = NULL;
    if (cshape != NULL) {
        for (int i = 0; i < ndim; ++i) {
            xstore.chunkshape[i] = cshape[i];
            xstore.blockshape[i] = bshape[i];
        }
    }
    iarray_container_t *c_x;
    iarray_container_t *c_out;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buffer_x, buffer_x_len * type_size, &xstore, 0, &c_x));

    INA_TEST_ASSERT_SUCCEED(test_slice(ctx, c_x, start, stop, NULL, 0, &c_out));

    int64_t blockshape[IARRAY_DIMENSION_MAX] = {2, 2, 2, 2, 2, 2, 2, 2};
    iarray_iter_read_block_t *iter;
    iarray_iter_read_block_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &iter, c_out, blockshape, &val, false));

    while (INA_SUCCEED(iarray_iter_read_block_has_next(iter))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(iter, NULL, 0));
        uint8_t *block_buffer = malloc(val.block_size * type_size);
        int64_t block_stop[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < c_out->dtshape->ndim; ++i) {
            block_stop[i] = val.elem_index[i] + val.block_shape[i];
        }

        iarray_get_slice_buffer(ctx, c_out, val.elem_index, block_stop, block_buffer, val.block_size * type_size);

        for (int j = 0; j < val.block_size; ++j) {
            if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) val.block_pointer)[j], ((double *) block_buffer)[j]);
            } else {
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) val.block_pointer)[j], ((float *) block_buffer)[j]);
            }
        }
        free(block_buffer);
    }
    iarray_iter_read_block_free(&iter);
    INA_TEST_ASSERT_SUCCEED(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_out);

    ina_mem_free(buffer_x);

    return INA_SUCCESS;
}

INA_TEST_DATA(view_block_iter) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(view_block_iter) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(view_block_iter) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(view_block_iter, 2_d_p_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t *cshape = NULL;
    int64_t *bshape = NULL;
    int64_t start[] = {-5, -7};
    int64_t stop[] = {-1, 10};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop));
}

INA_TEST_FIXTURE(view_block_iter, 3_f_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {7, 5, 7};
    int64_t bshape[] = {3, 5, 2};
    int64_t start[] = {3, 0, 3};
    int64_t stop[] = {-4, -3, 10};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop));
}


INA_TEST_FIXTURE(view_block_iter, 4_d_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {3, 5, 2, 7};
    int64_t bshape[] = {2, 2, 2, 2};
    int64_t start[] = {5, -7, 9, 2};
    int64_t stop[] = {-1, 6, 10, -3};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop));
}

INA_TEST_FIXTURE(view_block_iter, 5_f_p_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 5;
    int64_t shape[] = {10, 10, 10, 10, 10};
    int64_t *cshape = NULL;
    int64_t *bshape = NULL;
    int64_t start[] = {-4, 0, -5, 5, 7};
    int64_t stop[] = {8, 9, -4, -4, 10};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop));
}

INA_TEST_FIXTURE(view_block_iter, 6_d_p_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 6;
    int64_t shape[] = {10, 10, 10, 10, 10, 10};
    int64_t *cshape = NULL;
    int64_t *bshape = NULL;
    int64_t start[] = {0, 4, -8, 4, 5, 1};
    int64_t stop[] = {1, 7, 4, -4, 8, 3};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop));
}

INA_TEST_FIXTURE(view_block_iter, 7_f_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 7;
    int64_t shape[] = {10, 10, 10, 10, 10, 10, 10};
    int64_t cshape[] = {4, 5, 1, 8, 5, 3, 10};
    int64_t bshape[] = {2, 3, 1, 2, 2, 1, 4};
    int64_t start[] = {5, 4, 3, -2, 4, 5, -9};
    int64_t stop[] = {8, 6, 5, 9, 7, 7, -7};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_slice(data->ctx, dtype, type_size, ndim, shape, cshape, bshape,
                                                  start, stop));
}
