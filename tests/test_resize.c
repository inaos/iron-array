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

#include <src/iarray_private.h>
#include <libiarray/iarray.h>
#include <tests/iarray_test.h>

static ina_rc_t test_resize(iarray_container_t *c_x, int64_t *new_shape) {

    INA_TEST_ASSERT_SUCCEED(iarray_container_resize(c_x, new_shape));

    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_resize(iarray_context_t *ctx, iarray_data_type_t dtype, int64_t type_size, int8_t ndim,
                                      const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                                      int64_t *new_shape, bool contiguous, char *urlpath) {
    void *buffer_x;
    size_t buffer_x_len;

    buffer_x_len = 1;
    for (int i = 0; i < ndim; ++i) {
        buffer_x_len *= shape[i];
    }
    buffer_x = ina_mem_alloc(buffer_x_len * type_size);

    fill_buf(dtype, buffer_x, buffer_x_len);

    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int j = 0; j < xdtshape.ndim; ++j) {
        xdtshape.shape[j] = shape[j];
    }

    iarray_storage_t store;
    store.contiguous = contiguous;
    store.urlpath = urlpath;
    for (int j = 0; j < xdtshape.ndim; ++j) {
        store.chunkshape[j] = cshape[j];
        store.blockshape[j] = bshape[j];
    }
    int64_t buf_size = 1;

    blosc2_remove_urlpath(store.urlpath);

    for (int k = 0; k < ndim; ++k) {
        buf_size *= (new_shape[k] - shape[k]);

    }

    uint8_t *bufdes;
    int64_t buflen = buf_size * type_size;
    bufdes = ina_mem_alloc(buflen);

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buffer_x, buffer_x_len * type_size, &store, 0, &c_x));
    INA_TEST_ASSERT_SUCCEED(test_resize(c_x, new_shape));

    INA_TEST_ASSERT_SUCCEED(iarray_get_slice_buffer(ctx, c_x, shape, new_shape, bufdes, buflen));

    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            for (int64_t l = 0; l < buf_size; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], 0);
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            for (int64_t l = 0; l < buf_size; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], 0);
            }
            break;
        case IARRAY_DATA_TYPE_INT64:
            for (int64_t l = 0; l < buf_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], 0);
            }
            break;
        case IARRAY_DATA_TYPE_INT32:
            for (int64_t l = 0; l < buf_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int32_t *) bufdes)[l], 0);
            }
            break;
        case IARRAY_DATA_TYPE_INT16:
            for (int64_t l = 0; l < buf_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int16_t *) bufdes)[l], 0);
            }
            break;
        case IARRAY_DATA_TYPE_INT8:
            for (int64_t l = 0; l < buf_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int8_t *) bufdes)[l], 0);
            }
            break;
        case IARRAY_DATA_TYPE_UINT64:
            for (int64_t l = 0; l < buf_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], 0ULL);
            }
            break;
        case IARRAY_DATA_TYPE_UINT32:
            for (int64_t l = 0; l < buf_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) bufdes)[l], 0L);
            }
            break;
        case IARRAY_DATA_TYPE_UINT16:
            for (int64_t l = 0; l < buf_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) bufdes)[l], 0);
            }
            break;
        case IARRAY_DATA_TYPE_UINT8:
            for (int64_t l = 0; l < buf_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) bufdes)[l], 0);
            }
            break;
        case IARRAY_DATA_TYPE_BOOL:
            for (int64_t l = 0; l < buf_size; ++l) {
                INA_TEST_ASSERT(((bool *) bufdes)[l] == false);
            }
            break;
    }

    iarray_container_free(ctx, &c_x);

    ina_mem_free(buffer_x);
    ina_mem_free(bufdes);
    blosc2_remove_urlpath(store.urlpath);

    return INA_SUCCESS;
}

INA_TEST_DATA(resize) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(resize) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.eval_method = IARRAY_EVAL_METHOD_ITERCHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(resize) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(resize, 2_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 2};
    int64_t new_shape[] = {23, 22};


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_resize(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, new_shape,
                                                   false, NULL));
}

INA_TEST_FIXTURE(resize, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {3, 5, 2};
    int64_t bshape[] = {3, 5, 2};
    int64_t new_shape[] = {11, 15, 15};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_resize(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, new_shape,
                                                  false, "arr.iarr"));
}

INA_TEST_FIXTURE(resize, 5_i) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    int32_t type_size = sizeof(int32_t);

    const int8_t ndim = 5;
    int64_t shape[] = {10, 10, 10, 10, 10};
    int64_t cshape[] = {5, 5, 5, 5, 5};
    int64_t bshape[] = {2, 5, 1, 5, 2};
    int64_t new_shape[] = {12, 15, 12, 15, 11};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_resize(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, new_shape,
                                                  true, NULL));
}

INA_TEST_FIXTURE(resize, 5_s) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    int32_t type_size = sizeof(int16_t);

    const int8_t ndim = 5;
    int64_t shape[] = {10, 10, 10, 10, 10};
    int64_t cshape[] = {10, 10, 10, 10, 10};
    int64_t bshape[] = {5, 5, 5, 5, 5};
    int64_t new_shape[] = {15, 15, 15, 15, 15};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_resize(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, new_shape,
                                                  true, "arr.iarr"));
}

INA_TEST_FIXTURE(resize, 4_ull) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t type_size = sizeof(uint64_t);

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {7, 8, 8, 4};
    int64_t bshape[] = {3, 5, 2, 4};
    int64_t new_shape[] = {11, 11, 11, 11};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_resize(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, new_shape,
                                                  false, NULL));
}

INA_TEST_FIXTURE(resize, 2_b) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    int32_t type_size = sizeof(bool);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 5};
    int64_t new_shape[] = {13, 12};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_resize(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, new_shape,
                                                  false, "arr.iarr"));
}
