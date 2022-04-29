/*
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include "iarray_test.h"
#include <libiarray/iarray.h>
#include <src/iarray_private.h>

static ina_rc_t test_delete(iarray_context_t *ctx, iarray_container_t *c_x, int64_t axis, int64_t start, int64_t nitems) {

    INA_TEST_ASSERT_SUCCEED(iarray_container_delete(ctx, c_x, axis, start, nitems));

    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_delete(iarray_context_t *ctx, iarray_data_type_t dtype, int64_t type_size, int8_t ndim,
                                      const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                                      int64_t axis, int64_t start, int64_t nitems, bool contiguous, char *urlpath) {
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
    blosc2_remove_urlpath(store.urlpath);
    iarray_container_t *c_x;
    void *value = malloc(type_size);
    int8_t fill_value = 1;
    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            ((double *) value)[0] = (double) fill_value;
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &xdtshape, value, &store, &c_x));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            ((float *)  value)[0] = (float) fill_value;
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &xdtshape, value, &store, &c_x));
            break;
        case IARRAY_DATA_TYPE_INT64:
            ((int64_t *) value)[0] = (int64_t) fill_value;
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &xdtshape, value, &store, &c_x));
            break;
        case IARRAY_DATA_TYPE_INT32:
            ((int32_t *) value)[0] = (int32_t) fill_value;
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &xdtshape, value, &store, &c_x));
            break;
        case IARRAY_DATA_TYPE_INT16:
            ((int16_t *) value)[0] = (int16_t) fill_value;
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &xdtshape, value, &store, &c_x));
            break;
        case IARRAY_DATA_TYPE_INT8:
            ((int8_t *) value)[0] = (int8_t) fill_value;
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &xdtshape, value, &store, &c_x));
            break;
        case IARRAY_DATA_TYPE_UINT64:
            ((uint64_t *) value)[0] = (uint64_t) fill_value;
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &xdtshape, value, &store, &c_x));
            break;
        case IARRAY_DATA_TYPE_UINT32:
            ((uint32_t *) value)[0] = (uint32_t) fill_value;
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &xdtshape, value, &store, &c_x));
            break;
        case IARRAY_DATA_TYPE_UINT16:
            ((uint16_t *) value)[0] = (uint16_t) fill_value;
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &xdtshape, value, &store, &c_x));
            break;
        case IARRAY_DATA_TYPE_UINT8:
            ((uint8_t *) value)[0] = (uint8_t) fill_value;
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &xdtshape, value, &store, &c_x));
            break;
        case IARRAY_DATA_TYPE_BOOL:
            ((bool *) value)[0] = (bool) fill_value;
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &xdtshape, value, &store, &c_x));
            break;
        default:
            return INA_ERR_EXCEEDED;
    }

    INA_TEST_ASSERT_SUCCEED(test_delete(ctx, c_x, axis, start, nitems));


    /* Fill buffer with whole array */
    void *buffer = ina_mem_alloc(c_x->catarr->nitems * type_size);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buffer, c_x->catarr->nitems * type_size));

    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            for (int64_t l = 0; l < c_x->catarr->nitems; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) buffer)[l], (double) fill_value);
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            for (int64_t l = 0; l < c_x->catarr->nitems; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) buffer)[l], (float) fill_value);
            }
            break;
        case IARRAY_DATA_TYPE_INT64:
            for (int64_t l = 0; l < c_x->catarr->nitems; ++l) {
                INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) buffer)[l], (int64_t) fill_value);
            }
            break;
        case IARRAY_DATA_TYPE_INT32:
            for (int64_t l = 0; l < c_x->catarr->nitems; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int32_t *) buffer)[l], (int32_t) fill_value);
            }
            break;
        case IARRAY_DATA_TYPE_INT16:
            for (int64_t l = 0; l < c_x->catarr->nitems; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int16_t *) buffer)[l], (int16_t) fill_value);
            }
            break;
        case IARRAY_DATA_TYPE_INT8:
            for (int64_t l = 0; l < c_x->catarr->nitems; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int8_t *) buffer)[l], (int8_t) fill_value);
            }
            break;
        case IARRAY_DATA_TYPE_UINT64:
            for (int64_t l = 0; l < c_x->catarr->nitems; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) buffer)[l], (uint64_t) fill_value);
            }
            break;
        case IARRAY_DATA_TYPE_UINT32:
            for (int64_t l = 0; l < c_x->catarr->nitems; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) buffer)[l], (uint32_t) fill_value);
            }
            break;
        case IARRAY_DATA_TYPE_UINT16:
            for (int64_t l = 0; l < c_x->catarr->nitems; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) buffer)[l], (uint16_t) fill_value);
            }
            break;
        case IARRAY_DATA_TYPE_UINT8:
            for (int64_t l = 0; l < c_x->catarr->nitems; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) buffer)[l], (uint8_t) fill_value);
            }
            break;
        case IARRAY_DATA_TYPE_BOOL:
            for (int64_t l = 0; l < c_x->catarr->nitems; ++l) {
                INA_TEST_ASSERT(((bool *) buffer)[l] == (bool) fill_value);
            }
            break;
        default:
            IARRAY_TRACE1(iarray.error, "Invalid dtype");
            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }

    iarray_container_free(ctx, &c_x);

    ina_mem_free(buffer);
    free(value);
    blosc2_remove_urlpath(store.urlpath);

    return INA_SUCCESS;
}

INA_TEST_DATA(delete) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(delete) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.eval_method = IARRAY_EVAL_METHOD_ITERCHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(delete) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(delete, 2_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 2};
    int64_t axis = 1;
    int64_t start = 0;
    int64_t nitems = 5;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_delete(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, axis,
                                                   start, nitems, false, NULL));
}

INA_TEST_FIXTURE(delete, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {3, 5, 2};
    int64_t bshape[] = {3, 5, 2};
    int64_t axis = 2;
    int64_t start = 4;
    int64_t nitems = 4;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_delete(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, axis,
                                                   start, nitems, false, "arr.iarr"));
}

INA_TEST_FIXTURE(delete, 5_i) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    int32_t type_size = sizeof(int32_t);

    const int8_t ndim = 5;
    int64_t shape[] = {10, 10, 10, 10, 10};
    int64_t cshape[] = {5, 5, 5, 5, 5};
    int64_t bshape[] = {2, 5, 1, 5, 2};
    int64_t axis = 3;
    int64_t start = 0;
    int64_t nitems = 5;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_delete(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, axis,
                                                   start, nitems, true, NULL));
}

INA_TEST_FIXTURE(delete, 5_s) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    int32_t type_size = sizeof(int16_t);

    const int8_t ndim = 5;
    int64_t shape[] = {10, 10, 10, 10, 10};
    int64_t cshape[] = {10, 10, 10, 10, 10};
    int64_t bshape[] = {5, 5, 5, 5, 5};
    int64_t axis = 4;
    int64_t start = 2;
    int64_t nitems = 8;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_delete(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, axis,
                                                   start, nitems, true, "arr.iarr"));
}

INA_TEST_FIXTURE(delete, 4_ull) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t type_size = sizeof(uint64_t);

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {7, 8, 8, 4};
    int64_t bshape[] = {3, 5, 2, 4};
    int64_t axis = 2;
    int64_t start = 6;
    int64_t nitems = 4;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_delete(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, axis,
                                                   start, nitems, false, NULL));
}

INA_TEST_FIXTURE(delete, 2_b) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    int32_t type_size = sizeof(bool);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 5};
    int64_t axis = 0;
    int64_t start = 7;
    int64_t nitems = 3;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_delete(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, axis,
                                                   start, nitems, false, "arr.iarr"));
}

INA_TEST_FIXTURE(delete, 2_ui) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    int32_t type_size = sizeof(uint32_t);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 5};
    int64_t axis = 1;
    int64_t start = 5;
    int64_t nitems = 5;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_delete(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, axis,
                                                   start, nitems, false, "arr.iarr"));
}
