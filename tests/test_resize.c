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

#include <src/iarray_private.h>
#include <libiarray/iarray.h>

static ina_rc_t test_resize(iarray_context_t *ctx, iarray_container_t *c_x, int64_t *new_shape, int64_t *start) {

    INA_TEST_ASSERT_SUCCEED(iarray_container_resize(ctx, c_x, new_shape, start));

    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_resize(iarray_context_t *ctx, iarray_data_type_t dtype, int64_t type_size, int8_t ndim,
                                      const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                                      int64_t *new_shape, int64_t *start, bool contiguous, char *urlpath) {
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

    int64_t buffersize = type_size;
    bool only_shrink = true;
    for (int i = 0; i < ndim; ++i) {
        if (new_shape[i] > shape[i]) {
            only_shrink = false;
        }
        buffersize *= new_shape[i];
    }
    INA_TEST_ASSERT_SUCCEED(test_resize(ctx, c_x, new_shape, start));

    // Create aux array to compare values
    iarray_dtshape_t ydtshape;
    ydtshape.dtype = dtype;
    ydtshape.ndim = ndim;
    for (int j = 0; j < ydtshape.ndim; ++j) {
        ydtshape.shape[j] = new_shape[j];
    }
    iarray_storage_t ystore;
    ystore.contiguous = contiguous;
    ystore.urlpath = NULL;
    for (int j = 0; j < ydtshape.ndim; ++j) {
        ystore.chunkshape[j] = cshape[j];
        ystore.blockshape[j] = bshape[j];
    }
    iarray_container_t *c_y;
    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &ydtshape, value, &ystore, &c_y));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &ydtshape, value, &ystore, &c_y));
            break;
        case IARRAY_DATA_TYPE_INT64:
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &ydtshape, value, &ystore, &c_y));
            break;
        case IARRAY_DATA_TYPE_INT32:
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &ydtshape, value, &ystore, &c_y));
            break;
        case IARRAY_DATA_TYPE_INT16:
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &ydtshape, value, &ystore, &c_y));
            break;
        case IARRAY_DATA_TYPE_INT8:
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &ydtshape, value, &ystore, &c_y));
            break;
        case IARRAY_DATA_TYPE_UINT64:
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &ydtshape, value, &ystore, &c_y));
            break;
        case IARRAY_DATA_TYPE_UINT32:
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &ydtshape, value, &ystore, &c_y));
            break;
        case IARRAY_DATA_TYPE_UINT16:
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &ydtshape, value, &ystore, &c_y));
            break;
        case IARRAY_DATA_TYPE_UINT8:
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &ydtshape, value, &ystore, &c_y));
            break;
        case IARRAY_DATA_TYPE_BOOL:
            INA_TEST_ASSERT_SUCCEED(iarray_fill(ctx, &ydtshape, value, &ystore, &c_y));
            break;
        default:
            return INA_ERR_EXCEEDED;
    }

    if (!only_shrink) {
        for (int i = 0; i < ndim; ++i) {
            if (new_shape[i] <= shape[i]) {
                continue;
            }
            int64_t slice_start[CATERVA_MAX_DIM] = {0};
            int64_t slice_stop[CATERVA_MAX_DIM];
            int64_t slice_shape[CATERVA_MAX_DIM] = {0};
            int64_t buffer_len = 1;
            for (int j = 0; j < ndim; ++j) {
                if (j != i) {
                    slice_shape[j] = new_shape[j];
                    buffer_len *= slice_shape[j];
                    slice_stop[j] = new_shape[j];
                }
            }
            slice_start[i] = start[i];
            slice_shape[i] = new_shape[i] - shape[i];
            if (slice_start[i] % cshape[i] != 0) {
                // Old padding was filled with ones
                slice_shape[i] -= cshape[i] - slice_start[i] % cshape[i];
                slice_start[i] += cshape[i] - slice_start[i] % cshape[i];
            }
            if (slice_start[i] > new_shape[i]) {
                continue;
            }
            slice_stop[i] = slice_start[i] + slice_shape[i];
            buffer_len *= slice_shape[i];
            uint8_t *buffer = calloc((size_t) buffer_len, (size_t) type_size);
            INA_TEST_ASSERT_SUCCEED(iarray_set_slice_buffer(ctx, c_y, slice_start, slice_stop, buffer,
                                                            buffer_len * type_size));
            free(buffer);
        }
    }

    /* Fill buffers with whole arrays */
    uint8_t *xbuffer = ina_mem_alloc((size_t) buffersize);
    uint8_t *ybuffer = ina_mem_alloc((size_t) buffersize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, xbuffer, buffersize));
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_y, ybuffer, buffersize));
    int64_t buf_len = buffersize / type_size;
    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            for (int64_t l = 0; l < buf_len; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) xbuffer)[l], ((double *) ybuffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            for (int64_t l = 0; l < buf_len; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) xbuffer)[l], ((float *) ybuffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT64:
            for (int64_t l = 0; l < buf_len; ++l) {
                INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) xbuffer)[l], ((int64_t *) ybuffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT32:
            for (int64_t l = 0; l < buf_len; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int32_t *) xbuffer)[l], ((int32_t *) ybuffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT16:
            for (int64_t l = 0; l < buf_len; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int16_t *) xbuffer)[l], ((int16_t *) ybuffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT8:
            for (int64_t l = 0; l < buf_len; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int8_t *) xbuffer)[l], ((int8_t *) ybuffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT64:
            for (int64_t l = 0; l < buf_len; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) xbuffer)[l], ((uint64_t *) ybuffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT32:
            for (int64_t l = 0; l < buf_len; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) xbuffer)[l], ((uint32_t *) ybuffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT16:
            for (int64_t l = 0; l < buf_len; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) xbuffer)[l], ((uint16_t *) ybuffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT8:
            for (int64_t l = 0; l < buf_len; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) xbuffer)[l], ((uint8_t *) ybuffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_BOOL:
            for (int64_t l = 0; l < buf_len; ++l) {
                INA_TEST_ASSERT(((bool *) xbuffer)[l] == ((bool *) ybuffer)[l]);
            }
            break;
        default:
            IARRAY_TRACE1(iarray.error, "Invalid dtype");
            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);

    ina_mem_free(ybuffer);
    ina_mem_free(xbuffer);
    free(value);
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
    int64_t new_shape[] = {20, 25};
    int64_t start[] = {10, 5};


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_resize(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, new_shape,
                                                   start, false, NULL));
}

INA_TEST_FIXTURE(resize, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 3;
    int64_t shape[] = {10, 10, 10};
    int64_t cshape[] = {3, 5, 2};
    int64_t bshape[] = {3, 5, 2};
    int64_t new_shape[] = {11, 5, 15};
    int64_t start[] = {10, 5, 10};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_resize(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, new_shape,
                                                   start, false, "arr.iarr"));
}

INA_TEST_FIXTURE(resize, 5_i) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    int32_t type_size = sizeof(int32_t);

    const int8_t ndim = 5;
    int64_t shape[] = {10, 10, 10, 10, 10};
    int64_t cshape[] = {5, 5, 5, 5, 5};
    int64_t bshape[] = {2, 5, 1, 5, 2};
    int64_t new_shape[] = {5, 5, 15, 15, 20};
    int64_t start[] = {0, 0, 0, 0, 0};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_resize(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, new_shape,
                                                   start, true, NULL));
}

INA_TEST_FIXTURE(resize, 5_s) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    int32_t type_size = sizeof(int16_t);

    const int8_t ndim = 5;
    int64_t shape[] = {10, 10, 10, 10, 10};
    int64_t cshape[] = {10, 10, 10, 10, 10};
    int64_t bshape[] = {5, 5, 5, 5, 5};
    int64_t new_shape[] = {15, 15, 5, 10, 10};
    int64_t start[] = {10, 10, 5, 10, 10};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_resize(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, new_shape,
                                                   start, true, "arr.iarr"));
}

INA_TEST_FIXTURE(resize, 4_ull) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t type_size = sizeof(uint64_t);

    const int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t cshape[] = {7, 8, 8, 4};
    int64_t bshape[] = {3, 5, 2, 4};
    int64_t new_shape[] = {5, 5, 11, 11};
    int64_t start[] = {5, 5, 10, 10};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_resize(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, new_shape,
                                                   start, false, NULL));
}

INA_TEST_FIXTURE(resize, 2_b) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    int32_t type_size = sizeof(bool);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 5};
    int64_t new_shape[] = {15, 5};
    int64_t start[] = {5, 5};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_resize(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, new_shape,
                                                   start, false, "arr.iarr"));
}

INA_TEST_FIXTURE(resize, 2_ui) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    int32_t type_size = sizeof(uint32_t);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 5};
    int64_t bshape[] = {2, 5};
    int64_t new_shape[] = {15, 7};
    int64_t start[] = {0, 7};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_resize(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, new_shape,
                                                   start, false, "arr.iarr"));
}
