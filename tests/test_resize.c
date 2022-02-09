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

static ina_rc_t test_resize(iarray_context_t *ctx, iarray_container_t *c_x, int64_t *new_shape) {

    INA_TEST_ASSERT_SUCCEED(iarray_container_resize(ctx, c_x, new_shape));

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
    blosc2_remove_urlpath(store.urlpath);

    // Get shapes and buffer sizes from each part
    int64_t shrink_shape[CATERVA_MAX_DIM] = {0};
    int64_t shrink_len = 1;
    int64_t start_extend_slice[CATERVA_MAX_DIM] = {0};
    int64_t stop_extend_shape[CATERVA_MAX_DIM] = {0};
    int64_t extend_len = 1;
    bool only_shrink = true;
    for (int i = 0; i < ndim; ++i) {
        if (new_shape[i] <= shape[i]) {
            shrink_shape[i] = new_shape[i];
            start_extend_slice[i] = 0;
            extend_len *= new_shape[i];
        } else {
            shrink_shape[i] = shape[i];
            start_extend_slice[i] = shape[i];
            extend_len *= (new_shape[i] - shape[i]);
            only_shrink = false;
        }
        stop_extend_shape[i] = new_shape[i];
        shrink_len *= shrink_shape[i];
    }

    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buffer_x, buffer_x_len * type_size, &store, 0, &c_x));

    // Get original values in shrinked slice
    uint8_t *original_buffer = ina_mem_alloc(shrink_len * type_size);
    int64_t start_shape[CATERVA_MAX_DIM] = {0};

    INA_TEST_ASSERT_SUCCEED(iarray_get_slice_buffer(ctx, c_x, start_shape, shrink_shape, original_buffer, shrink_len * type_size));


    INA_TEST_ASSERT_SUCCEED(test_resize(ctx, c_x, new_shape));

    // Get the same slice after the resize
    uint8_t *shrink_buffer = ina_mem_alloc(shrink_len * type_size);
    INA_TEST_ASSERT_SUCCEED(iarray_get_slice_buffer(ctx, c_x, start_shape, shrink_shape, shrink_buffer, shrink_len * type_size));

    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            for (int64_t l = 0; l < shrink_len; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) shrink_buffer)[l], ((double *) original_buffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            for (int64_t l = 0; l < shrink_len; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) shrink_buffer)[l], ((float *) original_buffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT64:
            for (int64_t l = 0; l < shrink_len; ++l) {
                INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) shrink_buffer)[l], ((int64_t *) original_buffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT32:
            for (int64_t l = 0; l < shrink_len; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int32_t *) shrink_buffer)[l], ((int32_t *) original_buffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT16:
            for (int64_t l = 0; l < shrink_len; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int16_t *) shrink_buffer)[l], ((int16_t *) original_buffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_INT8:
            for (int64_t l = 0; l < shrink_len; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int8_t *) shrink_buffer)[l], ((int8_t *) original_buffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT64:
            for (int64_t l = 0; l < shrink_len; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) shrink_buffer)[l], ((uint64_t *) original_buffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT32:
            for (int64_t l = 0; l < shrink_len; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) shrink_buffer)[l], ((uint32_t *) original_buffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT16:
            for (int64_t l = 0; l < shrink_len; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) shrink_buffer)[l], ((uint16_t *) original_buffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_UINT8:
            for (int64_t l = 0; l < shrink_len; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) shrink_buffer)[l], ((uint8_t *) original_buffer)[l]);
            }
            break;
        case IARRAY_DATA_TYPE_BOOL:
            for (int64_t l = 0; l < shrink_len; ++l) {
                INA_TEST_ASSERT(((bool *) shrink_buffer)[l] == ((bool *) original_buffer)[l]);
            }
            break;
    }

    if (!only_shrink) {
        // Get the slice corresponding to the extended part
        uint8_t *extend_buffer = ina_mem_alloc(extend_len * type_size);
        INA_TEST_ASSERT_SUCCEED(iarray_get_slice_buffer(ctx, c_x, start_extend_slice, stop_extend_shape, extend_buffer, extend_len * type_size));

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                for (int64_t l = 0; l < extend_len; ++l) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((double *) extend_buffer)[l], 0);
                }
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                for (int64_t l = 0; l < extend_len; ++l) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((float *) extend_buffer)[l], 0);
                }
                break;
            case IARRAY_DATA_TYPE_INT64:
                for (int64_t l = 0; l < extend_len; ++l) {
                    INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) extend_buffer)[l], 0);
                }
                break;
            case IARRAY_DATA_TYPE_INT32:
                for (int64_t l = 0; l < extend_len; ++l) {
                    INA_TEST_ASSERT_EQUAL_INT(((int32_t *) extend_buffer)[l], 0);
                }
                break;
            case IARRAY_DATA_TYPE_INT16:
                for (int64_t l = 0; l < extend_len; ++l) {
                    INA_TEST_ASSERT_EQUAL_INT(((int16_t *) extend_buffer)[l], 0);
                }
                break;
            case IARRAY_DATA_TYPE_INT8:
                for (int64_t l = 0; l < extend_len; ++l) {
                    INA_TEST_ASSERT_EQUAL_INT(((int8_t *) extend_buffer)[l], 0);
                }
                break;
            case IARRAY_DATA_TYPE_UINT64:
                for (int64_t l = 0; l < extend_len; ++l) {
                    INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) extend_buffer)[l], 0ULL);
                }
                break;
            case IARRAY_DATA_TYPE_UINT32:
                for (int64_t l = 0; l < extend_len; ++l) {
                    INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) extend_buffer)[l], 0L);
                }
                break;
            case IARRAY_DATA_TYPE_UINT16:
                for (int64_t l = 0; l < extend_len; ++l) {
                    INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) extend_buffer)[l], 0);
                }
                break;
            case IARRAY_DATA_TYPE_UINT8:
                for (int64_t l = 0; l < extend_len; ++l) {
                    INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) extend_buffer)[l], 0);
                }
                break;
            case IARRAY_DATA_TYPE_BOOL:
                for (int64_t l = 0; l < extend_len; ++l) {
                    INA_TEST_ASSERT(((bool *) extend_buffer)[l] == false);
                }
                break;
        }
        ina_mem_free(extend_buffer);
    }

    iarray_container_free(ctx, &c_x);

    ina_mem_free(buffer_x);
    ina_mem_free(original_buffer);
    ina_mem_free(shrink_buffer);
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
    int64_t new_shape[] = {11, 5, 15};

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
    int64_t new_shape[] = {5, 5, 6, 3, 4};

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
    int64_t new_shape[] = {15, 15, 5, 10, 10};

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
    int64_t new_shape[] = {5, 5, 11, 11};

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
