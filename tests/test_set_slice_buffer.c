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

static ina_rc_t
_execute_iarray_set_slice(iarray_context_t *ctx, iarray_data_type_t dtype, int64_t type_size,
                          int8_t ndim, const int64_t *shape, const int64_t *cshape, int64_t *bshape,
                          int64_t *start, int64_t *stop, bool xcontiguous, char *xurlpath) {
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

    iarray_storage_t xstore;
    for (int i = 0; i < ndim; ++i) {
        xstore.chunkshape[i] = cshape[i];
        xstore.blockshape[i] = bshape[i];
    }
    xstore.contiguous = xcontiguous;
    xstore.urlpath = xurlpath;
    blosc2_remove_urlpath(xstore.urlpath);

    int64_t bufdes_size = 1;

    for (int k = 0; k < ndim; ++k) {
        int64_t st = (start[k] + shape[k]) % shape[k];
        int64_t sp = (stop[k] + shape[k] - 1) % shape[k] + 1;
        bufdes_size *= (int64_t) sp - st;
    }

    int64_t buflen = bufdes_size * type_size;

    uint8_t *bufdes = ina_mem_alloc(bufdes_size * type_size);

    for (int i = 0; i < bufdes_size; ++i) {
        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                ((double *) bufdes)[i] = (double) i;
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                ((float *) bufdes)[i] = (float) i;
                break;
            case IARRAY_DATA_TYPE_INT64:
                ((int64_t *) bufdes)[i] = (int64_t) i;
                break;
            case IARRAY_DATA_TYPE_INT32:
                ((int32_t *) bufdes)[i] = (int32_t) i;
                break;
            case IARRAY_DATA_TYPE_INT16:
                ((int16_t *) bufdes)[i] = (int16_t) i;
                break;
            case IARRAY_DATA_TYPE_INT8:
                ((int8_t *) bufdes)[i] = (int8_t) i;
                break;
            case IARRAY_DATA_TYPE_UINT64:
                ((uint64_t *) bufdes)[i] = (uint64_t) i;
                break;
            case IARRAY_DATA_TYPE_UINT32:
                ((uint32_t *) bufdes)[i] = (uint32_t) i;
                break;
            case IARRAY_DATA_TYPE_UINT16:
                ((uint16_t *) bufdes)[i] = (uint16_t) i;
                break;
            case IARRAY_DATA_TYPE_UINT8:
                ((uint8_t *) bufdes)[i] = (uint8_t) i;
                break;
            case IARRAY_DATA_TYPE_BOOL:
                ((boolean_t *) bufdes)[i] = (boolean_t) i;
                break;
        }
    }

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buffer_x, buffer_x_len * type_size, &xstore, 0, &c_x));

    INA_TEST_ASSERT_SUCCEED(test_set_slice_buffer(ctx, c_x, start, stop, bufdes, buflen));

    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdes)[l], (double) l);
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdes)[l], (float) l);
            }
            break;
        case IARRAY_DATA_TYPE_INT64:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT64(((int64_t *) bufdes)[l], (int64_t) l);
            }
            break;
        case IARRAY_DATA_TYPE_INT32:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int32_t *) bufdes)[l], (int32_t) l);
            }
            break;
        case IARRAY_DATA_TYPE_INT16:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int16_t *) bufdes)[l], (int16_t) l);
            }
            break;
        case IARRAY_DATA_TYPE_INT8:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_INT(((int8_t *) bufdes)[l], (int8_t) l);
            }
            break;
        case IARRAY_DATA_TYPE_UINT64:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT64(((uint64_t *) bufdes)[l], (uint64_t) l);
            }
            break;
        case IARRAY_DATA_TYPE_UINT32:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint32_t *) bufdes)[l], (uint32_t) l);
            }
            break;
        case IARRAY_DATA_TYPE_UINT16:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint16_t *) bufdes)[l], (uint16_t) l);
            }
            break;
        case IARRAY_DATA_TYPE_UINT8:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT_EQUAL_UINT(((uint8_t *) bufdes)[l], (uint8_t) l);
            }
            break;
        case IARRAY_DATA_TYPE_BOOL:
            for (int64_t l = 0; l < bufdes_size; ++l) {
                INA_TEST_ASSERT(((boolean_t *) bufdes)[l] == (boolean_t) l);
            }
            break;
    }

    iarray_container_free(ctx, &c_x);

    ina_mem_free(buffer_x);
    ina_mem_free(bufdes);
    blosc2_remove_urlpath(xstore.urlpath);

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


INA_TEST_FIXTURE(set_slice_buffer, 2_f_blosc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t cshape[] = {100, 2};
    int64_t bshape[] = {50, 1};
    int64_t start[] = {21, 17};
    int64_t stop[] = {-21, 55};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape,
                                                      cshape, bshape,
                                                      start, stop, false, "xarr.iar"));
}

INA_TEST_FIXTURE(set_slice_buffer, 2_d_t_blosc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 3};
    int64_t bshape[] = {2, 2};
    int64_t start[] = {0, 0};
    int64_t stop[] = {-5, 5};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape,
                                                      cshape, bshape,
                                                      start, stop, false, NULL));
}

INA_TEST_FIXTURE(set_slice_buffer, 2_ll_t_blosc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    int32_t type_size = sizeof(int64_t);

    const int8_t ndim = 2;
    int64_t shape[] = {20, 14};
    int64_t cshape[] = {13, 13};
    int64_t bshape[] = {11, 3};
    int64_t start[] = {3, 1};
    int64_t stop[] = {-2, 5};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape,
                                                      cshape, bshape,
                                                      start, stop, true, NULL));
}

INA_TEST_FIXTURE(set_slice_buffer, 3_ui_blosc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    int32_t type_size = sizeof(uint32_t);

    const int8_t ndim = 3;
    int64_t shape[] = {100, 123, 234};
    int64_t cshape[] = {31, 1, 21};
    int64_t bshape[] = {13, 1, 21};
    int64_t start[] = {23, 31, 22};
    int64_t stop[] = {54, 78, 76};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape,
                                                      cshape, bshape,
                                                      start, stop, true, "xarr.iarr"));
}

INA_TEST_FIXTURE(set_slice_buffer, 2_uc_blosc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    int32_t type_size = sizeof(uint8_t);

    const int8_t ndim = 2;
    int64_t shape[] = {30, 8};
    int64_t cshape[] = {20, 2};
    int64_t bshape[] = {20, 1};
    int64_t start[] = {23, 7};
    int64_t stop[] = {29, 8};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape,
                                                      cshape, bshape,
                                                      start, stop, false, "xarr.iarr"));
}

INA_TEST_FIXTURE(set_slice_buffer, 5_i_blosc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    int32_t type_size = sizeof(int32_t);

    const int8_t ndim = 5;
    int64_t shape[] = {10, 12, 32, 14, 14};
    int64_t cshape[] = {5, 5, 5, 5, 5};
    int64_t bshape[] = {2, 2, 1, 2, 5};
    int64_t start[] = {1, 2, 4, 5, 6};
    int64_t stop[] = {8, 9, 11, 12, 13};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape,
                                                      cshape, bshape,
                                                      start, stop, false, NULL));
}

INA_TEST_FIXTURE(set_slice_buffer, 6_ull_blosc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t type_size = sizeof(uint64_t);

    const int8_t ndim = 6;
    int64_t shape[] = {8, 7, 6, 7, 8, 5};
    int64_t cshape[] = {4, 3, 2, 3, 2, 2};
    int64_t bshape[] = {2, 2, 2, 1, 2, 1};
    int64_t start[] = {1, 2, 3, 4, 5, 2};
    int64_t stop[] = {3, 4, 4, 7, 7, 4};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape,
                                                      cshape, bshape,
                                                      start, stop, true, "xarr.iarr"));
}

INA_TEST_FIXTURE(set_slice_buffer, 7_b_blosc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    int32_t type_size = sizeof(boolean_t);

    const int8_t ndim = 7;
    int64_t shape[] = {5, 7, 6, 4, 8, 6, 5};
    int64_t cshape[] = {2, 3, 2, 2, 2, 2, 2};
    int64_t bshape[] = {1, 2, 1, 1, 2, 1, 2};
    int64_t start[] = {1, 2, 1, 2, 0, 2, 1};
    int64_t stop[] = {5, 4, 4, 3, 6, 3, 4};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape,
                                                      cshape, bshape,
                                                      start, stop, true, NULL));
}

INA_TEST_FIXTURE(set_slice_buffer, 2_s_blosc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    int32_t type_size = sizeof(int32_t);

    const int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t cshape[] = {100, 2};
    int64_t bshape[] = {50, 1};
    int64_t start[] = {21, 17};
    int64_t stop[] = {-21, 55};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape,
                                                      cshape, bshape,
                                                      start, stop, false, "xarr.iar"));
}

INA_TEST_FIXTURE(set_slice_buffer, 2_sc_t_blosc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    int32_t type_size = sizeof(int8_t);

    const int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {5, 3};
    int64_t bshape[] = {2, 2};
    int64_t start[] = {0, 0};
    int64_t stop[] = {-5, 5};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape,
                                                      cshape, bshape,
                                                      start, stop, false, NULL));
}

INA_TEST_FIXTURE(set_slice_buffer, 2_us_t_blosc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;
    int32_t type_size = sizeof(uint16_t);

    const int8_t ndim = 2;
    int64_t shape[] = {20, 14};
    int64_t cshape[] = {13, 13};
    int64_t bshape[] = {11, 3};
    int64_t start[] = {3, 1};
    int64_t stop[] = {-2, 5};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_set_slice(data->ctx, dtype, type_size, ndim, shape,
                                                      cshape, bshape,
                                                      start, stop, true, NULL));
}
