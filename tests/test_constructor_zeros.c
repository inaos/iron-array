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

static ina_rc_t test_zeros(iarray_context_t *ctx,
                          iarray_data_type_t dtype,
                          size_t type_size,
                          int8_t ndim,
                          const int64_t *shape,
                          const int64_t *cshape,
                          const int64_t *bshape, bool contiguous, char *urlpath)
{
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
    }

    iarray_storage_t store;
    store.contiguous = contiguous;
    store.urlpath = urlpath;
    for (int i = 0; i < ndim; ++i) {
        store.chunkshape[i] = cshape[i];
        store.blockshape[i] = bshape[i];
    }

    int64_t buf_size = 1;
    for (int j = 0; j < ndim; ++j) {
        buf_size *= shape[j];
    }

    uint8_t *buf_dest = malloc((size_t)buf_size * type_size);

    iarray_container_t *c_x;
    blosc2_remove_urlpath(store.urlpath);
    INA_TEST_ASSERT_SUCCEED(iarray_zeros(ctx, &xdtshape, &store, 0, &c_x));

    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buf_dest, (size_t)buf_size * type_size));

    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE: {
            double *buff = (double *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], 0);
            }
            break;
        }
        case IARRAY_DATA_TYPE_FLOAT: {
            float *buff = (float *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], 0);
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT64: {
            int64_t *buff = (int64_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_INT64(buff[i], 0);
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT32: {
            int32_t *buff = (int32_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_INT(buff[i], 0);
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT16: {
            int16_t *buff = (int16_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_INT(buff[i], 0);
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT8: {
            int8_t *buff = (int8_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_INT(buff[i], 0);
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT64: {
            uint64_t *buff = (uint64_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_UINT64(buff[i], 0);
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT32: {
            uint32_t *buff = (uint32_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_UINT(buff[i], 0);
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT16: {
            uint16_t *buff = (uint16_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_UINT(buff[i], 0);
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT8: {
            uint8_t *buff = (uint8_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_UINT(buff[i], 0);
            }
            break;
        }
        case IARRAY_DATA_TYPE_BOOL: {
            bool *buff = (bool *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT(buff[i] == false);
            }
            break;
        }
    }

    iarray_container_free(ctx, &c_x);
    free(buf_dest);
    blosc2_remove_urlpath(store.urlpath);

    return INA_SUCCESS;
}

INA_TEST_DATA(constructor_zeros) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_zeros)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_zeros)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_zeros, 5_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 5;
    int64_t shape[] = {10, 4, 12, 13, 12};
    int64_t cshape[] = {3, 4, 10, 3, 3};
    int64_t bshape[] = {2, 2, 3, 1, 3};

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(constructor_zeros, 7_f)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    int8_t ndim = 7;
    int64_t shape[] = {10, 6, 8, 6, 4, 4, 2};
    int64_t cshape[] = {4, 3, 5, 5, 3, 3, 2};
    int64_t bshape[] = {2, 1, 2, 2, 3, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, "arr.iarr"));
}


INA_TEST_FIXTURE(constructor_zeros, 5_ll)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    size_t type_size = sizeof(int64_t);

    int8_t ndim = 5;
    int64_t shape[] = {10, 4, 12, 13, 12};
    int64_t cshape[] = {3, 4, 10, 3, 3};
    int64_t bshape[] = {2, 2, 3, 1, 3};

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(constructor_zeros, 7_i)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    size_t type_size = sizeof(int32_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 6, 8, 6, 4, 4, 2};
    int64_t cshape[] = {4, 3, 5, 5, 3, 3, 2};
    int64_t bshape[] = {2, 1, 2, 2, 3, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, "arr.iarr"));
}


INA_TEST_FIXTURE(constructor_zeros, 5_s)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    size_t type_size = sizeof(int16_t);

    int8_t ndim = 5;
    int64_t shape[] = {10, 4, 12, 13, 12};
    int64_t cshape[] = {3, 4, 10, 3, 3};
    int64_t bshape[] = {2, 2, 3, 1, 3};

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(constructor_zeros, 7_sc)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    size_t type_size = sizeof(int8_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 6, 8, 6, 4, 4, 2};
    int64_t cshape[] = {4, 3, 5, 5, 3, 3, 2};
    int64_t bshape[] = {2, 1, 2, 2, 3, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, "arr.iarr"));
}


INA_TEST_FIXTURE(constructor_zeros, 5_ull)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    size_t type_size = sizeof(uint64_t);

    int8_t ndim = 5;
    int64_t shape[] = {10, 4, 12, 13, 12};
    int64_t cshape[] = {3, 4, 10, 3, 3};
    int64_t bshape[] = {2, 2, 3, 1, 3};

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(constructor_zeros, 7_ui)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    size_t type_size = sizeof(uint32_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 6, 8, 6, 4, 4, 2};
    int64_t cshape[] = {4, 3, 5, 5, 3, 3, 2};
    int64_t bshape[] = {2, 1, 2, 2, 3, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, "arr.iarr"));
}


INA_TEST_FIXTURE(constructor_zeros, 5_us)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;
    size_t type_size = sizeof(uint16_t);

    int8_t ndim = 5;
    int64_t shape[] = {10, 4, 12, 13, 12};
    int64_t cshape[] = {3, 4, 10, 3, 3};
    int64_t bshape[] = {2, 2, 3, 1, 3};

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(constructor_zeros, 7_uc)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    size_t type_size = sizeof(uint8_t);

    int8_t ndim = 7;
    int64_t shape[] = {10, 6, 8, 6, 4, 4, 2};
    int64_t cshape[] = {4, 3, 5, 5, 3, 3, 2};
    int64_t bshape[] = {2, 1, 2, 2, 3, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, "arr.iarr"));
}

INA_TEST_FIXTURE(constructor_zeros, 2_b)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    size_t type_size = sizeof(bool);

    int8_t ndim = 2;
    int64_t shape[] = {10, 6};
    int64_t cshape[] = {4, 3};
    int64_t bshape[] = {2, 1};

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, "arr.iarr"));
}
