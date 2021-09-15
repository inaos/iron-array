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

static ina_rc_t test_zeros(iarray_context_t *ctx,
                          iarray_data_type_t dtype,
                          size_t type_size,
                          int8_t ndim,
                          const int64_t *shape,
                          const int64_t *cshape,
                          const int64_t *bshape)
{
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
    }

    iarray_storage_t store;
    store.backend = cshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    store.contiguous = false;
    store.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        if (cshape != NULL) {
            store.chunkshape[i] = cshape[i];
            store.blockshape[i] = bshape[i];
        }
    }

    int64_t buf_size = 1;
    for (int j = 0; j < ndim; ++j) {
        buf_size *= shape[j];
    }

    uint8_t *buf_dest = malloc((size_t)buf_size * type_size);

    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(iarray_zeros(ctx, &xdtshape, &store, 0, &c_x));

    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buf_dest, (size_t)buf_size * type_size));

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        double *buff = (double *) buf_dest;
        for (int64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], 0);
        }
    } else {
        float *buff = (float *) buf_dest;
        for (int64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], 0);
        }
    }

    iarray_container_free(ctx, &c_x);
    free(buf_dest);

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

INA_TEST_FIXTURE(constructor_zeros, 2_d_p)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t *cshape = NULL;
    int64_t *bshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape));
}

INA_TEST_FIXTURE(constructor_zeros, 4_f_p)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    int8_t ndim = 4;
    int64_t shape[] = {10, 15, 20, 12};
    int64_t *cshape = NULL;
    int64_t *bshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape));
}

INA_TEST_FIXTURE(constructor_zeros, 5_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 5;
    int64_t shape[] = {10, 4, 12, 13, 12};
    int64_t cshape[] = {3, 4, 10, 3, 3};
    int64_t bshape[] = {2, 2, 3, 1, 3};

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape));
}

INA_TEST_FIXTURE(constructor_zeros, 7_f)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    int8_t ndim = 7;
    int64_t shape[] = {10, 6, 8, 6, 4, 4, 2};
    int64_t cshape[] = {4, 3, 5, 5, 3, 3, 2};
    int64_t bshape[] = {2, 1, 2, 2, 3, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_zeros(data->ctx, dtype, type_size, ndim, shape, cshape, bshape));
}