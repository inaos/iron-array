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

static ina_rc_t test_fill(iarray_context_t *ctx,
                          iarray_data_type_t dtype,
                          size_t type_size,
                          int8_t ndim,
                          const int64_t *shape,
                          const int64_t *pshape,
                          const int64_t *bshape,
                          void *value)
{
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        if (pshape) {
            xdtshape.pshape[i] = pshape[i];
            xdtshape.bshape[i] = bshape[i];
        }
    }

    iarray_storage_t store;
    store.backend = pshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    store.enforce_frame = false;
    store.filename = NULL;

    int64_t buf_size = 1;
    for (int j = 0; j < ndim; ++j) {
        buf_size *= shape[j];
    }

    uint8_t *buf_dest = malloc((size_t)buf_size * type_size);

    iarray_container_t *c_x;

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        INA_TEST_ASSERT_SUCCEED(iarray_fill_double(ctx, &xdtshape, *((double *) value), &store, 0, &c_x));
    } else {
        INA_TEST_ASSERT_SUCCEED(iarray_fill_float(ctx, &xdtshape, *((float *) value), &store, 0, &c_x));
    }

    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buf_dest, (size_t)buf_size * type_size));

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        double *buff = (double *) buf_dest;
        for (int64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], *((double *) value));
        }
    } else {
        float *buff = (float *) buf_dest;
        for (int64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], *((float *) value));
        }
    }

    iarray_container_free(ctx, &c_x);
    free(buf_dest);

    return INA_SUCCESS;
}

INA_TEST_DATA(constructor_fill) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_fill)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_fill)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_fill, 2_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {100, 312};
    int64_t pshape[] = {35, 101};
    int64_t bshape[] = {12, 12};
    double value = 3.1416;

    INA_TEST_ASSERT_SUCCEED(test_fill(data->ctx, dtype, type_size, ndim, shape, pshape, bshape, &value));
}

INA_TEST_FIXTURE(constructor_fill, 4_f_p)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    int8_t ndim = 4;
    int64_t shape[] = {10, 5, 5, 10};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    float value = 0.1416f;

    INA_TEST_ASSERT_SUCCEED(test_fill(data->ctx, dtype, type_size, ndim, shape, pshape, bshape, &value));
}

INA_TEST_FIXTURE(constructor_fill, 5_d_p)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 5;
    int64_t shape[] = {7, 10, 12, 11, 10};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    double value = 3.1416;

    INA_TEST_ASSERT_SUCCEED(test_fill(data->ctx, dtype, type_size, ndim, shape, pshape, bshape, &value));
}

INA_TEST_FIXTURE(constructor_fill, 7_f)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    int8_t ndim = 7;
    int64_t shape[] = {12, 11, 6, 5, 12, 6, 8};
    int64_t pshape[] = {11, 3, 3, 3, 3, 5, 3};
    int64_t bshape[] = {5, 2, 2, 2, 1, 2, 2};
    float value = -116.f;

    INA_TEST_ASSERT_SUCCEED(test_fill(data->ctx, dtype, type_size, ndim, shape, pshape, bshape, &value));
}