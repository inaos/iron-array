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

static ina_rc_t test_buffer(iarray_context_t *ctx,
                           iarray_data_type_t dtype,
                           size_t type_size,
                           int8_t ndim,
                           const int64_t *shape,
                           const int64_t *pshape)
{
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        if (pshape != NULL) {
            xdtshape.pshape[i] = pshape[i];
        }
    }

    int64_t buf_size = 1;
    for (int j = 0; j < ndim; ++j) {
        buf_size *= shape[j];
    }
    uint8_t *buf_src = malloc((size_t)buf_size * type_size);

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        double *buff = (double *) buf_src;
        for (int64_t i = 0; i < buf_size; ++i) {
            buff[i] = (double) i;
        }
    } else {
        float *buff = (float *) buf_src;
        for (int64_t i = 0; i < buf_size; ++i) {
            buff[i] = (float) i;
        }
    }

    iarray_store_properties_t xstore = {.filename=NULL, .enforce_frame=true};
    if (pshape == NULL) {
        xstore.backend = IARRAY_STORAGE_PLAINBUFFER;
    } else {
        xstore.backend = IARRAY_STORAGE_BLOSC;
    }

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buf_src, (size_t) buf_size, &xstore, 0, &c_x));

    uint8_t *buf_dest = malloc((size_t)buf_size * type_size);

    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buf_dest, (size_t)buf_size * type_size));

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        double *buff = (double *) buf_dest;
        for (int64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], (double) i);
        }
    } else {
        float *buff = (float *) buf_dest;
        for (int64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], (float) i);
        }
    }

    return INA_SUCCESS;
}

INA_TEST_DATA(constructor_buffer) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_buffer)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_buffer)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_buffer, 2_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {10, 50};
    int64_t pshape[] = {3, 4};

    INA_TEST_ASSERT_SUCCEED(test_buffer(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(constructor_buffer, 4_f_p)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    int8_t ndim = 4;
    int64_t shape[] = {10, 12, 10, 13};
    int64_t *pshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_buffer(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(constructor_buffer, 5_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 5;
    int64_t shape[] = {10, 11, 10, 6, 7};
    int64_t pshape[] = {3, 4, 3, 3, 3};

    INA_TEST_ASSERT_SUCCEED(test_buffer(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(constructor_buffer, 7_f)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    int8_t ndim = 7;
    int64_t shape[] = {7, 8, 10, 10, 4, 4, 11};
    int64_t pshape[] = {4, 3, 6, 2, 3, 3, 2};

    INA_TEST_ASSERT_SUCCEED(test_buffer(data->ctx, dtype, type_size, ndim, shape, pshape));
}