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
#include <iarray_private.h>

#include <tests/iarray_test.h>

static ina_rc_t test_fill(iarray_context_t *ctx,
                          iarray_data_type_t dtype,
                          size_t type_size,
                          uint8_t ndim,
                          uint64_t *shape,
                          uint64_t *pshape,
                          void *value)
{
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        xdtshape.pshape[i] = pshape[i];
    }

    uint64_t buf_size = 1;
    for (int j = 0; j < ndim; ++j) {
        buf_size *= shape[j];
    }

    uint8_t *buf_dest = malloc(buf_size * type_size);

    iarray_container_t *c_x;

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        INA_TEST_ASSERT_SUCCEED(iarray_fill_double(ctx, &xdtshape, *((double *) value), NULL, 0, &c_x));
    } else {
        INA_TEST_ASSERT_SUCCEED(iarray_fill_float(ctx, &xdtshape, *((float *) value), NULL, 0, &c_x));
    }

    iarray_to_buffer(ctx, c_x, buf_dest, buf_size);

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        double *buff = (double *) buf_dest;
        for (uint64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], *((double *) value));
        }
    } else {
        float *buff = (float *) buf_dest;
        for (uint64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], *((float *) value));
        }
    }

    return INA_SUCCESS;
}

INA_TEST_DATA(constructor_fill) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_fill)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(constructor_fill)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_fill, double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 5;
    uint64_t shape[] = {10, 10, 10, 10, 10};
    uint64_t pshape[] = {3, 4, 6, 3, 3};
    double value = 3.1416;

    INA_TEST_ASSERT_SUCCEED(test_fill(data->ctx, dtype, type_size, ndim, shape, pshape, &value));
}

INA_TEST_FIXTURE(constructor_fill, float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint8_t ndim = 5;
    uint64_t shape[] = {10, 10, 10, 10, 10};
    uint64_t pshape[] = {3, 4, 6, 3, 3};
    float value = 0.1416;

    INA_TEST_ASSERT_SUCCEED(test_fill(data->ctx, dtype, type_size, ndim, shape, pshape, &value));
}
