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

static ina_rc_t test_arange(iarray_context_t *ctx, iarray_data_type_t dtype, size_t type_size, uint8_t ndim,
                            const uint64_t *shape, const uint64_t *pshape, double start, double stop) {

    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    uint64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        xdtshape.pshape[i] = pshape[i];
        size *= shape[i];
    }

    double step = (stop - start) / size;
    iarray_container_t *c_x;

    iarray_arange(ctx, &xdtshape, start, stop, step, NULL, 0, &c_x);

    // Assert iterator reading it

    iarray_iter_read_t *I2;
    iarray_iter_read_new(ctx, c_x, &I2);

    for (iarray_iter_read_init(ctx, I2); !iarray_iter_read_finished(ctx, I2); iarray_iter_read_next(ctx, I2)) {

        iarray_iter_read_value_t val;
        iarray_iter_read_value(ctx, I2, &val);

        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            double value = val.nelem * step + start;
            INA_TEST_ASSERT_EQUAL_FLOATING(value, ((double *) val.pointer)[0]);
        } else {
            float value = (float) (val.nelem * step + start);
            INA_TEST_ASSERT_EQUAL_FLOATING(value, ((float *) val.pointer)[0]);
        }
    }

    iarray_iter_free(ctx, I2);

    iarray_container_free(ctx, &c_x);
    return INA_SUCCESS;
}

INA_TEST_DATA(arange) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(arange) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(arange) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(arange, double_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 2;
    uint64_t shape[] = {223, 456};
    uint64_t pshape[] = {31, 43};
    double start = - 0.1;
    double stop = - 0.25;

    INA_TEST_ASSERT_SUCCEED(test_arange(data->ctx, dtype, type_size, ndim, shape, pshape, start, stop));
}

INA_TEST_FIXTURE(arange, float_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint8_t ndim = 2;
    uint64_t shape[] = {445, 321};
    uint64_t pshape[] = {21, 17};
    double start = 3123;
    double stop = 45654;

    INA_TEST_ASSERT_SUCCEED(test_arange(data->ctx, dtype, type_size, ndim, shape, pshape, start, stop));
}

INA_TEST_FIXTURE(arange, double_5) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 5;
    uint64_t shape[] = {20, 18, 17, 13, 21};
    uint64_t pshape[] = {12, 12, 2, 3, 13};
    double start = 0.1;
    double stop = 0.2;

    INA_TEST_ASSERT_SUCCEED(test_arange(data->ctx, dtype, type_size, ndim, shape, pshape, start, stop));
}

INA_TEST_FIXTURE(arange, float_7) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint8_t ndim = 7;
    uint64_t shape[] = {5, 7, 8, 9, 6, 5, 7};
    uint64_t pshape[] = {2, 5, 3, 4, 3, 2, 3};
    double start = 10;
    double stop = 0;

    INA_TEST_ASSERT_SUCCEED(test_arange(data->ctx, dtype, type_size, ndim, shape, pshape, start, stop));
}
