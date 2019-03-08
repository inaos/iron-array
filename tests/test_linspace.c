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


static ina_rc_t test_linspace(iarray_context_t *ctx, iarray_data_type_t dtype, int8_t ndim,
                              const int64_t *shape, const int64_t *pshape, double start,
                              double stop) {
    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        xdtshape.pshape[i] = pshape[i];
        size *= shape[i];
    }

    iarray_container_t *c_x;

    iarray_linspace(ctx, &xdtshape, size, start, stop, NULL, 0, &c_x);

    // Assert iterator reading it

    iarray_iter_read_t *I2;
    iarray_iter_read_new(ctx, c_x, &I2);
    for (iarray_iter_read_init(I2); !iarray_iter_read_finished(I2); iarray_iter_read_next(I2)) {

        iarray_iter_read_value_t val;
        iarray_iter_read_value(I2, &val);

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                INA_TEST_ASSERT_EQUAL_FLOATING(val.nelem * (stop - start) / (size - 1) + start,
                                               ((double *) val.pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                INA_TEST_ASSERT_EQUAL_FLOATING((float) (val.nelem * (stop - start) / (size - 1) + start),
                                               ((float *) val.pointer)[0]);
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    iarray_iter_read_free(I2);

    iarray_container_free(ctx, &c_x);
    return INA_SUCCESS;
}

INA_TEST_DATA(linspace) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(linspace) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.eval_flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(linspace) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(linspace, double_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {223, 456};
    int64_t pshape[] = {31, 43};
    double start = - 0.1;
    double stop = - 0.25;

    INA_TEST_ASSERT_SUCCEED(test_linspace(data->ctx, dtype, ndim, shape, pshape, start, stop));
}

INA_TEST_FIXTURE(linspace, float_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {445, 321};
    int64_t pshape[] = {21, 17};
    double start = 3123;
    double stop = 45654;

    INA_TEST_ASSERT_SUCCEED(test_linspace(data->ctx, dtype, ndim, shape, pshape, start, stop));
}

INA_TEST_FIXTURE(linspace, double_5) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 5;
    int64_t shape[] = {20, 18, 17, 13, 21};
    int64_t pshape[] = {12, 12, 2, 3, 13};
    double start = 0.1;
    double stop = 0.2;

    INA_TEST_ASSERT_SUCCEED(test_linspace(data->ctx, dtype, ndim, shape, pshape, start, stop));
}

INA_TEST_FIXTURE(linspace, float_7) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 7;
    int64_t shape[] = {5, 7, 8, 9, 6, 5, 7};
    int64_t pshape[] = {2, 5, 3, 4, 3, 2, 3};
    double start = 10;
    double stop = 0;

    INA_TEST_ASSERT_SUCCEED(test_linspace(data->ctx, dtype, ndim, shape, pshape, start, stop));
}
