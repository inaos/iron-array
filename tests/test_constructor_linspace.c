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
                              const int64_t *shape, const int64_t *pshape, const int64_t *bshape, double start,
                              double stop) {
    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        if (pshape) {
            xdtshape.pshape[i] = pshape[i];
            xdtshape.bshape[i] = bshape[i];
        }
        size *= shape[i];
    }

    iarray_store_properties_t store;
    store.backend = pshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    store.enforce_frame = false;
    store.filename = NULL;

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &xdtshape, size, start, stop, &store, 0, &c_x));

    // Assert iterator reading it

    iarray_iter_read_t *I2;
    iarray_iter_read_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_new(ctx, &I2, c_x, &val));

    while (INA_SUCCEED(iarray_iter_read_has_next(I2))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_next(I2));

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                INA_TEST_ASSERT_EQUAL_FLOATING(val.elem_flat_index * (stop - start) / (size - 1) + start,
                                               ((double *) val.elem_pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                INA_TEST_ASSERT_EQUAL_FLOATING((float) (val.elem_flat_index * (stop - start) / (size - 1) + start),
                                               ((float *) val.elem_pointer)[0]);
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    iarray_iter_read_free(&I2);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    iarray_container_free(ctx, &c_x);
    return INA_SUCCESS;
}

INA_TEST_DATA(constructor_linspace) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_linspace) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_linspace) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_linspace, 2_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {223, 456};
    int64_t pshape[] = {31, 323};
    int64_t bshape[] = {10, 10};
    double start = - 0.1;
    double stop = - 0.25;

    INA_TEST_ASSERT_SUCCEED(test_linspace(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop));
}

INA_TEST_FIXTURE(constructor_linspace, 2_f_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {445, 321};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    double start = 3123;
    double stop = 45654;

    INA_TEST_ASSERT_SUCCEED(test_linspace(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop));
}

INA_TEST_FIXTURE(constructor_linspace, 5_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 5;
    int64_t shape[] = {20, 18, 17, 13, 21};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    double start = 0.1;
    double stop = 0.2;

    INA_TEST_ASSERT_SUCCEED(test_linspace(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop));
}

INA_TEST_FIXTURE(constructor_linspace, 7_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 7;
    int64_t shape[] = {5, 7, 8, 9, 6, 5, 7};
    int64_t pshape[] = {3, 5, 3, 3, 3, 2, 3};
    int64_t bshape[] = {2, 2, 2, 2, 2, 2, 2};
    double start = 10;
    double stop = 0;

    INA_TEST_ASSERT_SUCCEED(test_linspace(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop));
}
