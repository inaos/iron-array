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


static ina_rc_t test_arange(iarray_context_t *ctx, iarray_data_type_t dtype, int8_t ndim,
                           const int64_t *shape, const int64_t *pshape, const int64_t *bshape, double start,
                           double stop)
{

    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        size *= shape[i];
    }

    double step = (stop - start) / size;

    iarray_storage_t xstore = {.filename=NULL, .enforce_frame=false};
    if (pshape == NULL) {
        xstore.backend = IARRAY_STORAGE_PLAINBUFFER;
    } else {
        xstore.backend = IARRAY_STORAGE_BLOSC;
        for (int i = 0; i < ndim; ++i) {
            xstore.pshape[i] = pshape[i];
            xstore.bshape[i] = bshape[i];
        }
    }

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, start, stop, step, &xstore, 0, &c_x));

    // Assert iterator reading it

    iarray_iter_read_t *I2;
    iarray_iter_read_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_new(ctx, &I2, c_x, &val));

    while (INA_SUCCEED(iarray_iter_read_has_next(I2))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_next(I2));

        switch(dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                INA_TEST_ASSERT_EQUAL_FLOATING(val.elem_flat_index * step + start, ((double *) val.elem_pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                INA_TEST_ASSERT_EQUAL_FLOATING( (float) (val.elem_flat_index * step + start), ((float *) val.elem_pointer)[0]);
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

INA_TEST_DATA(constructor_arange) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_arange) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_arange) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_arange, 2_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    double start = 0;
    double stop = 100;

    INA_TEST_ASSERT_SUCCEED(test_arange(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop));
}

INA_TEST_FIXTURE(constructor_arange, 2_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t pshape[] = {60, 50};
    int64_t bshape[] = {17, 13};
    double start = 0;
    double stop = 100 * 100;

    INA_TEST_ASSERT_SUCCEED(test_arange(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop));
}


INA_TEST_FIXTURE(constructor_arange, 5_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 5;
    int64_t shape[] = {20, 18, 17, 13, 21};
    int64_t pshape[] = {3, 12, 14, 3, 20};
    int64_t bshape[] = {2, 5, 7, 2, 9};
    double start = 0.1;
    double stop = 0.2;

    INA_TEST_ASSERT_SUCCEED(test_arange(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop));
}

INA_TEST_FIXTURE(constructor_arange, 7_f_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 7;
    int64_t shape[] = {5, 7, 8, 9, 6, 5, 7};
    int64_t *pshape = NULL;
    int64_t *bshape = NULL;
    double start = 10;
    double stop = 0;

    INA_TEST_ASSERT_SUCCEED(test_arange(data->ctx, dtype, ndim, shape, pshape, bshape, start, stop));
}
