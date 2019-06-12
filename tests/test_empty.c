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

static ina_rc_t test_empty(iarray_context_t *ctx,
                           iarray_data_type_t dtype,
                           int8_t ndim,
                           const int64_t *shape,
                           const int64_t *pshape)
{
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        xdtshape.pshape[i] = pshape[i];
    }

    int64_t buf_size = 1;
    for (int j = 0; j < ndim; ++j) {
        buf_size *= shape[j];
    }

    // Empty array
    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &xdtshape, NULL, 0, &c_x));

    if (!iarray_is_empty(c_x)) {
        return INA_ERR_ERROR;
    }

    // Non-empty array
    iarray_container_t *z_x;
    INA_TEST_ASSERT_SUCCEED(iarray_zeros(ctx, &xdtshape, NULL, 0, &z_x));

    if (iarray_is_empty(z_x)) {
        return INA_ERR_ERROR;
    }

    return INA_SUCCESS;

}

INA_TEST_DATA(constructor_empty) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_empty)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_empty)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_empty, 1_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 1;
    int64_t shape[] = {10};
    int64_t pshape[] = {3};

    INA_TEST_ASSERT_SUCCEED(test_empty(data->ctx, dtype, ndim, shape, pshape));
}

// TODO: this will be solved after https://github.com/inaos/iron-array/issues/139 would be fixed.
INA_TEST_FIXTURE(constructor_empty, 1_d_1)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 1;
    int64_t shape[] = {1};
    int64_t pshape[] = {1};

    INA_TEST_ASSERT_SUCCEED(test_empty(data->ctx, dtype, ndim, shape, pshape));
}

INA_TEST_FIXTURE(constructor_empty, 2_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t pshape[] = {3, 4};

    INA_TEST_ASSERT_SUCCEED(test_empty(data->ctx, dtype, ndim, shape, pshape));
}

INA_TEST_FIXTURE(constructor_empty, 4_f_p)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int8_t ndim = 4;
    int64_t shape[] = {10, 10, 10, 10};
    int64_t pshape[] = {0, 0, 0, 0};

    INA_TEST_ASSERT_SUCCEED(test_empty(data->ctx, dtype, ndim, shape, pshape));
}

INA_TEST_FIXTURE(constructor_empty, 5_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 5;
    int64_t shape[] = {10, 10, 10, 10, 10};
    int64_t pshape[] = {3, 4, 6, 3, 3};

    INA_TEST_ASSERT_SUCCEED(test_empty(data->ctx, dtype, ndim, shape, pshape));
}

INA_TEST_FIXTURE(constructor_empty, 7_f_p)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int8_t ndim = 7;
    int64_t shape[] = {10, 10, 10, 10, 10, 10, 10};
    int64_t pshape[] = {4, 3, 6, 2, 3, 3, 2};

    INA_TEST_ASSERT_SUCCEED(test_empty(data->ctx, dtype, ndim, shape, pshape));
}
