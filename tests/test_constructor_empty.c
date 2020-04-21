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
        if (pshape)
            xdtshape.pshape[i] = pshape[i];
    }

    iarray_store_properties_t store;
    store.backend = pshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    store.enforce_frame = false;
    store.filename = NULL;

    // Empty array
    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &xdtshape, &store, 0, &c_x));

    if (!iarray_is_empty(c_x)) {
        return INA_ERROR(INA_ERR_ERROR);
    }

    // Non-empty array
    iarray_container_t *z_x;
    INA_TEST_ASSERT_SUCCEED(iarray_zeros(ctx, &xdtshape, &store, 0, &z_x));

    if (iarray_is_empty(z_x)) {
        return INA_ERROR(INA_ERR_ERROR);
    }

    int64_t nbytes;
    int64_t cbytes;
    INA_TEST_ASSERT_SUCCEED(iarray_container_info(z_x, &nbytes, &cbytes));
    INA_TEST_ASSERT_SUCCEED(cbytes <= nbytes);

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &z_x);

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
    int64_t shape[] = {15, 1112};
    int64_t pshape[] = {3, 4};

    INA_TEST_ASSERT_SUCCEED(test_empty(data->ctx, dtype, ndim, shape, pshape));
}

INA_TEST_FIXTURE(constructor_empty, 4_f_p)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int8_t ndim = 4;
    int64_t shape[] = {10, 5, 6, 10};
    int64_t *pshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_empty(data->ctx, dtype, ndim, shape, pshape));
}

INA_TEST_FIXTURE(constructor_empty, 5_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 5;
    int64_t shape[] = {11, 12, 8, 5, 3};
    int64_t pshape[] = {3, 4, 2, 4, 3};

    INA_TEST_ASSERT_SUCCEED(test_empty(data->ctx, dtype, ndim, shape, pshape));
}

INA_TEST_FIXTURE(constructor_empty, 7_f_p)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int8_t ndim = 7;
    int64_t shape[] = {10, 6, 6, 4, 12, 7, 10};
    int64_t *pshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_empty(data->ctx, dtype, ndim, shape, pshape));
}
