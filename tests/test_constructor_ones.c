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

#include <src/iarray_private.h>
#include <libiarray/iarray.h>

static ina_rc_t test_ones(iarray_context_t *ctx,
                          iarray_data_type_t dtype,
                          size_t type_size,
                          int8_t ndim,
                          const int64_t *shape,
                          const int64_t *cshape,
                          const int64_t *bshape,
                          bool contiguous,
                          char *urlpath)
{
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
    }

    iarray_storage_t store;
    store.contiguous = contiguous;
    store.urlpath = urlpath;
    for (int i = 0; i < ndim; ++i) {
        store.chunkshape[i] = cshape[i];
        store.blockshape[i] = bshape[i];
    }

    int64_t buf_size = 1;
    for (int j = 0; j < ndim; ++j) {
        buf_size *= shape[j];
    }

    uint8_t *buf_dest = malloc((size_t)buf_size * type_size);

    iarray_container_t *c_x;
    blosc2_remove_urlpath(store.urlpath);

    INA_TEST_ASSERT_SUCCEED(iarray_ones(ctx, &xdtshape, &store, 0, &c_x));

    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buf_dest, (size_t)buf_size * type_size));

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        double *buff = (double *) buf_dest;
        for (int64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], 1);
        }
    } else {
        float *buff = (float *) buf_dest;
        for (int64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], 1);
        }
    }

    iarray_container_free(ctx, &c_x);
    free(buf_dest);
    blosc2_remove_urlpath(store.urlpath);

    return INA_SUCCESS;
}

INA_TEST_DATA(constructor_ones) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_ones)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_ones)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_ones, 2_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {120, 100};
    int64_t cshape[] = {30, 40};
    int64_t bshape[] = {13, 14};

    INA_TEST_ASSERT_SUCCEED(test_ones(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, false, NULL));
}

INA_TEST_FIXTURE(constructor_ones, 5_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 5;
    int64_t shape[] = {10, 14, 12, 16, 10};
    int64_t cshape[] = {3, 4, 6, 8, 3};
    int64_t bshape[] = {2, 2, 2, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_ones(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}
