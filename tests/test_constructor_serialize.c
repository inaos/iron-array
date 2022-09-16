/*
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <src/iarray_private.h>
#include <libiarray/iarray.h>


static ina_rc_t test_serialize(iarray_context_t *ctx, iarray_data_type_t dtype, int8_t ndim,
                              const int64_t *shape, const int64_t *cshape, const int64_t *bshape, double start,
                              double stop, bool contiguous, char *urlpath) {
    // Create dtshape
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

    iarray_container_t *c_x;
    blosc2_remove_urlpath(store.urlpath);

    INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &xdtshape, start, stop, &store, &c_x));

    uint8_t *cframe;
    int64_t cframe_len;
    bool needs_free;

    INA_TEST_ASSERT_SUCCEED(iarray_to_cframe(ctx, c_x, &cframe, &cframe_len, &needs_free));

    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_from_cframe(ctx, cframe, cframe_len, (bool) (rand() % 2), &c_y));

    switch (dtype) {
        case IARRAY_DATA_TYPE_FLOAT:
            INA_TEST_ASSERT_SUCCEED(iarray_container_almost_equal(c_x, c_y, 1e-4));
            break;
        case IARRAY_DATA_TYPE_DOUBLE:
            INA_TEST_ASSERT_SUCCEED(iarray_container_almost_equal(c_x, c_y, 1e-12));
            break;
        default:
            INA_TEST_ASSERT_SUCCEED(iarray_container_equal(c_x, c_y));
    }

    if (needs_free) {
        free(cframe);
    }

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    blosc2_remove_urlpath(store.urlpath);
    return INA_SUCCESS;
}

INA_TEST_DATA(constructor_serialize) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_serialize) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_serialize) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_serialize, 2_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {223, 456};
    int64_t cshape[] = {31, 323};
    int64_t bshape[] = {10, 10};
    double start = -0.1;
    double stop = -0.25;

    INA_TEST_ASSERT_SUCCEED(test_serialize(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, false, NULL));
}

INA_TEST_FIXTURE(constructor_serialize, 3_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {22, 202, 12};
    int64_t cshape[] = {11, 40, 5};
    int64_t bshape[] = {4, 10, 5};
    double start = -1;
    double stop = -2;

    INA_TEST_ASSERT_SUCCEED(test_serialize(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, false, "serialize.iarr"));
}

INA_TEST_FIXTURE(constructor_serialize, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {5, 7, 8};
    int64_t cshape[] = {3, 5, 3};
    int64_t bshape[] = {2, 2, 2};
    double start = 10.;
    double stop = 0.;

    INA_TEST_ASSERT_SUCCEED(test_serialize(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, false, NULL));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(constructor_serialize, 7_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 7;
    int64_t shape[] = {5, 7, 8, 9, 6, 5, 7};
    int64_t cshape[] = {3, 5, 3, 3, 3, 2, 3};
    int64_t bshape[] = {2, 2, 2, 2, 2, 2, 2};
    double start = 10;
    double stop = 0;

    INA_TEST_ASSERT_SUCCEED(test_serialize(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, true, "arr.iarr"));
}
*/
