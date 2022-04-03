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

#include <libiarray/iarray.h>
#include <src/iarray_private.h>

static ina_rc_t test_copy_transpose(iarray_context_t *ctx, iarray_data_type_t dtype, int8_t ndim,
                                    const int64_t *shape, const int64_t *cshape,
                                    const int64_t *bshape, double start, double stop,
                                    bool contiguous, char *xurlpath, char *yurlpath)
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

    iarray_storage_t store;
    store.urlpath = xurlpath;
    store.contiguous = contiguous;
    for (int i = 0; i < ndim; ++i) {
        store.chunkshape[i] = cshape[i];
        store.blockshape[i] = bshape[i];
    }

    iarray_storage_t ystore;
    ystore.urlpath = yurlpath;
    ystore.contiguous = contiguous;
    for (int i = 0; i < ndim; ++i) {
        ystore.chunkshape[i] = cshape[ndim - 1 - i];
        ystore.blockshape[i] = bshape[ndim - 1 - i];
    }

    double step = (stop - start) / size;

    iarray_container_t *c_x;
    iarray_container_t *c_trans;
    blosc2_remove_urlpath(store.urlpath);
    blosc2_remove_urlpath(ystore.urlpath);


    INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, start, step, &store, &c_trans));

    INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_trans, &c_x));


    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_copy(ctx, c_x, false, &ystore, &c_y));

    // Assert iterator reading it
    double tol;
    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        tol = 1e-14;
    } else {
        tol = 1e-6;
    }
    iarray_container_almost_equal(c_x, c_y, tol);

    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_trans);
    blosc2_remove_urlpath(store.urlpath);
    blosc2_remove_urlpath(ystore.urlpath);

    return INA_SUCCESS;
}

INA_TEST_DATA(constructor_copy_transpose) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_copy_transpose) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_BLOSCLZ;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_copy_transpose) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_copy_transpose, 2_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {10, 2500};
    int64_t cshape[] = {5, 1023};
    int64_t bshape[] = {2, 350};

    double start = -1000;
    double stop = 1;

    INA_TEST_ASSERT_SUCCEED(test_copy_transpose(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, true, NULL, NULL));
}

INA_TEST_FIXTURE(constructor_copy_transpose, 2_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {50, 25};
    int64_t cshape[] = {5, 10};
    int64_t bshape[] = {5, 10};

    double start = -5.3;
    double stop = 1.1245;

    INA_TEST_ASSERT_SUCCEED(test_copy_transpose(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, true, NULL, "arr2.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(constructor_copy_transpose, 2_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {3450, 2500};
    int64_t cshape[] = {5, 1000};
    int64_t bshape[] = {5, 1000};

    double start = -5.3;
    double stop = 1.1245;

    INA_TEST_ASSERT_SUCCEED(test_copy_transpose(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, true, NULL, "arr2.iarr"));
}
*/
