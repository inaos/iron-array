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

static ina_rc_t test_cfg(iarray_context_t *ctx,
                         iarray_data_type_t dtype,
                         int8_t ndim,
                         const int64_t *shape,
                         const int64_t *cshape,
                         const int64_t *bshape, bool contiguous, char *urlpath)
{
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
    }

    iarray_storage_t xstore;
    xstore.backend = (cshape == NULL) ? IARRAY_STORAGE_PLAINBUFFER : IARRAY_STORAGE_BLOSC;
    xstore.contiguous = contiguous;
    xstore.urlpath = urlpath;
    for (int i = 0; i < ndim; ++i) {
        if (cshape != NULL) {
            xstore.chunkshape[i] = cshape[i];
            xstore.blockshape[i] = bshape[i];
        }
    }

    // Empty array
    iarray_container_t *c_x;
    blosc2_remove_urlpath(xstore.urlpath);
    INA_TEST_ASSERT_SUCCEED(iarray_empty(ctx, &xdtshape, &xstore, 0, &c_x));


    // Non-empty array
    iarray_container_t *z_x;

    if (xstore.urlpath != NULL) {
        xstore.urlpath = "zarr.iarr";
        blosc2_remove_urlpath(xstore.urlpath);
    }
    INA_TEST_ASSERT_SUCCEED(iarray_zeros(ctx, &xdtshape, &xstore, 0, &z_x));

    if (iarray_is_empty(z_x)) {
        return INA_ERROR(INA_ERR_ERROR);
    }

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &z_x);
    blosc2_remove_urlpath(urlpath);
    blosc2_remove_urlpath(xstore.urlpath);
    return INA_SUCCESS;

}

INA_TEST_DATA(constructor_cfg) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_cfg)
{
    INA_UNUSED(data);
    iarray_init();
}

INA_TEST_TEARDOWN(constructor_cfg)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_cfg, 1_d)
{
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.filter_flags = IARRAY_COMP_DELTA;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 1;
    int64_t shape[] = {1045};
    int64_t cshape[] = {341};
    int64_t bshape[] = {77};

    INA_TEST_ASSERT_SUCCEED(test_cfg(data->ctx, dtype, ndim, shape, cshape, bshape, false, NULL));
}

INA_TEST_FIXTURE(constructor_cfg, 1_d_1)
{
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.filter_flags = IARRAY_COMP_BITSHUFFLE;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 1;
    int64_t shape[] = {128};
    int64_t cshape[] = {128};
    int64_t bshape[] = {128};

    INA_TEST_ASSERT_SUCCEED(test_cfg(data->ctx, dtype, ndim, shape, cshape, bshape, false, "arr.iarr"));
}

INA_TEST_FIXTURE(constructor_cfg, 2_d)
{
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.filter_flags = IARRAY_COMP_SHUFFLE;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 2;
    int64_t shape[] = {15, 1112};
    int64_t cshape[] = {4, 231};
    int64_t bshape[] = {2, 53};

    INA_TEST_ASSERT_SUCCEED(test_cfg(data->ctx, dtype, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(constructor_cfg, 4_f_p)
{
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.filter_flags = IARRAY_COMP_TRUNC_PREC;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int8_t ndim = 4;
    int64_t shape[] = {10, 5, 6, 10};
    int64_t *cshape = NULL;
    int64_t *bshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_cfg(data->ctx, dtype, ndim, shape, cshape, bshape, true, "arr.iarr"));
}

INA_TEST_FIXTURE(constructor_cfg, 5_d)
{
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 5;
    int64_t shape[] = {11, 12, 8, 5, 3};
    int64_t cshape[] = {11, 4, 6, 5, 3};
    int64_t bshape[] = {5, 2, 2, 5, 3};

    INA_TEST_ASSERT_SUCCEED(test_cfg(data->ctx, dtype, ndim, shape, cshape, bshape, false, NULL));
}

INA_TEST_FIXTURE(constructor_cfg, 7_f_p)
{
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int8_t ndim = 7;
    int64_t shape[] = {10, 6, 6, 4, 12, 7, 10};
    int64_t *cshape = NULL;
    int64_t * bshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_cfg(data->ctx, dtype, ndim, shape, cshape, bshape, false, "arr.iarr"));
}
