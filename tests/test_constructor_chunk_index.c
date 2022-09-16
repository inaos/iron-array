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

static ina_rc_t test_constructor_chunk_index(iarray_context_t *ctx,
                          iarray_data_type_t dtype,
                          int32_t type_size,
                          int8_t ndim,
                          const int64_t *shape,
                          const int64_t *cshape,
                          const int64_t *bshape,
                          bool contiguous,
                          char *urlpath)
{
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.dtype_size = type_size;
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

    iarray_container_t *src;
    blosc2_remove_urlpath(store.urlpath);

    INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &xdtshape, 0, 1, &store, &src));

    int64_t shape2[] = {30, 100};
    int64_t index[] = {0, 4, 8};
    int8_t nindex = 3;

    iarray_container_t *dest;
    INA_TEST_ASSERT_SUCCEED(iarray_from_chunk_index(ctx, src, shape2, index, nindex, &dest));

    iarray_container_free(ctx, &src);
    iarray_container_free(ctx, &dest);
    blosc2_remove_urlpath(store.urlpath);

    return INA_SUCCESS;
}

INA_TEST_DATA(constructor_chunk_index) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_chunk_index)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_chunk_index)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_chunk_index, 2_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {120, 100};
    int64_t cshape[] = {30, 40};
    int64_t bshape[] = {13, 14};

    INA_TEST_ASSERT_SUCCEED(test_constructor_chunk_index(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, false, NULL));
}

