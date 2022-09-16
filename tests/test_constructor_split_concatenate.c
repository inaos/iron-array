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

static ina_rc_t test_split(iarray_context_t *ctx,
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

    INA_TEST_ASSERT_SUCCEED(iarray_ones(ctx, &xdtshape, &store, &src));

    iarray_container_t **dest = ina_mem_alloc(src->catarr->chunknitems * sizeof(iarray_container_t *));
    INA_TEST_ASSERT_SUCCEED(iarray_split(ctx, src, dest));
    iarray_container_t *src2;
    INA_TEST_ASSERT_SUCCEED(iarray_concatenate(ctx, dest, src->dtshape, src->storage, &src2));

    switch (src->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
        case IARRAY_DATA_TYPE_FLOAT:
            INA_TEST_ASSERT_SUCCEED(iarray_container_almost_equal(src, src2, 1e-8));
            break;
        default:
            INA_TEST_ASSERT_SUCCEED(iarray_container_equal(src, src2));
    }
    iarray_container_free(ctx, &src);
    INA_MEM_FREE_SAFE(dest);
    blosc2_remove_urlpath(store.urlpath);

    return INA_SUCCESS;
}

INA_TEST_DATA(constructor_split) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_split)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_split)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_split, 2_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {120, 100};
    int64_t cshape[] = {30, 40};
    int64_t bshape[] = {13, 14};

    INA_TEST_ASSERT_SUCCEED(test_split(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, false, NULL));
}


INA_TEST_FIXTURE(constructor_split, 5_f)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 5;
    int64_t shape[] = {10, 14, 12, 16, 10};
    int64_t cshape[] = {3, 4, 6, 8, 3};
    int64_t bshape[] = {2, 2, 2, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_split(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(constructor_split, 2_ll)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    int32_t type_size = sizeof(int64_t);

    int8_t ndim = 2;
    int64_t shape[] = {120, 100};
    int64_t cshape[] = {30, 40};
    int64_t bshape[] = {13, 14};

    INA_TEST_ASSERT_SUCCEED(test_split(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, false, NULL));
}

INA_TEST_FIXTURE(constructor_split, 5_i)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    int32_t type_size = sizeof(int32_t);

    int8_t ndim = 5;
    int64_t shape[] = {10, 14, 12, 16, 10};
    int64_t cshape[] = {3, 4, 6, 8, 3};
    int64_t bshape[] = {2, 2, 2, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_split(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(constructor_split, 2_s)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    int32_t type_size = sizeof(int16_t);

    int8_t ndim = 2;
    int64_t shape[] = {120, 100};
    int64_t cshape[] = {30, 40};
    int64_t bshape[] = {13, 14};

    INA_TEST_ASSERT_SUCCEED(test_split(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, false, NULL));
}


INA_TEST_FIXTURE(constructor_split, 5_sc)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    int32_t type_size = sizeof(int8_t);

    int8_t ndim = 5;
    int64_t shape[] = {10, 14, 12, 16, 10};
    int64_t cshape[] = {3, 4, 6, 8, 3};
    int64_t bshape[] = {2, 2, 2, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_split(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(constructor_split, 2_ull)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    int32_t type_size = sizeof(uint64_t);

    int8_t ndim = 2;
    int64_t shape[] = {120, 100};
    int64_t cshape[] = {30, 40};
    int64_t bshape[] = {13, 14};

    INA_TEST_ASSERT_SUCCEED(test_split(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, false, NULL));
}


INA_TEST_FIXTURE(constructor_split, 5_ui)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    int32_t type_size = sizeof(uint32_t);

    int8_t ndim = 5;
    int64_t shape[] = {10, 14, 12, 16, 10};
    int64_t cshape[] = {3, 4, 6, 8, 3};
    int64_t bshape[] = {2, 2, 2, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_split(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(constructor_split, 2_us)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;
    int32_t type_size = sizeof(uint16_t);

    int8_t ndim = 2;
    int64_t shape[] = {120, 100};
    int64_t cshape[] = {30, 40};
    int64_t bshape[] = {13, 14};

    INA_TEST_ASSERT_SUCCEED(test_split(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, false, NULL));
}


INA_TEST_FIXTURE(constructor_split, 5_uc)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    int32_t type_size = sizeof(uint8_t);

    int8_t ndim = 5;
    int64_t shape[] = {10, 14, 12, 16, 10};
    int64_t cshape[] = {3, 4, 6, 8, 3};
    int64_t bshape[] = {2, 2, 2, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_split(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(constructor_split, 3_b)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    int32_t type_size = sizeof(bool);

    int8_t ndim = 3;
    int64_t shape[] = {10, 14, 12};
    int64_t cshape[] = {3, 4, 6};
    int64_t bshape[] = {2, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_split(data->ctx, dtype, type_size, ndim, shape, cshape, bshape, true, NULL));
}

