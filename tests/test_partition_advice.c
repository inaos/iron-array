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

static ina_rc_t test_partition_advice(iarray_context_t *ctx,
                                      iarray_data_type_t dtype,
                                      int8_t ndim,
                                      const int64_t *shape,
                                      const int64_t *cshape)
{
    iarray_dtshape_t dtshape;
    dtshape.dtype = dtype;
    dtshape.ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        dtshape.shape[i] = shape[i];
    }
    // We want to specify a [low, high] range explicitly, because L3 size is CPU-dependent
    int64_t low = 128 * 1024;
    int64_t high = 1024 * 1024;

    iarray_storage_t storage = {0};
    storage.backend = IARRAY_STORAGE_BLOSC;
    storage.enforce_frame = false;
    storage.filename = NULL;
    INA_TEST_ASSERT_SUCCEED(iarray_chunk_advice(ctx, &dtshape, &storage, low, high));

    for (int i = 0; i < ndim; i++) {
        INA_TEST_ASSERT_EQUAL_INT64(cshape[i], storage.chunkshape[i]);
        INA_TEST_ASSERT_EQUAL_INT64(cshape[i], storage.blockshape[i]);
    }

    return INA_SUCCESS;

}

INA_TEST_DATA(partition_advice) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(partition_advice)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(partition_advice)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(partition_advice, 1_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 1;
    int64_t shape[] = {1000 * 1000};
    int64_t cshape[] = {128 * 1024};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->ctx, dtype, ndim, shape, cshape));
}

INA_TEST_FIXTURE(partition_advice, 1_d_1)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 1;
    int64_t shape[] = {1};
    int64_t cshape[] = {1};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->ctx, dtype, ndim, shape, cshape));
}

INA_TEST_FIXTURE(partition_advice, 2_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 2;
    int64_t shape[] = {15 * 1000, 1112 * 1000};
    int64_t cshape[] = {32, 4 * 1024};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->ctx, dtype, ndim, shape, cshape));
}

INA_TEST_FIXTURE(partition_advice, 2_d_near_bounds)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 2;
    int64_t shape[] = {513, 257};
    int64_t cshape[] = {256, 128};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->ctx, dtype, ndim, shape, cshape));
}

INA_TEST_FIXTURE(partition_advice, 3_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 3;
    int64_t shape[] = {17 * 1000, 3 * 1000, 300 * 1000};
    int64_t cshape[] = {32, 4, 1024};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->ctx, dtype, ndim, shape, cshape));
}

INA_TEST_FIXTURE(partition_advice, 4_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 4;
    int64_t shape[] = {17 * 1000, 3 * 1000, 30 * 1000, 10 * 1000};
    int64_t cshape[] = {32, 4, 32, 32};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->ctx, dtype, ndim, shape, cshape));
}
