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
#include "iarray_private.h"

static ina_rc_t test_partition_advice(iarray_config_t cfg,
                                      iarray_data_type_t dtype,
                                      int8_t ndim,
                                      const int64_t *shape,
                                      const int64_t *cshape,
                                      const int64_t *bshape,
                                      bool contiguous, char *urlpath)
{
    iarray_context_t *ctx;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &ctx));
    INA_TEST_ASSERT_SUCCEED(ctx->cfg->compression_favor <= IARRAY_COMPRESSION_FAVOR_CRATIO);
    iarray_dtshape_t dtshape;
    dtshape.dtype = dtype;
    int64_t max_chunksize = 0;
    int64_t max_blocksize = 0;
    dtshape.ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        dtshape.shape[i] = shape[i];
    }

    if (cshape[0] > 0) {
        // We want to specify max for chunskize, blocksize explicitly, because L2/L3 size is CPU-dependent
        max_chunksize = 1024 * 1024;
        max_blocksize = 64 * 1024;
    }

    iarray_storage_t storage = {0};
    storage.contiguous = contiguous;
    storage.urlpath = urlpath;
    blosc2_remove_urlpath(storage.urlpath);
    INA_TEST_ASSERT_SUCCEED(iarray_partition_advice(ctx, &dtshape, &storage, 0, max_chunksize, 0, max_blocksize));

    if (max_chunksize > 0) {
        for (int i = 0; i < ndim; i++) {
            INA_TEST_ASSERT_EQUAL_INT64(cshape[i], storage.chunkshape[i]);
            INA_TEST_ASSERT_EQUAL_INT64(bshape[i], storage.blockshape[i]);
        }
    }
    else {
        for (int i = 0; i < ndim; i++) {
            INA_TEST_ASSERT(storage.chunkshape[i] > 0);
            INA_TEST_ASSERT(storage.blockshape[i] > 0);
            // In automatic mode, we can say at least that chunkshape > blockshape
            // (at least for the range of values tested here, and for decent values of L3)
            INA_TEST_ASSERT(storage.chunkshape[i] > storage.blockshape[i]);
        }
    }

    iarray_context_free(&ctx);
    blosc2_remove_urlpath(storage.urlpath);

    return INA_SUCCESS;

}


INA_TEST_DATA(partition_advice) {
    iarray_config_t cfg;
};

INA_TEST_SETUP(partition_advice)
{
    iarray_init();

    data->cfg = IARRAY_CONFIG_DEFAULTS;
}

INA_TEST_TEARDOWN(partition_advice)
{
    INA_UNUSED(data);
    iarray_destroy();
}

INA_TEST_FIXTURE(partition_advice, 1_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 1;
    int64_t shape[] = {1000 * 1000};
    int64_t cshape[] = {128 * 1024};
    int64_t bshape[] = {8 * 1024};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_BALANCE;
    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->cfg, dtype, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(partition_advice, 1_d_1)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 1;
    int64_t shape[] = {1};
    int64_t cshape[] = {1};
    int64_t bshape[] = {1};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_SPEED;
    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->cfg, dtype, ndim, shape, cshape, bshape, false, "arr.iarr"));
}

INA_TEST_FIXTURE(partition_advice, 2_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 2;
    int64_t shape[] = {15 * 1000, 1112 * 1000};
    int64_t cshape[] = {32, 4 * 1024};
    int64_t bshape[] = {8, 1024};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_CRATIO;
    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->cfg, dtype, ndim, shape, cshape, bshape, true, "arr.iarr"));
}

INA_TEST_FIXTURE(partition_advice, 2_d_automatic)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 2;
    int64_t shape[] = {15 * 1000, 1112 * 1000};
    int64_t cshape[] = {0, 0};
    int64_t bshape[] = {0, 0};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->cfg, dtype, ndim, shape, cshape, bshape, false, NULL));
}

INA_TEST_FIXTURE(partition_advice, 2_d_near_bounds)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 2;
    int64_t shape[] = {513, 257};
    int64_t cshape[] = {256, 256};
    int64_t bshape[] = {64, 128};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->cfg, dtype, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(partition_advice, 3_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 3;
    int64_t shape[] = {17 * 1000, 3 * 1000, 300 * 1000};
    int64_t cshape[] = {32, 4, 1024};
    int64_t bshape[] = {8, 2, 512};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->cfg, dtype, ndim, shape, cshape, bshape, true, NULL));
}

INA_TEST_FIXTURE(partition_advice, 4_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 4;
    int64_t shape[] = {17 * 1000, 3 * 1000, 30 * 1000, 10 * 1000};
    int64_t cshape[] = {32, 4, 32, 32};
    int64_t bshape[] = {16, 2, 16, 16};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_SPEED;
    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->cfg, dtype, ndim, shape, cshape, bshape, false, "arr.iarr"));
}

INA_TEST_FIXTURE(partition_advice, 4_d_automatic)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 4;
    int64_t shape[] = {17 * 1000, 3 * 1000, 30 * 1000, 10 * 1000};
    int64_t cshape[] = {0, 0, 0, 0};
    int64_t bshape[] = {0, 0, 0, 0};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_CRATIO;
    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->cfg, dtype, ndim, shape, cshape, bshape, false, NULL));
}
