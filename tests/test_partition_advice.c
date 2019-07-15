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
                                      const int32_t *pshape)
{
    int32_t _pshape[IARRAY_DIMENSION_MAX];
    INA_TEST_ASSERT_SUCCEED(iarray_partition_advice(ctx, dtype, ndim, shape, _pshape));

    for (int i = 0; i < ndim; i++) {
        printf("pshapes: %d, %d\n", pshape[i], _pshape[i]);
    }

    for (int i = 0; i < ndim; i++) {
        INA_TEST_ASSERT_EQUAL_INT(pshape[i], _pshape[i]);
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
    int32_t pshape[] = {128 * 1024};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->ctx, dtype, ndim, shape, pshape));
}

INA_TEST_FIXTURE(partition_advice, 1_d_1)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 1;
    int64_t shape[] = {1};
    int32_t pshape[] = {1};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->ctx, dtype, ndim, shape, pshape));
}

INA_TEST_FIXTURE(partition_advice, 2_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 2;
    int64_t shape[] = {15 * 1000, 1112 * 1000};
    int32_t pshape[] = {32, 4 * 1024};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->ctx, dtype, ndim, shape, pshape));
}

INA_TEST_FIXTURE(partition_advice, 3_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 3;
    int64_t shape[] = {17 * 1000, 3 * 1000, 300 * 1000};
    int32_t pshape[] = {32, 4, 1024};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->ctx, dtype, ndim, shape, pshape));
}

INA_TEST_FIXTURE(partition_advice, 4_d)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 4;
    int64_t shape[] = {17 * 1000, 3 * 1000, 30 * 1000, 10 * 1000};
    int32_t pshape[] = {32, 4, 32, 32};

    INA_TEST_ASSERT_SUCCEED(test_partition_advice(data->ctx, dtype, ndim, shape, pshape));
}
