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

static ina_rc_t test_matmul_advice(iarray_context_t *ctx,
                                      iarray_data_type_t dtype,
                                      const int64_t *shape_a,
                                      const int64_t *shape_b,
                                      const int64_t *bshape_a,
                                      const int64_t *bshape_b)
{
    // We want to specify a [low, high] range explicitly, because L3 size is CPU-dependent
    int64_t low = 128 * 1024;
    int64_t high = 1024 * 1024;

    int ndim = 2;

    // Build array A
    iarray_dtshape_t dtshape_a;
    dtshape_a.dtype = dtype;
    dtshape_a.ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        dtshape_a.shape[i] = shape_a[i];
        dtshape_a.pshape[i] = 0;
    }
    INA_TEST_ASSERT_SUCCEED(iarray_partition_advice(ctx, &dtshape_a, low, high));
    iarray_container_t *c_a;
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &dtshape_a, NULL, 0, &c_a));

    // Build array B
    iarray_dtshape_t dtshape_b;
    dtshape_b.dtype = dtype;
    dtshape_b.ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        dtshape_b.shape[i] = shape_b[i];
        dtshape_b.pshape[i] = 0;
    }
    INA_TEST_ASSERT_SUCCEED(iarray_partition_advice(ctx, &dtshape_b, low, high));
    iarray_container_t *c_b;
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &dtshape_b, NULL, 0, &c_b));

    // Get the advice
    int64_t *_bshape_a;
    int64_t *_bshape_b;
    INA_TEST_ASSERT_SUCCEED(iarray_matmul_advice(ctx, c_a, c_b, &_bshape_a, &_bshape_b, low, high));

//    printf("pshape_a: ");
//    for (int i = 0; i < ndim; i++) {
//        printf("(real: %lld, expected: %lld), ", _bshape_a[i], bshape_a[i]);
//    }
//    printf("\n");
//
//    printf("pshape_b: ");
//    for (int i = 0; i < ndim; i++) {
//        printf("(real: %lld, expected: %lld), ", _bshape_b[i], bshape_b[i]);
//    }
//    printf("\n");

    for (int i = 0; i < ndim; i++) {
        INA_TEST_ASSERT_EQUAL_INT(_bshape_a[i], bshape_a[i]);
        INA_TEST_ASSERT_EQUAL_INT(_bshape_a[i], bshape_a[i]);
    }

    free(_bshape_a);
    free(_bshape_b);

    return INA_SUCCESS;

}

INA_TEST_DATA(matmul_advice) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(matmul_advice)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(matmul_advice)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(matmul_advice, squared)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int64_t shape_a[] = {1000, 1000};
    int64_t shape_b[] = {1000, 1000};
    int64_t bshape_a[] = {256, 256};
    int64_t bshape_b[] = {256, 512};

    INA_TEST_ASSERT_SUCCEED(test_matmul_advice(data->ctx, dtype, shape_a, shape_b, bshape_a, bshape_b));
}

INA_TEST_FIXTURE(matmul_advice, rect_symm)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int64_t shape_a[] = {100, 10000};
    int64_t shape_b[] = {10000, 100};
    int64_t bshape_a[] = {64, 2048};
    int64_t bshape_b[] = {2048, 64};

    INA_TEST_ASSERT_SUCCEED(test_matmul_advice(data->ctx, dtype, shape_a, shape_b, bshape_a, bshape_b));
}

INA_TEST_FIXTURE(matmul_advice, asymm1)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int64_t shape_a[] = {100, 10000};
    int64_t shape_b[] = {10000, 10};
    int64_t bshape_a[] = {64, 2048};
    int64_t bshape_b[] = {2048, 8};

    INA_TEST_ASSERT_SUCCEED(test_matmul_advice(data->ctx, dtype, shape_a, shape_b, bshape_a, bshape_b));
}

INA_TEST_FIXTURE(matmul_advice, asymm2)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int64_t shape_a[] = {2, 10000};
    int64_t shape_b[] = {10000, 3};
    int64_t bshape_a[] = {2, 8192};
    int64_t bshape_b[] = {8192, 2};

    INA_TEST_ASSERT_SUCCEED(test_matmul_advice(data->ctx, dtype, shape_a, shape_b, bshape_a, bshape_b));
}

INA_TEST_FIXTURE(matmul_advice, asymm3)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int64_t shape_a[] = {1, 10000};
    int64_t shape_b[] = {10000, 2};
    int64_t bshape_a[] = {1, 16384};
    int64_t bshape_b[] = {16384, 2};

    INA_TEST_ASSERT_SUCCEED(test_matmul_advice(data->ctx, dtype, shape_a, shape_b, bshape_a, bshape_b));
}
