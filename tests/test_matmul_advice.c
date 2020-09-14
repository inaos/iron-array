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
#include <src/iarray_private.h>


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

    int8_t ndim = 2;

    // Build array A
    iarray_dtshape_t dtshape_a;
    dtshape_a.dtype = dtype;
    dtshape_a.ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        dtshape_a.shape[i] = shape_a[i];
    }

    iarray_storage_t store;
    store.backend = IARRAY_STORAGE_BLOSC;
    store.filename = NULL;
    store.enforce_frame = false;

    INA_TEST_ASSERT_SUCCEED(iarray_partition_advice(ctx, &dtshape_a, &store, low, high));
    iarray_container_t *c_a;
    INA_TEST_ASSERT_SUCCEED(iarray_ones(ctx, &dtshape_a, &store, 0, &c_a));

    // Build array B
    iarray_dtshape_t dtshape_b;
    dtshape_b.dtype = dtype;
    dtshape_b.ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        dtshape_b.shape[i] = shape_b[i];
    }
    INA_TEST_ASSERT_SUCCEED(iarray_partition_advice(ctx, &dtshape_b, &store, low, high));
    iarray_container_t *c_b;
    INA_TEST_ASSERT_SUCCEED(iarray_ones(ctx, &dtshape_b, &store, 0, &c_b));

    // Build array C
    iarray_dtshape_t dtshape_c;
    dtshape_c.dtype = dtype;
    dtshape_c.ndim = ndim;
    dtshape_c.shape[0] = shape_a[0];
    dtshape_c.shape[1] = shape_b[1];
    INA_TEST_ASSERT_SUCCEED(iarray_partition_advice(ctx, &dtshape_c, &store, low, high));
    iarray_container_t *c_c;
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &dtshape_c, &store, 0, &c_c));

//    printf("cshape_a: (%lld, %lld)\n", c_a->dtshape->chunkshape[0], c_a->dtshape->chunkshape[1]);
//    printf("cshape_b: (%lld, %lld)\n", c_b->dtshape->chunkshape[0], c_b->dtshape->chunkshape[1]);
//    printf("cshape_c: (%lld, %lld)\n", c_c->dtshape->chunkshape[0], c_c->dtshape->chunkshape[1]);

    // Get the advice for matmul itself
    int64_t _bshape_a[2];
    int64_t _bshape_b[2];
    INA_TEST_ASSERT_SUCCEED(iarray_matmul_advice(ctx, c_a, c_b, c_c, _bshape_a, _bshape_b, low, high));

//    printf("bshape_a: ");
//    for (int i = 0; i < ndim; i++) {
//        printf("(real: %lld, expected: %lld), ", _bshape_a[i], bshape_a[i]);
//    }
//    printf("\n");
//
//    printf("bshape_b: ");
//    for (int i = 0; i < ndim; i++) {
//        printf("(real: %lld, expected: %lld), ", _bshape_b[i], bshape_b[i]);
//    }
//    printf("\n");

    for (int i = 0; i < ndim; i++) {
        INA_TEST_ASSERT_EQUAL_INT64(_bshape_a[i], bshape_a[i]);
        INA_TEST_ASSERT_EQUAL_INT64(_bshape_b[i], bshape_b[i]);
    }

    if (INA_FAILED(iarray_linalg_matmul(ctx, c_a, c_b ,c_c, _bshape_a, _bshape_b, IARRAY_OPERATOR_GENERAL))) {
        printf("Error in linalg_matmul: %s\n", ina_err_strerror(ina_err_get_rc()));
        exit(1);
    }

    int64_t size_c = dtshape_c.shape[0] * dtshape_c.shape[1];
    double *buffer_c = (double *) malloc(size_c * sizeof(double));
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_c, buffer_c, size_c * sizeof(double)));

    double mult_value = (double) dtshape_a.shape[1];
    for (int i = 0; i < size_c; ++i) {
        if (fabs((buffer_c[i] - mult_value) / buffer_c[i]) > 1e-8) {
            printf("%f - %f = %f\n", buffer_c[i], mult_value, buffer_c[i] - mult_value);
            printf("Error in element %d\n", i);
            return INA_ERROR(INA_ERR_ERROR);
        }
    }

    free(buffer_c);
    iarray_container_free(ctx, &c_a);
    iarray_container_free(ctx, &c_b);
    iarray_container_free(ctx, &c_c);

    return INA_SUCCESS;
}

INA_TEST_DATA(matmul_advice) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(matmul_advice)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.max_num_threads = 1;
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
    int64_t bshape_a[] = {1, 10000};
    int64_t bshape_b[] = {10000, 2};

    INA_TEST_ASSERT_SUCCEED(test_matmul_advice(data->ctx, dtype, shape_a, shape_b, bshape_a, bshape_b));
}

INA_TEST_FIXTURE(matmul_advice, matvec)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int64_t shape_a[] = {10, 1000};
    int64_t shape_b[] = {1000, 1};
    int64_t bshape_a[] = {8, 1000};
    int64_t bshape_b[] = {1000, 1};

    INA_TEST_ASSERT_SUCCEED(test_matmul_advice(data->ctx, dtype, shape_a, shape_b, bshape_a, bshape_b));
}
