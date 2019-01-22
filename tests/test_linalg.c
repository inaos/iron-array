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
#include <iarray_private.h>

#include <tests/iarray_test.h>

static ina_rc_t test_gemm(iarray_context_t *ctx,
                          iarray_container_t *c_x,
                          iarray_container_t *c_y,
                          iarray_container_t *c_out,
                          uint64_t *bshape_a,
                          uint64_t *bshape_b,
                          iarray_container_t *c_res,
                          double tol)
{
    INA_TEST_ASSERT_SUCCEED(iarray_linalg_matmul(ctx, c_x, c_y, c_out, bshape_a, bshape_b, IARRAY_OPERATOR_GENERAL));
    if (iarray_container_almost_equal(c_out, c_res, tol) == INA_ERR_FAILED) {
        return INA_ERROR(INA_ERR_FAILED);
    }
    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_gemm(iarray_context_t *ctx,
                                     iarray_data_type_t dtype,
                                     size_t type_size,
                                     uint64_t *shape_x,
                                     uint64_t *pshape_x,
                                     uint64_t *shape_y,
                                     uint64_t *pshape_y,
                                     uint64_t *bshape_a,
                                     uint64_t *bshape_b)
{
    void *buffer_x;
    void *buffer_y;
    void *buffer_r;
    size_t buffer_x_len;
    size_t buffer_y_len;
    size_t buffer_r_len;
    double tol;

    buffer_x_len = type_size * shape_x[0] * shape_x[1];
    buffer_y_len = type_size * shape_y[0] * shape_y[1];
    buffer_r_len = type_size * shape_x[0] * shape_y[1];

    buffer_x = ina_mem_alloc(buffer_x_len);
    buffer_y = ina_mem_alloc(buffer_y_len);
    buffer_r = ina_mem_alloc(buffer_r_len);

    if (type_size == sizeof(float)) {
        tol = 1e-06;
        ffill_buf((float *) buffer_x, shape_x[0] * shape_x[1]);
        ffill_buf((float *) buffer_y, shape_y[0] * shape_y[1]);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int32_t) shape_x[0], (int32_t) shape_y[1],
                    (int32_t) shape_x[1], 1.0, (float *) buffer_x, (int32_t) shape_x[1], (float *) buffer_y,
                    (int32_t) shape_y[1], 0.0, (float *) buffer_r, (int32_t) shape_y[1]);
    } else {
        tol = 1e-14;
        dfill_buf((double *) buffer_x, shape_x[0] * shape_x[1]);
        dfill_buf((double *) buffer_y, shape_y[0] * shape_y[1]);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int32_t) shape_x[0], (int32_t) shape_y[1],
                    (int32_t) shape_x[1], 1.0, (double *) buffer_x, (int32_t) shape_x[1], (double *) buffer_y,
                    (int32_t) shape_y[1], 0.0, (double *) buffer_r, (int32_t) shape_y[1]);
    }

    iarray_dtshape_t xshape;
    iarray_dtshape_t yshape;
    iarray_dtshape_t oshape;
    iarray_dtshape_t rshape;

    xshape.dtype = dtype;
    xshape.ndim = 2;
    for (int i = 0; i < 2; ++i) {
         xshape.shape[i] = shape_x[i];
         xshape.pshape[i] = pshape_x[i];
    }


    yshape.dtype = dtype;
    yshape.ndim = 2;
    for (int i = 0; i < 2; ++i) {
         yshape.shape[i] = shape_y[i];
         yshape.pshape[i] = pshape_y[i];
    }

    oshape.dtype = dtype;
    oshape.ndim = 2;
    oshape.shape[0] = shape_x[0];
    oshape.shape[1] = shape_y[1];
    oshape.pshape[0] = (uint64_t) bshape_a[0];
    oshape.pshape[1] = (uint64_t) bshape_b[1];

    rshape.dtype = dtype;
    rshape.ndim = 2;
    rshape.shape[0] = shape_x[0];
    rshape.shape[1] = shape_y[1];
    rshape.pshape[0] = (uint64_t) bshape_a[0];
    rshape.pshape[1] = (uint64_t) bshape_b[1];

    iarray_container_t *c_x;
    iarray_container_t *c_y;
    iarray_container_t *c_out;
    iarray_container_t *c_res;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xshape, buffer_x, buffer_x_len, NULL, 0, &c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &yshape, buffer_y, buffer_y_len, NULL, 0, &c_y));
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &rshape, buffer_r, buffer_r_len, NULL, 0, &c_res));
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &oshape, NULL, 0, &c_out));

    INA_TEST_ASSERT_SUCCEED(test_gemm(ctx, c_x, c_y, c_out, bshape_a, bshape_b, c_res, tol));

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_res);

    ina_mem_free(buffer_x);
    ina_mem_free(buffer_y);
    ina_mem_free(buffer_r);

    return INA_SUCCESS;
}

static ina_rc_t test_gemv(iarray_context_t *ctx,
                          iarray_container_t *c_x,
                          iarray_container_t *c_y,
                          iarray_container_t *c_out,
                          iarray_container_t *c_res,
                          double tol)
{
    iarray_linalg_matmul(ctx, c_x, c_y, c_out, c_x->dtshape->pshape, c_x->dtshape->pshape, IARRAY_OPERATOR_GENERAL);
    if (iarray_container_almost_equal(c_out, c_res, tol) == INA_ERR_FAILED) {
        return INA_ERROR(INA_ERR_FAILED);
    }
    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_gemv(iarray_context_t *ctx, iarray_data_type_t dtype, size_t type_size, uint64_t M, uint64_t K, int32_t P)
{
    void *buffer_x;
    void *buffer_y;
    void *buffer_r;
    size_t buffer_x_len;
    size_t buffer_y_len;
    size_t buffer_r_len;
    double tol;

    buffer_x_len = type_size * M * K;
    buffer_y_len = type_size * K;
    buffer_r_len = type_size * M;
    buffer_x = ina_mem_alloc(buffer_x_len);
    buffer_y = ina_mem_alloc(buffer_y_len);
    buffer_r = ina_mem_alloc(buffer_r_len);

    if (type_size == sizeof(float)) {
        tol = 1e-06;
        ffill_buf((float*)buffer_x, M * K);
        ffill_buf((float*)buffer_y, K);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int32_t) M, (int32_t) K, 1.0, (float*)buffer_x, (int32_t) K, (float*)buffer_y, 1, 0.0, (float*)buffer_r, 1);
    }
    else {
        tol = 1e-14;
        dfill_buf((double*)buffer_x, M * K);
        dfill_buf((double*)buffer_y, K);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (int32_t) M, (int32_t) K, 1.0, (double*)buffer_x, (int32_t) K, (double*)buffer_y, 1, 0.0, (double*)buffer_r, 1);
    }

    iarray_dtshape_t xshape;
    iarray_dtshape_t yshape;
    iarray_dtshape_t oshape;
    iarray_dtshape_t rshape;

    xshape.dtype = dtype;
    xshape.ndim = 2;
    xshape.shape[0] = M;
    xshape.shape[1] = K;
    xshape.pshape[0] = (uint64_t) P;
    xshape.pshape[1] = (uint64_t) P;

    yshape.dtype = dtype;
    yshape.ndim = 1;
    yshape.shape[0] = K;
    yshape.pshape[0] = (uint64_t) P;

    oshape.dtype = dtype;
    oshape.ndim = 1;
    oshape.shape[0] = M;
    oshape.pshape[0] = (uint64_t) P;

    rshape.dtype = dtype;
    rshape.ndim = 1;
    rshape.shape[0] = M;
    rshape.pshape[0] = (uint64_t) P;

    iarray_container_t *c_x;
    iarray_container_t *c_y;
    iarray_container_t *c_out;
    iarray_container_t *c_res;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xshape, buffer_x, buffer_x_len, NULL, 0, &c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &yshape, buffer_y, buffer_y_len, NULL, 0, &c_y));
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &rshape, buffer_r, buffer_r_len, NULL, 0, &c_res));
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &oshape, NULL, 0, &c_out));

    INA_TEST_ASSERT_SUCCEED(test_gemv(ctx, c_x, c_y, c_out, c_res, tol));

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_res);

    ina_mem_free(buffer_x);
    ina_mem_free(buffer_y);
    ina_mem_free(buffer_r);

    return INA_SUCCESS;
}

INA_TEST_DATA(linalg_gemm) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(linalg_gemm) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(linalg_gemm) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE_SKIP(linalg_gemm, different_shapes) {

}

INA_TEST_FIXTURE_SKIP(linalg_gemm, test_partition_compatibility) {

}

INA_TEST_FIXTURE_SKIP(linalg_gemm, test_error_handling) {

}

INA_TEST_FIXTURE(linalg_gemm, double_data) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint64_t shape_x[] = {1230, 3456};
    uint64_t shape_y[] = {3456, 2856};
    uint64_t pshape_x[] = {80, 123};
    uint64_t pshape_y[] = {200, 97};

    uint64_t bshape_x[] = {300, 150};
    uint64_t bshape_y[] = {150, 300};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_gemm(data->ctx, dtype, type_size, shape_x, pshape_x,
                                                 shape_y, pshape_y, bshape_x, bshape_y));
}

INA_TEST_FIXTURE(linalg_gemm, float_data) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint64_t shape_x[] = {1243, 1256};
    uint64_t shape_y[] = {1256, 1234};

    uint64_t pshape_x[] = {124, 356};
    uint64_t pshape_y[] = {312, 265};

    uint64_t bshape_x[] = {1000, 1000};
    uint64_t bshape_y[] = {1000, 1000};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_gemm(data->ctx, dtype, type_size, shape_x, pshape_x,
                                                 shape_y, pshape_y, bshape_x, bshape_y));
}

/*

INA_TEST_DATA(linalg_gemv) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(linalg_gemv)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(linalg_gemv)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(linalg_gemv, double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint64_t M = 4163;
    uint64_t K = 5135;
    int32_t P = 453;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_gemv(data->ctx, dtype, type_size, M, K, P));
}

INA_TEST_FIXTURE(linalg_gemv, float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint64_t M = 3485;
    uint64_t K = 3555;
    int32_t P = 519;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_gemv(data->ctx, dtype, type_size, M, K, P));
}
*/