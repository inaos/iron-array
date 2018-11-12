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

static ina_rc_t test_gemv(iarray_container_t *c_x, iarray_container_t *c_y, iarray_container_t *c_out, iarray_container_t *c_res)
{
    iarray_gemv(c_x, c_y, c_out);
    if (iarray_equal_data(c_out, c_res) != 0) {
        return INA_ERROR(INA_ERR_FAILED);
    }
    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_gemv(iarray_context_t *ctx,
    iarray_data_type_t dtype,
    int M,
    int K,
    int P,
    void *buffer_x,
    void *buffer_y,
    void *buffer_r,
    size_t buffer_x_len,
    size_t buffer_y_len,
    size_t buffer_r_len)
{
    iarray_dtshape_t xshape;
    iarray_dtshape_t yshape;
    iarray_dtshape_t oshape;
    iarray_dtshape_t rshape;

    xshape.dtype = dtype;
    xshape.ndim = 2;
    xshape.dims[0] = K;
    xshape.dims[1] = M;
    xshape.partshape[0] = P;
    xshape.partshape[1] = P;

    yshape.dtype = dtype;
    yshape.ndim = 1;
    yshape.dims[0] = K;
    yshape.partshape[0] = P;

    oshape.dtype = dtype;
    oshape.ndim = 1;
    oshape.dims[0] = M;
    oshape.partshape[0] = P;

    rshape.dtype = dtype;
    rshape.ndim = 1;
    rshape.dims[0] = M;
    rshape.partshape[1] = P;

    iarray_container_t *c_x;
    iarray_container_t *c_y;
    iarray_container_t *c_out;
    iarray_container_t *c_res;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xshape, buffer_x, buffer_x_len, IARRAY_STORAGE_ROW_WISE, NULL, 0, &c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &yshape, buffer_y, buffer_y_len, IARRAY_STORAGE_ROW_WISE, NULL, 0, &c_y));
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &rshape, buffer_r, buffer_r_len, IARRAY_STORAGE_ROW_WISE, NULL, 0, &c_res));
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &oshape, NULL, 0, &c_out));

    INA_TEST_ASSERT_SUCCEED(test_gemv(c_x, c_y, c_out, c_res));

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_res);

    return INA_SUCCESS;
}

INA_TEST_DATA(gemv) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(gemv)
{

    iarray_init();

    iarray_config_t cfg;
    ina_mem_set(&cfg, 0, sizeof(iarray_config_t));
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.max_num_threads = 1;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(gemv)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(gemv, double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    double *buffer_x;
    double *buffer_y;
    double *buffer_r;
    size_t buffer_x_len;
    size_t buffer_y_len;
    size_t buffer_r_len;

    int M = 163;
    int K = 135;
    int P = 24;
    
    buffer_x_len = sizeof(double) * M * K;
    buffer_y_len = sizeof(double) * K;
    buffer_r_len = sizeof(double) * M;
    buffer_x = ina_mem_alloc(buffer_x_len);
    buffer_y = ina_mem_alloc(buffer_y_len);
    buffer_r = ina_mem_alloc(buffer_r_len);
    dfill_buf(buffer_x, M * K);
    dfill_buf(buffer_y, K);
    dmv_mul(M, K, buffer_x, buffer_y, buffer_r);

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_gemv(data->ctx,
        dtype, M, K, P, buffer_x, buffer_y, buffer_r, buffer_x_len, buffer_y_len, buffer_r_len));

    ina_mem_free(buffer_x);
    ina_mem_free(buffer_y);
    ina_mem_free(buffer_r);
}

INA_TEST_FIXTURE(gemv, float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    float *buffer_x;
    float *buffer_y;
    float *buffer_r;
    size_t buffer_x_len;
    size_t buffer_y_len;
    size_t buffer_r_len;

    int M = 345;
    int K = 65;
    int P = 15;

    buffer_x_len = sizeof(float) * M * K;
    buffer_y_len = sizeof(float) * K;
    buffer_r_len = sizeof(float) * M;
    buffer_x = ina_mem_alloc(buffer_x_len);
    buffer_y = ina_mem_alloc(buffer_y_len);
    buffer_r = ina_mem_alloc(buffer_r_len);
    ffill_buf(buffer_x, M * K);
    ffill_buf(buffer_y, K);
    fmv_mul(M, K, buffer_x, buffer_y, buffer_r);

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_gemv(data->ctx,
        dtype, M, K, P, buffer_x, buffer_y, buffer_r, buffer_x_len, buffer_y_len, buffer_r_len));

    ina_mem_free(buffer_x);
    ina_mem_free(buffer_y);
    ina_mem_free(buffer_r);
}
