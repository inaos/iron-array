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

static ina_rc_t _test_operator_add(iarray_context_t *ctx,
                                   iarray_container_t *c_x,
                                   iarray_container_t * c_y,
                                   iarray_container_t * c_out,
                                   iarray_container_t * c_res,
                                   double tol)
{
    INA_TEST_ASSERT_SUCCEED(iarray_operator_add(ctx, c_x, c_y, c_out));
    return iarray_container_almost_equal(c_out, c_res, tol);
}

static ina_rc_t _execute_iarray_operator_add(iarray_context_t *ctx,
                                             iarray_data_type_t dtype,
                                             size_t type_size,
                                             uint64_t n,
                                             int32_t p)
{
    void *buffer_x;
    void *buffer_y;
    void *buffer_r;
    size_t buffer_x_len;
    size_t buffer_y_len;
    size_t buffer_r_len;
    double tol;

    buffer_x_len = type_size * n * n;
    buffer_y_len = type_size * n * n;
    buffer_r_len = type_size * n * n;
    buffer_x = ina_mem_alloc(buffer_x_len);
    buffer_y = ina_mem_alloc(buffer_y_len);
    buffer_r = ina_mem_alloc(buffer_r_len);

    if (type_size == sizeof(float)) {
        tol = 1e-06;
        ffill_buf((float*)buffer_x, n*n);
        ffill_buf((float*)buffer_y, n*n);
        vsAdd((const int)n*n, buffer_x, buffer_y, buffer_r);
    }
    else {
        tol = 1e-14;
        dfill_buf((double*)buffer_x, n*n);
        dfill_buf((double*)buffer_y, n*n);
        vdAdd((const int)n*n, buffer_x, buffer_y, buffer_r);
    }

    iarray_dtshape_t shape;

    shape.dtype = dtype;
    shape.ndim = 2;
    shape.shape[0] = n;
    shape.shape[1] = n;
    shape.partshape[0] = (uint64_t)p;
    shape.partshape[1] = (uint64_t)p;

    iarray_container_t *c_x;
    iarray_container_t *c_y;
    iarray_container_t *c_out;
    iarray_container_t *c_res;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &shape, buffer_x, buffer_x_len, NULL, 0, &c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &shape, buffer_y, buffer_y_len, NULL, 0, &c_y));
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &shape, buffer_r, buffer_r_len, NULL, 0, &c_res));
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &shape, NULL, 0, &c_out));

    INA_TEST_ASSERT_SUCCEED(_test_operator_add(ctx, c_x, c_y, c_out, c_res, tol));

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_res);

    ina_mem_free(buffer_x);
    ina_mem_free(buffer_y);
    ina_mem_free(buffer_r);

    return INA_SUCCESS;
}

INA_TEST_DATA(operator_add) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(operator_add)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(operator_add)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(operator_add, float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    uint64_t N = 387;
    int32_t P = 44;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_add(data->ctx, dtype, type_size, N, P));
}

INA_TEST_FIXTURE(operator_add, double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint64_t N = 298;
    int32_t P = 22;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_add(data->ctx, dtype, type_size, N, P));
}
