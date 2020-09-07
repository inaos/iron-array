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

typedef ina_rc_t(*_test_operator_elwise_x)(iarray_context_t *ctx,
                                           iarray_container_t *x,
                                           iarray_container_t *o);

typedef ina_rc_t(*_test_operator_elwise_xy)(iarray_context_t *ctx,
                                            iarray_container_t *x,
                                            iarray_container_t *y,
                                            iarray_container_t *o);

static ina_rc_t _test_operator_x(iarray_context_t *ctx,
                                 iarray_container_t *c_x,
                                 iarray_container_t * c_out,
                                 iarray_container_t * c_res,
                                 _test_operator_elwise_x test_fun,
                                 double tol)
{
    INA_TEST_ASSERT_SUCCEED(test_fun(ctx, c_x, c_out));
    return iarray_container_almost_equal(c_out, c_res, tol);
}

static ina_rc_t _test_operator_xy(iarray_context_t *ctx,
                                  iarray_container_t *c_x,
                                  iarray_container_t * c_y,
                                  iarray_container_t * c_out,
                                  iarray_container_t * c_res,
                                  _test_operator_elwise_xy test_fun,
                                  double tol)
{
    INA_TEST_ASSERT_SUCCEED(test_fun(ctx, c_x, c_y, c_out));
    return iarray_container_almost_equal(c_out, c_res, tol);
}

static ina_rc_t _execute_iarray_operator_x(iarray_context_t *ctx,
                                           _test_operator_elwise_x test_fun,
                                           _iarray_vml_fun_d_a vml_fun_d,
                                           _iarray_vml_fun_s_a vml_fun_s,
                                           iarray_data_type_t dtype,
                                           int32_t type_size,
                                           int64_t n,
                                           int64_t csize,
                                           int64_t bsize)
{
    void *buffer_x;
    void *buffer_r;
    size_t buffer_x_len;
    size_t buffer_r_len;
    double tol;

    buffer_x_len = (size_t)(type_size * n * n);
    buffer_r_len = (size_t)(type_size * n * n);
    buffer_x = ina_mem_alloc(buffer_x_len);
    buffer_r = ina_mem_alloc(buffer_r_len);

    if (type_size == sizeof(float)) {
        tol = 1e-06;
        ffill_buf((float*)buffer_x, (size_t)(n * n));
        vml_fun_s((const int)(n * n), buffer_x, buffer_r);
    }
    else {
        tol = 1e-14;
        dfill_buf((double*)buffer_x, (size_t)(n * n));
        vml_fun_d((const int)(n * n), buffer_x, buffer_r);
    }

    iarray_dtshape_t shape;

    shape.dtype = dtype;
    shape.ndim = 2;
    shape.shape[0] = (int64_t)n;
    shape.shape[1] = (int64_t)n;

    iarray_storage_t store;
    store.backend = IARRAY_STORAGE_BLOSC;
    store.filename = NULL;
    store.enforce_frame = false;
    for (int i = 0; i < shape.ndim; ++i) {
        store.chunkshape[i] = csize;
        store.blockshape[i] = bsize;
    }
    
    iarray_container_t *c_x;
    iarray_container_t *c_out;
    iarray_container_t *c_res;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &shape, buffer_x, buffer_x_len, &store, 0, &c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &shape, buffer_r, buffer_r_len, &store, 0, &c_res));
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &shape, &store, 0, &c_out));

    INA_TEST_ASSERT_SUCCEED(_test_operator_x(ctx, c_x, c_out, c_res, test_fun, tol));

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_res);

    ina_mem_free(buffer_x);
    ina_mem_free(buffer_r);

    return INA_SUCCESS;
}

static ina_rc_t _execute_iarray_operator_xy(iarray_context_t *ctx,
                                            _test_operator_elwise_xy test_fun,
                                            _iarray_vml_fun_d_ab vml_fun_d,
                                            _iarray_vml_fun_s_ab vml_fun_s,
                                            iarray_data_type_t dtype,
                                            int32_t type_size,
                                            int64_t n,
                                            int64_t csize,
                                            int64_t bsize)
{
    void *buffer_x;
    void *buffer_y;
    void *buffer_r;
    size_t buffer_x_len;
    size_t buffer_y_len;
    size_t buffer_r_len;
    double tol;

    buffer_x_len = (size_t)type_size * n * n;
    buffer_y_len = (size_t)type_size * n * n;
    buffer_r_len = (size_t)type_size * n * n;
    buffer_x = ina_mem_alloc(buffer_x_len);
    buffer_y = ina_mem_alloc(buffer_y_len);
    buffer_r = ina_mem_alloc(buffer_r_len);

    if (type_size == sizeof(float)) {
        tol = 1e-06;
        ffill_buf((float*)buffer_x, (size_t)(n * n));
        ffill_buf((float*)buffer_y, (size_t)(n * n));
        vml_fun_s((const int)(n * n), buffer_x, buffer_y, buffer_r);
    }
    else {
        tol = 1e-14;
        dfill_buf((double*)buffer_x, (size_t)(n * n));
        dfill_buf((double*)buffer_y, (size_t)(n * n));
        vml_fun_d((const int)(n * n), buffer_x, buffer_y, buffer_r);
    }

    iarray_dtshape_t shape;

    shape.dtype = dtype;
    shape.ndim = 2;
    shape.shape[0] = (int64_t)n;
    shape.shape[1] = (int64_t)n;


    iarray_storage_t store;
    store.backend = IARRAY_STORAGE_BLOSC;
    store.filename = NULL;
    store.enforce_frame = false;
    for (int i = 0; i < shape.ndim; ++i) {
        store.chunkshape[i] = csize;
        store.blockshape[i] = bsize;
    }
    iarray_container_t *c_x;
    iarray_container_t *c_y;
    iarray_container_t *c_out;
    iarray_container_t *c_res;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &shape, buffer_x, buffer_x_len, &store, 0, &c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &shape, buffer_y, buffer_y_len, &store, 0, &c_y));
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &shape, buffer_r, buffer_r_len, &store, 0, &c_res));
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &shape, &store, 0, &c_out));

    INA_TEST_ASSERT_SUCCEED(_test_operator_xy(ctx, c_x, c_y, c_out, c_res, test_fun, tol));

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_res);

    ina_mem_free(buffer_x);
    ina_mem_free(buffer_y);
    ina_mem_free(buffer_r);

    return INA_SUCCESS;
}

INA_TEST_DATA(operator_element_wise) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(operator_element_wise)
{
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(operator_element_wise)
{
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(operator_element_wise, add_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 387;
    int64_t P = 44;
    int64_t B = 22;
    
    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_xy(data->ctx, iarray_operator_add, vdAdd, vsAdd, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, add_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 298;
    int64_t P = 66;
    int64_t B = 22;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_xy(data->ctx, iarray_operator_add, vdAdd, vsAdd, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, sub_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 237;
    int64_t P = 66;
    int64_t B = 11;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_xy(data->ctx, iarray_operator_sub, vdSub, vsSub, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, sub_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 249;
    int64_t P = 46;
    int64_t B = 22;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_xy(data->ctx, iarray_operator_sub, vdSub, vsSub, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, mul_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 273;
    int64_t P = 77;
    int64_t B = 22;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_xy(data->ctx, iarray_operator_mul, vdMul, vsMul, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, mul_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 243;
    int64_t P = 48;
    int64_t B = 12;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_xy(data->ctx, iarray_operator_mul, vdMul, vsMul, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, div_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 153;
    int64_t P = 102;
    int64_t B = 14;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_xy(data->ctx, iarray_operator_div, vdDiv, vsDiv, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, div_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 223;
    int64_t P = 51;
    int64_t B = 22;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_xy(data->ctx, iarray_operator_div, vdDiv, vsDiv, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, abs_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 40;
    int64_t P = 20;
    int64_t B = 10;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_abs, vdAbs, vsAbs, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, abs_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 513;
    int64_t P = 129;
    int64_t B = 14;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_abs, vdAbs, vsAbs, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, acos_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 433;
    int64_t P = 77;
    int64_t B = 10;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_acos, vdAcos, vsAcos, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, acos_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 133;
    int64_t P = 23;
    int64_t B = 22;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_acos, vdAcos, vsAcos, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, asin_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 131;
    int64_t P = 32;
    int64_t B = 12;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_asin, vdAsin, vsAsin, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, asin_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 131;
    int64_t P = 22;
    int64_t B = 22;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_asin, vdAsin, vsAsin, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, atanc_float_data)
{
    INA_TEST_ASSERT_TRUE(iarray_operator_atanc(data->ctx, NULL, NULL) == INA_ERR_NOT_IMPLEMENTED);
}

INA_TEST_FIXTURE(operator_element_wise, atan2_float_data)
{
    INA_TEST_ASSERT_TRUE(iarray_operator_atan2(data->ctx, NULL, NULL) == INA_ERR_NOT_IMPLEMENTED);
}

INA_TEST_FIXTURE(operator_element_wise, ceil_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 211;
    int64_t P = 44;
    int64_t B = 11;


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_ceil, vdCeil, vsCeil, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, ceil_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 321;
    int64_t P = 66;
    int64_t B = 12;


    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_ceil, vdCeil, vsCeil, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, cos_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 310;
    int64_t P = 60;
    int64_t B = 12;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_cos, vdCos, vsCos, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, cos_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 177;
    int64_t P = 29;
    int64_t B = 8;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_cos, vdCos, vsCos, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, cosh_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 109;
    int64_t P = 55;
    int64_t B = 12;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_cosh, vdCosh, vsCosh, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, cosh_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 500;
    int64_t P = 100;
    int64_t B = 10;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_cosh, vdCosh, vsCosh, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, exp_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 112;
    int64_t P = 12;
    int64_t B = 12;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_exp, vdExp, vsExp, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, exp_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 324;
    int64_t P = 77;
    int64_t B = 9;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_exp, vdExp, vsExp, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, floor_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 222;
    int64_t P = 222;
    int64_t B = 11;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_floor, vdFloor, vsFloor, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, floor_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 237;
    int64_t P = 67;
    int64_t B = 67;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_floor, vdFloor, vsFloor, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, log_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 115;
    int64_t P = 115;
    int64_t B = 115;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_log, vdLn, vsLn, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, log_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 223;
    int64_t P = 66;
    int64_t B = 66;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_log, vdLn, vsLn, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, log10_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 108;
    int64_t P = 55;
    int64_t B = 12;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_log10, vdLog10, vsLog10, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, log10_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 108;
    int64_t P = 51;
    int64_t B = 31;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_log10, vdLog10, vsLog10, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, pow_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 307;
    int64_t P = 70;
    int64_t B = 15;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_xy(data->ctx, iarray_operator_pow, vdPow, vsPow, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, pow_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 107;
    int64_t P = 70;
    int64_t B = 15;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_xy(data->ctx, iarray_operator_pow, vdPow, vsPow, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, sin_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 116;
    int64_t P = 16;
    int64_t B = 16;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_sin, vdSin, vsSin, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, sin_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 116;
    int64_t P = 16;
    int64_t B = 16;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_sin, vdSin, vsSin, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, sinh_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 417;
    int64_t P = 57;
    int64_t B = 17;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_sinh, vdSinh, vsSinh, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, sinh_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 417;
    int64_t P = 57;
    int64_t B = 17;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_sinh, vdSinh, vsSinh, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, sqrt_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 118;
    int64_t P = 33;
    int64_t B = 11;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_sqrt, vdSqrt, vsSqrt, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, sqrt_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 118;
    int64_t P = 33;
    int64_t B = 11;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_sqrt, vdSqrt, vsSqrt, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, tan_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 321;
    int64_t P = 321;
    int64_t B = 11;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_tan, vdTan, vsTan, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, tan_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 321;
    int64_t P = 321;
    int64_t B = 11;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_tan, vdTan, vsTan, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, tanh_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 120;
    int64_t P = 20;
    int64_t B = 10;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_tanh, vdTanh, vsTanh, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, tanh_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 120;
    int64_t P = 20;
    int64_t B = 10;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_tanh, vdTanh, vsTanh, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, erf_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 221;
    int64_t P = 41;
    int64_t B = 11;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_erf, vdErf, vsErf, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, erf_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 221;
    int64_t P = 41;
    int64_t B = 11;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_erf, vdErf, vsErf, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, erfc_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 257;
    int64_t P = 257;
    int64_t B = 11;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_erfc, vdErfc, vsErfc, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, erfc_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 257;
    int64_t P = 257;
    int64_t B = 11;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_erfc, vdErfc, vsErfc, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, cdfnorm_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 123;
    int64_t P = 23;
    int64_t B = 10;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_cdfnorm, vdCdfNorm, vsCdfNorm, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, cdfnorm_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 123;
    int64_t P = 23;
    int64_t B = 10;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_cdfnorm, vdCdfNorm, vsCdfNorm, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, erfinv_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 156;
    int64_t P = 55;
    int64_t B = 15;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_erfinv, vdErfInv, vsErfInv, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, erfinv_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 156;
    int64_t P = 55;
    int64_t B = 15;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_erfinv, vdErfInv, vsErfInv, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, erfcinv_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 125;
    int64_t P = 25;
    int64_t B = 12;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_erfcinv, vdErfcInv, vsErfcInv, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, erfcinv_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 125;
    int64_t P = 25;
    int64_t B = 12;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_erfcinv, vdErfcInv, vsErfcInv, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, cdfnorminv_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 125;
    int64_t P = 25;
    int64_t B = 12;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_cdfnorminv, vdCdfNormInv, vsCdfNormInv, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, cdfnorminv_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 125;
    int64_t P = 25;
    int64_t B = 12;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_cdfnorminv, vdCdfNormInv, vsCdfNormInv, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, lgamma_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 237;
    int64_t P = 47;
    int64_t B = 13;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_lgamma, vdLGamma, vsLGamma, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, lgamma_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 237;
    int64_t P = 47;
    int64_t B = 13;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_lgamma, vdLGamma, vsLGamma, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, tgamma_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 237;
    int64_t P = 47;
    int64_t B = 13;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_tgamma, vdTGamma, vsTGamma, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, tgamma_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 237;
    int64_t P = 47;
    int64_t B = 13;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_tgamma, vdTGamma, vsTGamma, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, expint1_float_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int64_t N = 129;
    int64_t P = 119;
    int64_t B = 109;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_expint1, vdExpInt1, vsExpInt1, dtype, type_size, N, P, B));
}

INA_TEST_FIXTURE(operator_element_wise, expint1_double_data)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int64_t N = 129;
    int64_t P = 119;
    int64_t B = 109;

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_operator_x(data->ctx, iarray_operator_expint1, vdExpInt1, vsExpInt1, dtype, type_size, N, P, B));
}
