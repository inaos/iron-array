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
#include <tests/iarray_test.h>
#include <src/iarray_private.h>
#include <math.h>

#define NCHUNKS  2  // per construction, must be a minimum of 2
#define NITEMS_CHUNK (20 * 1000)
#define NELEM (((NCHUNKS - 1) * NITEMS_CHUNK) + 10)
#define NTHREADS 1


/* Compute and fill X values in a buffer */
static int _fill_x(float* x)
{
    /* Fill even values between 0. and 1. */
    float incx = 1.f / NELEM;
    for (int i = 0; i < NELEM; i++) {
        x[i] = incx * (float)i;
    }
    return 0;
}

/* Compute and fill Y values in a buffer */
static void _fill_y(const float* x, float* y, float (func)(float))
{
    for (int i = 0; i < NELEM; i++) {
        y[i] = func(x[i]);
    }
}

static ina_rc_t _execute_iarray_eval(iarray_config_t *cfg, const float *buffer_x, float *buffer_y,
                                     size_t buffer_len, bool plain_buffer, float (func)(float),
                                     char* expr_str)
{
    iarray_context_t *ctx;
    iarray_expression_t* e;
    iarray_container_t* c_x;
    iarray_container_t* c_out;

    iarray_dtshape_t shape;
    shape.dtype = IARRAY_DATA_TYPE_FLOAT;
    shape.ndim = 1;
    shape.shape[0] = NELEM;
    shape.pshape[0] = plain_buffer ? 0 : NITEMS_CHUNK;

    iarray_store_properties_t store;
    store.backend = plain_buffer ? IARRAY_STORAGE_PLAINBUFFER : IARRAY_STORAGE_BLOSC;
    store.enforce_frame = false;
    store.filename = NULL;

    _fill_y(buffer_x, buffer_y, func);

    INA_TEST_ASSERT_SUCCEED(iarray_context_new(cfg, &ctx));

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &shape, (void*)buffer_x, buffer_len, &store, 0, &c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &shape, &store, 0, &c_out));

    INA_TEST_ASSERT_SUCCEED(iarray_expr_new(ctx, &e));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind(e, "x", c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind_out(e, c_out));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_compile(e, expr_str));
    INA_TEST_ASSERT_SUCCEED(iarray_eval(e, c_out));

    // We use a quite low tolerance as MKL functions always differ from those in OS math libraries
    INA_TEST_ASSERT_SUCCEED(_iarray_test_container_flt_buffer_cmp(ctx, c_out, buffer_y, buffer_len, 5e-6));

    iarray_expr_free(ctx, &e);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_context_free(&ctx);

    return INA_SUCCESS;
}

INA_TEST_DATA(expression_eval_float)
{
    size_t buf_len;
    float *buffer_x;
    float *buffer_y;
    iarray_config_t cfg;
    float (*func)(float);
    char *expr_str;
};

INA_TEST_SETUP(expression_eval_float)
{
    iarray_init();

    data->cfg = IARRAY_CONFIG_DEFAULTS;
    data->cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    data->cfg.compression_level = 9;
    data->cfg.max_num_threads = NTHREADS;

    data->buf_len = sizeof(float)*NELEM;
    data->buffer_x = ina_mem_alloc(data->buf_len);
    data->buffer_y = ina_mem_alloc(data->buf_len);

    _fill_x(data->buffer_x);
}

INA_TEST_TEARDOWN(expression_eval_float)
{
    ina_mem_free(data->buffer_x);
    ina_mem_free(data->buffer_y);

    iarray_destroy();
}

static float expr0(const float x)
{
    return (fabsf(-x) - 1.35f) * ceilf(x) * floorf(x - 8.5f);
}


static float expr1(const float x)
{
    return (cosf(x) - 1.35f) * tanf(x) * sinf(x - 8.5f);
    //return (x - 1.35) + sin(.45);  // TODO: fix evaluation of func(constant)
}


static float expr2(const float x)
{
    return sinhf(x) + (coshf(x) - 1.35f) - tanhf(x + .2f);
}

INA_TEST_FIXTURE_SKIP(expression_eval_float, iterblosc_superchunk)
{
    data->cfg.eval_flags = IARRAY_EXPR_EVAL_ITERBLOSC;
    data->func = expr2;
    data->expr_str = "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)";

  INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, data->buffer_x, data->buffer_y,
      data->buf_len, false, data->func, data->expr_str));
}

static float expr3(const float x)
{
    return asinf(x) + (acosf(x) - 1.35f) - atanf(x + .2f);
}

INA_TEST_FIXTURE(expression_eval_float, iterchunk_superchunk)
{
    data->cfg.eval_flags = IARRAY_EXPR_EVAL_ITERCHUNK;
    data->func = expr3;
    data->expr_str = "asin(x) + (acos(x) - 1.35) - atan(x + .2)";

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, data->buffer_x, data->buffer_y,
        data->buf_len, false, data->func, data->expr_str));
}

static float expr4(const float x)
{
    return expf(x) + (logf(x) - 1.35f) - log10f(x + .2f);
}

static float expr5(const float x)
{
    return sqrtf(x) + atan2f(x, x) + powf(x, x);
}

INA_TEST_FIXTURE(expression_eval_float, iterchunk_plainbuffer)
{
    data->cfg.eval_flags = IARRAY_EXPR_EVAL_ITERCHUNK;
    data->func = expr5;
    data->expr_str = "sqrt(x) + atan2(x, x) + pow(x, x)";

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, data->buffer_x, data->buffer_y,
        data->buf_len, true, data->func, data->expr_str));
}
