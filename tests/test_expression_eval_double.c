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

// Use 2-dim arrays here
#define NROWS 50
#define NCOLS 3000
#define NROWS_CHUNK 20
#define NCOLS_CHUNK 1000
#define NELEM (NROWS * NCOLS)
#define NTHREADS 1


/* Compute and fill X values in a buffer */
static int _fill_x(double* x)
{
    /* Fill even values between 0. and 1. */
    double incx = 1. / NELEM;
    for (int i = 0; i < NELEM; i++) {
        x[i] = incx * i;
    }
    return 0;
}

/* Compute and fill Y values in a buffer */
static void _fill_y(const double* x, double* y, double (func)(double))
{
    for (int i = 0; i < NELEM; i++) {
        y[i] = func(x[i]);
    }
}

static ina_rc_t _execute_iarray_eval(iarray_config_t *cfg, const double *buffer_x, double *buffer_y,
                                     size_t buffer_len, bool plain_buffer, double (func)(double),
                                     char* expr_str)
{
    iarray_context_t *ctx;
    iarray_expression_t* e;
    iarray_container_t* c_x;
    iarray_container_t* c_out;

    iarray_dtshape_t shape;
    shape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    shape.ndim = 2;
    shape.shape[0] = NROWS;
    shape.shape[1] = NCOLS;
    shape.pshape[0] = plain_buffer ? 0 : NROWS_CHUNK;
    shape.pshape[1] = plain_buffer ? 0 : NCOLS_CHUNK;

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
    INA_TEST_ASSERT_SUCCEED(iarray_eval(e));

    // We use a quite low tolerance as MKL functions always differ from those in OS math libraries
    INA_TEST_ASSERT_SUCCEED(_iarray_test_container_dbl_buffer_cmp(ctx, c_out, buffer_y, buffer_len, 5e-13));

    iarray_expr_free(ctx, &e);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_context_free(&ctx);

    return INA_SUCCESS;
}

INA_TEST_DATA(expression_eval_double)
{
    size_t buf_len;
    double *buffer_x;
    double *buffer_y;
    iarray_config_t cfg;
    double (*func)(double);
    char *expr_str;
};

INA_TEST_SETUP(expression_eval_double)
{
    iarray_init();

    data->cfg = IARRAY_CONFIG_DEFAULTS;
    data->cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    data->cfg.compression_level = 9;
    data->cfg.max_num_threads = NTHREADS;

    data->buf_len = sizeof(double)*NELEM;
    data->buffer_x = ina_mem_alloc(data->buf_len);
    data->buffer_y = ina_mem_alloc(data->buf_len);

    _fill_x(data->buffer_x);
}

INA_TEST_TEARDOWN(expression_eval_double)
{
    ina_mem_free(data->buffer_x);
    ina_mem_free(data->buffer_y);

    iarray_destroy();
}

static double expr_(const double x)
{
    return (x - 2.3) * (x - 1.35) * (x + 4.2);
}

INA_TEST_FIXTURE(expression_eval_double, iterblosc_superchunk)
{
    data->cfg.eval_flags = IARRAY_EXPR_EVAL_ITERBLOSC;
    data->func = expr_;
    data->expr_str = "(x - 2.3) * (x - 1.35) * (x + 4.2)";

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, data->buffer_x, data->buffer_y,
                                                 data->buf_len, false, data->func, data->expr_str));
}

INA_TEST_FIXTURE(expression_eval_double, iterblosc2_superchunk)
{
    data->cfg.eval_flags = IARRAY_EXPR_EVAL_ITERBLOSC2;
    data->func = expr_;
    data->expr_str = "(x - 2.3) * (x - 1.35) * (x + 4.2)";

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, data->buffer_x, data->buffer_y,
                                                 data->buf_len, false, data->func, data->expr_str));
}

static double expr0(const double x)
{
    return (fabs(-x) - 1.35) * ceil(x) * floor(x - 8.5);
}

static double expr1(const double x)
{
    return (cos(x) - 1.35) * tan(x) * sin(x - 8.5);
    //return (x - 1.35) + sin(.45);  // TODO: fix evaluation of func(constant)
}

static double expr2(const double x)
{
    return sinh(x) + (cosh(x) - 1.35) - tanh(x + .2);
}

INA_TEST_FIXTURE(expression_eval_double, iterchunk_superchunk)
{
    data->cfg.eval_flags = IARRAY_EXPR_EVAL_ITERCHUNK;
    data->func = expr2;
    data->expr_str = "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)";

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, data->buffer_x, data->buffer_y,
                                                 data->buf_len, false, data->func, data->expr_str));
}

static double expr3(const double x)
{
    return asin(x) + (acos(x) - 1.35) - atan(x + .2);
}

INA_TEST_FIXTURE(expression_eval_double, iterchunk_superchunk2)
{
    data->cfg.eval_flags = IARRAY_EXPR_EVAL_ITERCHUNK;
    data->func = expr3;
    data->expr_str = "asin(x) + (acos(x) - 1.35) - atan(x + .2)";

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, data->buffer_x, data->buffer_y,
        data->buf_len, false, data->func, data->expr_str));
}


INA_TEST_FIXTURE(expression_eval_double, default_superchunk2)
{
    data->cfg.eval_flags = IARRAY_EXPR_EVAL_AUTO;
    data->func = expr3;
    data->expr_str = "asin(x) + (acos(x) - 1.35) - atan(x + .2)";

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, data->buffer_x, data->buffer_y,
                                                 data->buf_len, false, data->func, data->expr_str));
}

static double expr4(const double x)
{
    return exp(x) + (log(x) - 1.35) - log10(x + .2);
}

static double expr5(const double x)
{
    return sqrt(x) + atan2(x, x) + pow(x, x);
}


INA_TEST_FIXTURE(expression_eval_double, iterchunk_plainbuffer)
{
    data->cfg.eval_flags = IARRAY_EXPR_EVAL_ITERCHUNK;
    data->func = expr5;
    data->expr_str = "sqrt(x) + atan2(x, x) + pow(x, x)";

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, data->buffer_x, data->buffer_y,
        data->buf_len, true, data->func, data->expr_str));
}


INA_TEST_FIXTURE(expression_eval_double, default_plainbuffer)
{
    data->cfg.eval_flags = IARRAY_EXPR_EVAL_AUTO;
    data->func = expr5;
    data->expr_str = "sqrt(x) + atan2(x, x) + pow(x, x)";

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, data->buffer_x, data->buffer_y,
                                                 data->buf_len, true, data->func, data->expr_str));
}
