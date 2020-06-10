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


#define NTHREADS 2


/* Compute and fill X values in a buffer */
static int _fill_x(double* x, int64_t nelem)
{
    /* Fill even values between 0. and 1. */
    double incx = 1. / nelem;
    for (int i = 0; i < nelem; i++) {
        x[i] = incx * i;
    }
    return 0;
}

/* Compute and fill Y values in a buffer */
static void _fill_y(const double* x, double* y, int64_t nelem, double (func)(double))
{
    for (int i = 0; i < nelem; i++) {
        y[i] = func(x[i]);
    }
}

static ina_rc_t _execute_iarray_eval(iarray_config_t *cfg, int8_t ndim, int64_t *shape, int64_t *pshape, int64_t *bshape,
                                     bool plain_buffer, double (func)(double), char* expr_str)
{
    iarray_context_t *ctx;
    iarray_expression_t* e;
    iarray_container_t* c_x;
    iarray_container_t* c_x2;
    iarray_container_t* c_out;

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape.ndim = ndim;
    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }

    iarray_dtshape_t dtshape2;
    dtshape2.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape2.ndim = ndim;
    int64_t nelem2 = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape2.shape[i] = shape[i] / 2;
        nelem2 *= dtshape2.shape[i];
    }

    iarray_storage_t store;
    store.backend = plain_buffer ? IARRAY_STORAGE_PLAINBUFFER : IARRAY_STORAGE_BLOSC;
    store.enforce_frame = false;
    store.filename = NULL;
    if (!plain_buffer) {
        for (int i = 0; i < ndim; ++i) {
            store.pshape[i] = pshape[i];
            store.bshape[i] = bshape[i];
        }
    }

    double *buffer_x = (double *) ina_mem_alloc(nelem * sizeof(double));
    double *buffer_y = (double *) ina_mem_alloc(nelem * sizeof(double));

    _fill_x(buffer_x, nelem);

    INA_TEST_ASSERT_SUCCEED(iarray_context_new(cfg, &ctx));

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &dtshape, (void*)buffer_x, nelem * sizeof(double), &store, 0, &c_x));

    int64_t start[IARRAY_DIMENSION_MAX];
    int64_t stop[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ndim; ++i) {
        start[i] = 20;
        stop[i] = shape[i] / 2 + 20;
    }

    INA_TEST_ASSERT_SUCCEED(iarray_get_slice(ctx, c_x, start, stop, true, &store, 0, &c_x2));
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x2, buffer_x, nelem * sizeof(double)));

    _fill_y(buffer_x, buffer_y, nelem2, func);

    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &dtshape2, &store, 0, &c_out));

    INA_TEST_ASSERT_SUCCEED(iarray_expr_new(ctx, &e));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind(e, "x", c_x2));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind_out_properties(e, &dtshape2, &store));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_compile(e, expr_str));
    INA_TEST_ASSERT_SUCCEED(iarray_eval(e, &c_out));

    // We use a quite low tolerance as MKL functions always differ from those in OS math libraries
    INA_TEST_ASSERT_SUCCEED(_iarray_test_container_dbl_buffer_cmp(ctx, c_out, buffer_y, nelem2 * sizeof(double), 1e-13));

    iarray_expr_free(ctx, &e);
    ina_mem_free(buffer_x);
    ina_mem_free(buffer_y);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_context_free(&ctx);

    return INA_SUCCESS;
}

INA_TEST_DATA(expression_eval_view)
{
    iarray_config_t cfg;
    double (*func)(double);
    char *expr_str;
};

INA_TEST_SETUP(expression_eval_view)
{
    iarray_init();

    data->cfg = IARRAY_CONFIG_DEFAULTS;
    data->cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    data->cfg.compression_level = 9;
    data->cfg.max_num_threads = NTHREADS;
}

INA_TEST_TEARDOWN(expression_eval_view)
{
    INA_UNUSED(data);
    iarray_destroy();
}

//static double expr0(const double x)
//{
//    return (fabs(-x) - 1.35) * ceil(x) * floor(x - 8.5);
//}


//static double expr1(const double x)
//{
//    return (x - 1.35) + sin(.45);  // TODO: fix evaluation of func(constant)
//}

static double expr2(const double x)
{
    return sinh(x) + (cosh(x) - 1.35) - tanh(x + .2);
}

INA_TEST_FIXTURE_SKIP(expression_eval_view, iterblosc_superchunk_2)
{
    data->cfg.eval_flags = IARRAY_EVAL_METHOD_ITERBLOSC | (IARRAY_EVAL_ENGINE_COMPILER << 3);
    data->func = expr2;
    data->expr_str = "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)";

    int8_t ndim = 1;
    int64_t shape[] = {20000};
    int64_t pshape[] = {3456};
    int64_t bshape[] = {236};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, ndim, shape, pshape, bshape, false, data->func, data->expr_str));
}

static double expr3(const double x)
{
    return asin(x) + (acos(x) - 1.35) - atan(x + .2);
}

INA_TEST_FIXTURE(expression_eval_view, iterchunk_superchunk_3)
{
    data->cfg.eval_flags = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = expr3;
    data->expr_str = "asin(x) + (acos(x) - 1.35) - atan(x + .2)";

    int8_t ndim = 3;
    int64_t shape[] = {100, 100, 100};
    int64_t pshape[] = {20, 20, 20};
    int64_t bshape[] = {10, 10, 10};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, ndim, shape, pshape, bshape, false, data->func, data->expr_str));
}

static double expr4(const double x)
{
    return exp(x) + (log(x) - 1.35) - log10(x + .2);
}

INA_TEST_FIXTURE(expression_eval_view, iterchunk_plainbuffer_4)
{
    data->cfg.eval_flags = IARRAY_EVAL_METHOD_ITERCHUNK;
    data->func = expr4;
    data->expr_str = "exp(x) + (log(x) - 1.35) - log10(x + .2)";

    int8_t ndim = 3;
    int64_t shape[] = {121, 121, 123};
    int64_t pshape[] = {0, 0, 0};
    int64_t bshape[] = {0, 0, 0};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, ndim, shape, pshape, bshape, true, data->func, data->expr_str));
}

static double expr5(const double x)
{
    return sqrt(x) + atan2(x, x) + pow(x, x);
}

INA_TEST_FIXTURE(expression_eval_view, iterchunk_plainbuffer_5)
{
    data->cfg.eval_flags = IARRAY_EVAL_METHOD_ITERCHUNK;
    data->func = expr5;
    data->expr_str = "sqrt(x) + atan2(x, x) + pow(x, x)";

    int8_t ndim = 3;
    int64_t shape[] = {121, 121, 123};
    int64_t pshape[] = {0, 0, 0};
    int64_t bshape[] = {0, 0, 0};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, ndim, shape, pshape, bshape, true, data->func, data->expr_str));
}

