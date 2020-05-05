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


#define NTHREADS 1


/* Compute and fill X values in a buffer */
static int _fill_x(float* x, int nelem)
{
    /* Fill even values between 0. and 1. */
    float incx = 1.f / nelem;
    for (int i = 0; i < nelem; i++) {
        x[i] = incx * (float)i;
    }
    return 0;
}

/* Compute and fill Y values in a buffer */
static void _fill_y(const float* x, float* y, int nelem, float (func)(float))
{
    for (int i = 0; i < nelem; i++) {
        y[i] = func(x[i]);
    }
}

static ina_rc_t _execute_iarray_eval(iarray_config_t *cfg, int8_t ndim, int64_t *shape, int64_t *pshape,
                                     bool plain_buffer, float (func)(float), char* expr_str)
{
    iarray_context_t *ctx;
    iarray_expression_t* e;
    iarray_container_t* c_x;
    iarray_container_t* c_out;

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_FLOAT;
    dtshape.ndim = ndim;
    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        dtshape.pshape[i] = plain_buffer ? 0 : pshape[i];
        nelem *= shape[i];
    }

    iarray_store_properties_t store;
    store.backend = plain_buffer ? IARRAY_STORAGE_PLAINBUFFER : IARRAY_STORAGE_BLOSC;
    store.enforce_frame = false;
    store.filename = NULL;

    float *buffer_x = (float *) ina_mem_alloc(nelem * sizeof(float));
    float *buffer_y = (float *) ina_mem_alloc(nelem * sizeof(float));

    _fill_x(buffer_x, nelem);
    _fill_y(buffer_x, buffer_y, nelem, func);
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(cfg, &ctx));

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &dtshape, (void*)buffer_x, nelem * sizeof(float), &store, 0, &c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &dtshape, &store, 0, &c_out));

    INA_TEST_ASSERT_SUCCEED(iarray_expr_new(ctx, &e));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind(e, "x", c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind_out_properties(e, &dtshape, &store));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_compile(e, expr_str));
    INA_TEST_ASSERT_SUCCEED(iarray_eval(e, &c_out));

    // We use a quite low tolerance as MKL functions always differ from those in OS math libraries
    INA_TEST_ASSERT_SUCCEED(_iarray_test_container_flt_buffer_cmp(ctx, c_out, buffer_y, nelem * sizeof(float), 5e-3));

    iarray_expr_free(ctx, &e);
    ina_mem_free(buffer_x);
    ina_mem_free(buffer_y);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_context_free(&ctx);

    return INA_SUCCESS;
}

INA_TEST_DATA(expression_eval_float)
{
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
}

INA_TEST_TEARDOWN(expression_eval_float)
{
    iarray_destroy();
}

//static float expr0(const float x)
//{
//    return (fabsf(-x) - 1.35f) * ceilf(x) * floorf(x - 8.5f);
//}
//
//
//static float expr1(const float x)
//{
//    return (x - 1.35) + sin(.45);  // TODO: fix evaluation of func(constant)
//}


static float expr2(const float x)
{
    return sinhf(x) + (coshf(x) - 1.35f) - tanhf(x + .2f);
}

INA_TEST_FIXTURE(expression_eval_float, iterblosc_superchunk)
{
    data->cfg.eval_flags = IARRAY_EVAL_METHOD_ITERBLOSC | (IARRAY_EVAL_ENGINE_INTERPRETER << 3);
    data->func = expr2;
    data->expr_str = "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)";

    int8_t ndim = 1;
    int64_t shape[] = {20000};
    int64_t pshape[] = {3456};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, ndim, shape, pshape, false, data->func, data->expr_str));
}

static float expr3(const float x)
{
    return asinf(x) + (acosf(x) - 1.35f) - atanf(x + .2f);
}

INA_TEST_FIXTURE(expression_eval_float, iterchunk_superchunk)
{
    data->cfg.eval_flags = IARRAY_EVAL_METHOD_ITERCHUNK;
    data->func = expr3;
    data->expr_str = "asin(x) + (acos(x) - 1.35) - atan(x + .2)";

    int8_t ndim = 3;
    int64_t shape[] = {100, 230, 121};
    int64_t pshape[] = {31, 32, 17};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, ndim, shape, pshape, false, data->func, data->expr_str));
}

//static float expr4(const float x)
//{
//    return expf(x) + (logf(x) - 1.35f) - log10f(x + .2f); //TODO: Fix error with this function
//}
//
//INA_TEST_FIXTURE(expression_eval_float, iterchunk_plainbuffer_4)
//{
//    data->cfg.eval_flags = IARRAY_EVAL_METHOD_ITERCHUNK;
//    data->func = expr4;
//    data->expr_str = "expf(x) + (logf(x) - 1.35f) - log10f(x + .2f)";
//
//    int8_t ndim = 3;
//    int64_t shape[] = {121, 121, 123};
//    int64_t pshape[] = {0, 0, 0};
//
//    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, ndim, shape, pshape, true, data->func, data->expr_str));
//}

static float expr5(const float x)
{
    return sqrtf(x) + atan2f(x, x) + powf(x, x);
}

INA_TEST_FIXTURE(expression_eval_float, iterchunk_plainbuffer)
{
    data->cfg.eval_flags = IARRAY_EVAL_METHOD_ITERCHUNK;
    data->func = expr5;
    data->expr_str = "sqrt(x) + atan2(x, x) + pow(x, x)";

    int8_t ndim = 3;
    int64_t shape[] = {121, 2, 123};
    int64_t pshape[] = {0, 0, 0};

    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, ndim, shape, pshape, true, data->func, data->expr_str));
}
