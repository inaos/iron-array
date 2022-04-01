/*
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <libiarray/iarray.h>
#include <tests/iarray_test.h>
#include <src/iarray_private.h>
#include <math.h>


#define NTHREADS 1


/* Compute and fill X values in a buffer */
static int ffill_x(float* x, int64_t nelem)
{
    /* Fill even values between 0. and 1. */
    float incx = 1.f / nelem;
    for (int i = 0; i < nelem; i++) {
        x[i] = incx * (float)i;
    }
    return 0;
}

/* Compute and fill Y values in a buffer */
static void ffill_y(const float* x, float* y, int64_t nelem, float (func)(float))
{
    for (int i = 0; i < nelem; i++) {
        y[i] = func(x[i]);
    }
}

static ina_rc_t
fexecute_iarray_eval(iarray_config_t *cfg, int8_t ndim, const int64_t *shape, const int64_t *cshape,
                     const int64_t *bshape, float (*func)(float), char *expr_str, bool contiguous,
                     char *urlpath)
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
        nelem *= shape[i];
    }

    iarray_storage_t store;
    store.contiguous = contiguous;
    store.urlpath = urlpath;
    for (int i = 0; i < ndim; ++i) {
        store.chunkshape[i] = cshape[i];
        store.blockshape[i] = bshape[i];
    }
    blosc2_remove_urlpath(store.urlpath);
    float *buffer_x = (float *) ina_mem_alloc(nelem * sizeof(float));
    float *buffer_y = (float *) ina_mem_alloc(nelem * sizeof(float));

    ffill_x(buffer_x, nelem);
    ffill_y(buffer_x, buffer_y, nelem, func);
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(cfg, &ctx));

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &dtshape, (void *) buffer_x,
                                               nelem * sizeof(float), &store, &c_x));

    iarray_storage_t outstore;
    outstore.contiguous = contiguous;
    outstore.urlpath = NULL;
    if (urlpath != NULL) {
        outstore.urlpath = "outarr.iarr";
    }
    for (int i = 0; i < ndim; ++i) {
        outstore.chunkshape[i] = cshape[i];
        outstore.blockshape[i] = bshape[i];
    }
    blosc2_remove_urlpath(outstore.urlpath);
    INA_TEST_ASSERT_SUCCEED(iarray_expr_new(ctx, dtshape.dtype, &e));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind(e, "x", c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind_out_properties(e, &dtshape, &outstore));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_compile(e, expr_str));
    INA_TEST_ASSERT_SUCCEED(iarray_eval(e, &c_out));

    // We use a quite low tolerance as MKL functions always differ from those in OS math libraries
    INA_TEST_ASSERT_SUCCEED(test_float_buffer_cmp(ctx, c_out, buffer_y, nelem * sizeof(float), 5e-3, 1e-5));

    iarray_expr_free(ctx, &e);
    ina_mem_free(buffer_x);
    ina_mem_free(buffer_y);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_context_free(&ctx);
    blosc2_remove_urlpath(store.urlpath);
    blosc2_remove_urlpath(outstore.urlpath);

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
    INA_UNUSED(data);
    iarray_destroy();
}

static float expr0(const float x)
{
    return (fabsf(-x) - 1.35f) * ceilf(x) * floorf(x - 8.5f);
}

INA_TEST_FIXTURE(expression_eval_float, iterblosc_superchunk0)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = expr0;
    data->expr_str = "(abs(-x) - 1.35) * ceil(x) * floor(x - 8.5)";

    int8_t ndim = 1;
    int64_t shape[] = {20000};
    int64_t cshape[] = {3456};
    int64_t bshape[] = {456};

    INA_TEST_ASSERT_SUCCEED(fexecute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape,
                                                 data->func, data->expr_str, false, NULL));
}

static float expr1(const float x)
{
    return (x - 1.35f) + sinf(.45f);
}

INA_TEST_FIXTURE(expression_eval_float, iterblosc_superchunk1)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = expr1;
    data->expr_str = "(x - 1.35) + sin(.45)";

    int8_t ndim = 1;
    int64_t shape[] = {20000};
    int64_t cshape[] = {3456};
    int64_t bshape[] = {456};

    INA_TEST_ASSERT_SUCCEED(fexecute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape,
                                                 data->func, data->expr_str, false, "arr.iarr"));
}

static float expr2(const float x)
{
    return sinhf(x) + (coshf(x) - 1.35f) - tanhf(x + .2f);
}

INA_TEST_FIXTURE(expression_eval_float, iterblosc_superchunk2)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = expr2;
    data->expr_str = "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)";

    int8_t ndim = 1;
    int64_t shape[] = {20000};
    int64_t cshape[] = {3456};
    int64_t bshape[] = {456};

    INA_TEST_ASSERT_SUCCEED(fexecute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape,
                                                 data->func, data->expr_str, true, NULL));
}

static float expr3(const float x)
{
    return asinf(x) + (acosf(x) - 1.35f) - atanf(x + .2f);
}

INA_TEST_FIXTURE(expression_eval_float, iterblosc_superchunk)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = expr3;
    data->expr_str = "asin(x) + (acos(x) - 1.35) - atan(x + .2)";

    int8_t ndim = 3;
    int64_t shape[] = {10, 23, 21};
    int64_t cshape[] = {5, 3, 17};
    int64_t bshape[] = {3, 2, 7};

    INA_TEST_ASSERT_SUCCEED(fexecute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape,
                                                 data->func, data->expr_str, true, "arr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(expression_eval_float, iterblosc_superchunk)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = expr3;
    data->expr_str = "asin(x) + (acos(x) - 1.35) - atan(x + .2)";

    int8_t ndim = 3;
    int64_t shape[] = {100, 230, 121};
    int64_t cshape[] = {31, 32, 17};
    int64_t bshape[] = {7, 7, 7};

    INA_TEST_ASSERT_SUCCEED(fexecute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape,
                                                 data->func, data->expr_str, true, "arr.iarr"));
}
*/

static float expr4(const float x)
{
    return expf(x) + (logf(x) - 1.35f) - log10f(x + .2f);
}

INA_TEST_FIXTURE(expression_eval_float, iterblosc_superchunk_4)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = expr4;
    data->expr_str = "exp(x) + (log(x) - 1.35) - log10(x + .2)";

    int8_t ndim = 3;
    int64_t shape[] = {100, 230, 121};
    int64_t cshape[] = {31, 32, 17};
    int64_t bshape[] = {7, 7, 7};

    INA_TEST_ASSERT_SUCCEED(fexecute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape,
                                                 data->func, data->expr_str, false, NULL));
}

static float expr5(const float x)
{
    return powf(2.71828, x) / x * logf(x);
}

INA_TEST_FIXTURE(expression_eval_float, iterblosc_superchunk_5)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = expr5;
    data->expr_str = "2.71828**x / x * log(x)";

    int8_t ndim = 3;
    int64_t shape[] = {100, 230, 121};
    int64_t cshape[] = {31, 32, 17};
    int64_t bshape[] = {7, 7, 7};

    INA_TEST_ASSERT_SUCCEED(fexecute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape,
                                                 data->func, data->expr_str, false, NULL));
}
