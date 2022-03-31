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
static int fill_x(double* x, int64_t nelem)
{
    /* Fill even values between 0. and 1. */
    double incx = 1. / nelem;
    for (int i = 0; i < nelem; i++) {
        x[i] = incx * i;
    }
    return 0;
}

/* Compute and fill Y values in a buffer */
static void fill_y(const double* x, double* y, int64_t nelem, double (func)(double))
{
    for (int i = 0; i < nelem; i++) {
        y[i] = func(x[i]);
    }
}

static ina_rc_t
execute_iarray_eval(iarray_config_t *cfg, int8_t ndim, const int64_t *shape, const int64_t *cshape,
                    const int64_t *bshape, double (*func)(double), char *expr_str, bool contiguous,
                    char *urlpath)
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
    store.contiguous = contiguous;
    store.urlpath = urlpath;
    for (int i = 0; i < ndim; ++i) {
        store.chunkshape[i] = cshape[i];
        store.blockshape[i] = bshape[i];
    }
    blosc2_remove_urlpath(store.urlpath);

    double *buffer_x = (double *) ina_mem_alloc(nelem * sizeof(double));
    double *buffer_y = (double *) ina_mem_alloc(nelem * sizeof(double));

    fill_x(buffer_x, nelem);

    INA_TEST_ASSERT_SUCCEED(iarray_context_new(cfg, &ctx));

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &dtshape, (void*)buffer_x, nelem * sizeof(double), &store, 0, &c_x));

    int64_t start[IARRAY_DIMENSION_MAX];
    int64_t stop[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ndim; ++i) {
        start[i] = 10;
        stop[i] = shape[i] / 2 + 10;
    }

    INA_TEST_ASSERT_SUCCEED(iarray_get_slice(ctx, c_x, start, stop, true, &store, 0, &c_x2));
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x2, buffer_x, nelem * sizeof(double)));

    fill_y(buffer_x, buffer_y, nelem2, func);

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
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind(e, "x", c_x2));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind_out_properties(e, &dtshape2, &outstore));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_compile(e, expr_str));
    INA_TEST_ASSERT_SUCCEED(iarray_eval(e, &c_out));

    // We use a quite low tolerance as MKL functions always differ from those in OS math libraries
    INA_TEST_ASSERT_SUCCEED(test_double_buffer_cmp(ctx, c_out, buffer_y, nelem2 * sizeof(double), 5e-15, 1e-14));

    iarray_expr_free(ctx, &e);

    ina_mem_free(buffer_x);
    ina_mem_free(buffer_y);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_x2);
    iarray_context_free(&ctx);
    blosc2_remove_urlpath(store.urlpath);
    blosc2_remove_urlpath(outstore.urlpath);

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

INA_TEST_FIXTURE(expression_eval_view, iterblosc_superchunk_2)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = expr2;
    data->expr_str = "sinh(x) + (cosh(x) - 1.35) - tanh(x + .2)";

    int8_t ndim = 2;
    int64_t shape[] = {200, 1000};
    int64_t cshape[] = {50, 200};
    int64_t bshape[] = {25, 100};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func,
                                                data->expr_str, false, NULL));
}

static double expr3(const double x)
{
    return asin(x + .1) + (acos(x) - 1.35) - atan(x + .2);
}

INA_TEST_FIXTURE(expression_eval_view, iterchunk_superchunk_3)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERCHUNK;
    data->func = expr3;
    data->expr_str = "asin(x + 2) + (acos(x) - 1.35) - atan(x + .2)";

    int8_t ndim = 3;
    int64_t shape[] = {100, 100, 100};
    int64_t cshape[] = {20, 20, 20};
    int64_t bshape[] = {10, 10, 10};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func,
                                                data->expr_str, false, "arr.iarr"));
}

static double expr4(const double x)
{
    return exp(x) + (log(x) - 1.35) - log10(x + .2);
}

INA_TEST_FIXTURE(expression_eval_view, iterchunk_superchunk_4)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERCHUNK;
    data->func = expr4;
    data->expr_str = "exp(x) + (log(x) - 1.35) - log10(x + .2)";

    int8_t ndim = 3;
    int64_t shape[] = {121, 121, 123};
    int64_t cshape[] = {30, 30, 30};
    int64_t bshape[] = {10, 10, 10};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func,
                                                data->expr_str, true, NULL));
}

static double expr5(const double x)
{
    return sqrt(x) + atan2(x, x) + pow(x, x);
}

INA_TEST_FIXTURE(expression_eval_view, iterchunk_plainbuffer_5)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERCHUNK;
    data->func = expr5;
    data->expr_str = "sqrt(x) + arctan2(x, x) + pow(x, x)";

    int8_t ndim = 3;
    int64_t shape[] = {120, 120, 120};
    int64_t cshape[] = {20, 20, 20};
    int64_t bshape[] = {10, 10, 10};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func,
                                                data->expr_str, true, "arr.iarr"));
}
