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
static void fill_y(const double* x, double* y,double *z, int64_t nelem,
                   double (func)(double, double))
{
    for (int i = 0; i < nelem; i++) {
        z[i] = func(x[i], y[i]);
    }
}

static ina_rc_t execute_iarray_eval(iarray_config_t *cfg, int8_t ndim, const int64_t *shape,
                                    const int64_t *cshape, const int64_t *bshape,
                                    bool plain_buffer, double (func)(double, double),
                                    char *expr_str)
{
    iarray_context_t *ctx;
    iarray_expression_t* e;
    iarray_container_t* c_trans;
    iarray_container_t* c_y;
    iarray_container_t* c_x;
    iarray_container_t* c_out;

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape.ndim = ndim;
    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }

    iarray_dtshape_t transdtshape;
    transdtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    transdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        transdtshape.shape[i] = shape[ndim - 1 - i];
    }

    iarray_storage_t store;
    store.backend = plain_buffer ? IARRAY_STORAGE_PLAINBUFFER : IARRAY_STORAGE_BLOSC;
    store.enforce_frame = false;
    store.urlpath = NULL;
    if (!plain_buffer) {
        for (int i = 0; i < ndim; ++i) {
            store.chunkshape[i] = cshape[i];
            store.blockshape[i] = bshape[i];
        }
    }

    iarray_storage_t transstore;
    transstore.backend = plain_buffer ? IARRAY_STORAGE_PLAINBUFFER : IARRAY_STORAGE_BLOSC;
    transstore.enforce_frame = false;
    transstore.urlpath = NULL;
    if (!plain_buffer) {
        for (int i = 0; i < ndim; ++i) {
            transstore.chunkshape[i] = cshape[ndim - 1 - i];
            transstore.blockshape[i] = bshape[ndim - 1 - i];
        }
    }
    double *buffer_x = (double *) ina_mem_alloc(nelem * sizeof(double));
    double *buffer_y = (double *) ina_mem_alloc(nelem * sizeof(double));
    double *buffer_z = (double *) ina_mem_alloc(nelem * sizeof(double));

    fill_x(buffer_x, nelem);
    fill_x(buffer_y, nelem);

    INA_TEST_ASSERT_SUCCEED(iarray_context_new(cfg, &ctx));

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &transdtshape, (void*)buffer_x,
                                               nelem * sizeof(double), &transstore, 0, &c_trans));
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &dtshape, (void*)buffer_y,
                                               nelem * sizeof(double), &store, 0, &c_y));

    int64_t start[IARRAY_DIMENSION_MAX];
    int64_t stop[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ndim; ++i) {
        start[i] = 10;
        stop[i] = shape[i] / 2 + 10;
    }

    INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_trans, &c_x));

    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buffer_x, nelem * sizeof(double)));

    fill_y(buffer_x, buffer_y, buffer_z, nelem, func);

    INA_TEST_ASSERT_SUCCEED(iarray_expr_new(ctx, &e));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind(e, "x", c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind(e, "y", c_y));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind_out_properties(e, &dtshape, &store));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_compile(e, expr_str));
    INA_TEST_ASSERT_SUCCEED(iarray_eval(e, &c_out));

    // We use a quite low tolerance as MKL functions always differ from those in OS math libraries
    INA_TEST_ASSERT_SUCCEED(test_double_buffer_cmp(ctx, c_out, buffer_z, nelem * sizeof(double),
                                                   5e-15, 1e-14));

    iarray_expr_free(ctx, &e);

    ina_mem_free(buffer_x);
    ina_mem_free(buffer_y);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_trans);
    iarray_container_free(ctx, &c_x);
    iarray_context_free(&ctx);

    return INA_SUCCESS;
}

INA_TEST_DATA(expression_eval_transpose)
{
    iarray_config_t cfg;
    double (*func)(double, double);
    char *expr_str;
};

INA_TEST_SETUP(expression_eval_transpose)
{
    iarray_init();

    data->cfg = IARRAY_CONFIG_DEFAULTS;
    data->cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    data->cfg.compression_level = 9;
    data->cfg.max_num_threads = NTHREADS;
}

INA_TEST_TEARDOWN(expression_eval_transpose)
{
    INA_UNUSED(data);
    iarray_destroy();
}


static double expr(const double x, const double y)
{
    return sinh(x) + (cosh(x) - 1.35) - tanh(y + .2);
}

INA_TEST_FIXTURE(expression_eval_transpose, iterblosc_superchunk_2)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = expr;
    data->expr_str = "sinh(x) + (cosh(x) - 1.35) - tanh(y + .2)";

    int8_t ndim = 2;
    int64_t shape[] = {500, 1000};
    int64_t cshape[] = {245, 200};
    int64_t bshape[] = {25, 100};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, false, data->func, data->expr_str));
}

INA_TEST_FIXTURE(expression_eval_transpose, iterchunk_superchunk_3)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERCHUNK;
    data->func = expr;
    data->expr_str = "sinh(x) + (cosh(x) - 1.35) - tanh(y + .2)";

    int8_t ndim = 2;
    int64_t shape[] = {1299, 31};
    int64_t cshape[] = {500, 15};
    int64_t bshape[] = {200, 5};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, false, data->func, data->expr_str));
}

INA_TEST_FIXTURE(expression_eval_transpose, iterchunk_plainbuffer_4)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERCHUNK;
    data->func = expr;
    data->expr_str = "sinh(x) + (cosh(x) - 1.35) - tanh(y + .2)";

    int8_t ndim = 2;
    int64_t shape[] = {121, 121};
    int64_t cshape[] = {0, 0};
    int64_t bshape[] = {0, 0};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, true, data->func, data->expr_str));
}

