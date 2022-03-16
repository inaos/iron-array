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

/* Special case for a constant function */
static int32_t const_(const int32_t x)
{
    return 2 - (x - x);
}


/* Compute and fill X values in a buffer */
static int fill_x(int32_t* x, int64_t nelem)
{
    int32_t incx = 2 * nelem;
    for (int i = 0; i < nelem; i++) {
        x[i] = incx * i;
    }
    return 0;
}

/* Compute and fill Y values in a buffer */
static void fill_y(const int32_t* x, int32_t* y, int64_t nelem, int32_t (func)(int32_t))
{
    for (int i = 0; i < nelem; i++) {
        y[i] = func(x[i]);
    }
}

static ina_rc_t execute_iarray_eval(iarray_config_t *cfg, int8_t ndim, const int64_t *shape, const int64_t *cshape,
                                    const int64_t *bshape, int32_t (func)(int32_t), char* expr_str, bool contiguous,
                                    char *urlpath)
{
    iarray_context_t *ctx;
    iarray_expression_t* e;
    iarray_container_t* c_x;
    iarray_container_t* c_out;
    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_INT32;
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

    int32_t *buffer_x = (int32_t *) ina_mem_alloc(nelem * sizeof(int32_t));
    int32_t *buffer_y = (int32_t *) ina_mem_alloc(nelem * sizeof(int32_t));

    fill_x(buffer_x, nelem);
    fill_y(buffer_x, buffer_y, nelem, func);

    INA_TEST_ASSERT_SUCCEED(iarray_context_new(cfg, &ctx));
    blosc2_remove_urlpath(store.urlpath);
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &dtshape, (void*)buffer_x, nelem * sizeof(int32_t), &store, 0, &c_x));

    INA_TEST_ASSERT_SUCCEED(iarray_expr_new(ctx, &dtshape, &e));
    if (func != const_) {
        INA_TEST_ASSERT_SUCCEED(iarray_expr_bind(e, "x", c_x));
    }
    else {

    }
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

    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind_out_properties(e, &dtshape, &outstore));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_compile(e, expr_str));
    INA_TEST_ASSERT_SUCCEED(iarray_eval(e, &c_out));


    // We use a quite low tolerance as MKL functions always differ from those in OS math libraries
    INA_TEST_ASSERT_SUCCEED(test_double_buffer_cmp(ctx, c_out, buffer_y, nelem * sizeof(int32_t), 5e-15, 5e-14));

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

INA_TEST_DATA(expression_eval_int32)
{
    iarray_config_t cfg;
    int32_t (*func)(int32_t);
    char *expr_str;
};

INA_TEST_SETUP(expression_eval_int32)
{
    iarray_init();

    data->cfg = IARRAY_CONFIG_DEFAULTS;
    data->cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    data->cfg.compression_level = 9;
    data->cfg.max_num_threads = NTHREADS;
}

INA_TEST_TEARDOWN(expression_eval_int32)
{
    INA_UNUSED(data);
    iarray_destroy();
}

static int32_t expr_(const int32_t x)
{
    return (x - 3) * (x - 1) * (x + 4);
}

INA_TEST_FIXTURE(expression_eval_int32, iterblosc_constant)
{
data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
data->func = const_;
data->expr_str = "2";

int8_t ndim = 2;
int64_t shape[] = {100, 40};
int64_t cshape[] = {50, 20};
int64_t bshape[] = {15, 20};

INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, false, NULL));
}


INA_TEST_FIXTURE(expression_eval_int32, iterblosc_superchunk)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = expr_;
    data->expr_str = "(x - 3) * (x - 1) * (x + 4)";

    int8_t ndim = 2;
    int64_t shape[] = {100, 40};
    int64_t cshape[] = {50, 20};
    int64_t bshape[] = {15, 20};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, false, "arr.iarr"));
}


INA_TEST_FIXTURE(expression_eval_int32, iterblosc2_superchunk)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = expr_;
    data->expr_str = "(x - 3) * (x - 1) * (x + 4)";

    int8_t ndim = 3;
    int64_t shape[] = {100, 230, 121};
    int64_t cshape[] = {31, 32, 17};
    int64_t bshape[] = {7, 12, 5};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, "arr.iarr"));
}

static int32_t expr0(const int32_t x)
{
    return (INA_MIN(x, 35));
}

INA_TEST_FIXTURE(expression_eval_int32, iterblosc_superchunk0)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = expr0;
    data->expr_str = "min(x, 35)";

    int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t cshape[] = {25, 25};
    int64_t bshape[] = {10, 10};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
}
