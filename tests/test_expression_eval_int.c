/*
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <libiarray/iarray.h>
#include <tests/iarray_test.h>
#include <src/iarray_private.h>


#define NTHREADS 1

typedef enum test_func {
    CONST_INT64_ = -1,
    EXPR_INT64_ = 1,
    EXPR_MIN_INT64 = 2,
    EXPR_ABS_INT64 = 3,
    EXPR_MAX_INT64 = 4,
    CONST_INT32_ = -2,
    EXPR_INT32_ = 6,
    EXPR_MIN_INT32 = 7,
    EXPR_ABS_INT32 = 8,
    EXPR_MAX_INT32 = 9,
    CONST_INT16_ = -3,
    EXPR_INT16_ = 11,
    EXPR_MIN_INT16 = 12,
    EXPR_ABS_INT16 = 13,
    EXPR_MAX_INT16 = 14,
    CONST_INT8_ = -4,
    EXPR_INT8_ = 16,
    EXPR_MIN_INT8 = 17,
    EXPR_ABS_INT8 = 18,
    EXPR_MAX_INT8 = 19,
    CONST_BOOL_ = -5,
    EXPR_MIN_BOOL = 20,
    EXPR_MAX_BOOL = 21,
} test_func;

/* INT64 functions */
/* Special case for a constant function */
static int64_t const_int64_(const int64_t x)
{
    return (int64_t)(2 - (x - x));
}

static int64_t expr_int64_(const int64_t x)
{
    return (x - 3) * (x - 1) * (x + 4);
}
static int64_t expr_min_int64(const int64_t x)
{
    return (INA_MIN(x, 35));
}

static int64_t expr_abs_int64(const int64_t x)
{
    return (int64_t)(abs((int)x) - 35);
}

static int64_t expr_max_int64(const int64_t x)
{
    return (INA_MAX(x, 35));
}

/* INT32 functions */
/* Special case for a constant function */
static int32_t const_int32_(const int32_t x)
{
    return 2 - (x - x);
}

static int32_t expr_int32_(const int32_t x)
{
    return (x - 3) * (x - 1) * (x + 4);
}

static int32_t expr_min_int32(const int32_t x)
{
    return (INA_MIN(x, 35));
}

static int32_t expr_abs_int32(const int32_t x)
{
    return abs(x) - 35;
}

static int32_t expr_max_int32(const int32_t x)
{
    return (INA_MAX(x, 35));
}

/* INT16 functions */
/* Special case for a constant function */
static int16_t const_int16_(const int16_t x)
{
    return (int16_t)(2 - (x - x));
}

static int16_t expr_int16_(const int16_t x)
{
    return (int16_t)((x - 3) * (x - 1) * (x + 4));
}

static int16_t expr_min_int16(const int16_t x)
{
    return (int16_t)(INA_MIN(x, 35));
}

static int16_t expr_abs_int16(const int16_t x)
{
    return (int16_t)(abs(x) - 35);
}

static int16_t expr_max_int16(const int16_t x)
{
    return (int16_t)(INA_MAX(x, 35));
}

/* INT8 functions */
/* Special case for a constant function */
static int8_t const_int8_(const int8_t x)
{
    return (int8_t)(2 - (x - x));
}

static int8_t expr_int8_(const int8_t x)
{
    return (int8_t)((x - 3) * (x - 1) * (x + 4));
}

static int8_t expr_min_int8(const int8_t x)
{
    return (int8_t)(INA_MIN(x, 35));
}

static int8_t expr_abs_int8(const int8_t x)
{
    return (int8_t)(abs(x) - 35);
}

static int8_t expr_max_int8(const int8_t x)
{
    return (int8_t)(INA_MAX(x, 35));
}

/* BOOL functions */
static bool const_bool_(const bool x)
{
    return x;
}

static bool expr_min_bool(const bool x)
{
    return INA_MIN(x, true);
}

static bool expr_max_bool(const bool x)
{
    return INA_MAX(x, false);
}

/* Compute and fill Y values in a buffer */
static void fill_y(const void *x, void *y, int64_t nelem, enum test_func func)
{
    switch (func){
        case CONST_INT64_: {
            int64_t *x_ = (int64_t *) x;
            int64_t *y_ = (int64_t *) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = const_int64_(x_[i]);
            }
            break;
        }
        case EXPR_INT64_: {
            int64_t *x_ = (int64_t *) x;
            int64_t *y_ = (int64_t *) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_int64_(x_[i]);
            }
            break;
        }
        case EXPR_MIN_INT64: {
            int64_t *x_ = (int64_t *) x;
            int64_t *y_ = (int64_t *) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_min_int64(x_[i]);
            }
            break;
        }
        case EXPR_ABS_INT64: {
            int64_t *x_ = (int64_t *) x;
            int64_t *y_ = (int64_t *) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_abs_int64(x_[i]);
            }
            break;
        }
        case EXPR_MAX_INT64: {
            int64_t *x_ = (int64_t *) x;
            int64_t *y_ = (int64_t *) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_max_int64(x_[i]);
            }
            break;
        }
        case CONST_INT32_: {
            int32_t *x_ = (int32_t *) x;
            int32_t *y_ = (int32_t *) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = const_int32_(x_[i]);
            }
            break;
        }
        case EXPR_INT32_: {
            int32_t *x_ = (int32_t *) x;
            int32_t *y_ = (int32_t *) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_int32_(x_[i]);
            }
            break;
        }
        case EXPR_MIN_INT32: {
            int32_t *x_ = (int32_t *) x;
            int32_t *y_ = (int32_t *) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_min_int32(x_[i]);
            }
            break;
        }
        case EXPR_ABS_INT32: {
            int32_t *x_ = (int32_t *) x;
            int32_t *y_ = (int32_t *) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_abs_int32(x_[i]);
            }
            break;
        }
        case EXPR_MAX_INT32: {
            int32_t *x_ = (int32_t *) x;
            int32_t *y_ = (int32_t *) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_max_int32(x_[i]);
            }
            break;
        }
        case CONST_INT16_: {
            int16_t *x_ = (int16_t*) x;
            int16_t *y_ = (int16_t*) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = const_int16_(x_[i]);
            }
            break;
        }
        case EXPR_INT16_: {
            int16_t *x_ = (int16_t*) x;
            int16_t *y_ = (int16_t*) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_int16_(x_[i]);
            }
            break;
        }
        case EXPR_MIN_INT16: {
            int16_t *x_ = (int16_t*) x;
            int16_t *y_ = (int16_t*) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_min_int16(x_[i]);
            }
            break;
        }
        case EXPR_ABS_INT16: {
            int16_t *x_ = (int16_t*) x;
            int16_t *y_ = (int16_t*) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_abs_int16(x_[i]);
            }
            break;
        }
        case EXPR_MAX_INT16: {
            int16_t *x_ = (int16_t*) x;
            int16_t *y_ = (int16_t*) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_max_int16(x_[i]);
            }
            break;
        }
        case CONST_INT8_: {
            int8_t *x_ = (int8_t*) x;
            int8_t *y_ = (int8_t*) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = const_int8_(x_[i]);
            }
            break;
        }
        case EXPR_INT8_: {
            int8_t *x_ = (int8_t*) x;
            int8_t *y_ = (int8_t*) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_int8_(x_[i]);
            }
            break;
        }
        case EXPR_MIN_INT8: {
            int8_t *x_ = (int8_t*) x;
            int8_t *y_ = (int8_t*) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_min_int8(x_[i]);
            }
            break;
        }
        case EXPR_ABS_INT8: {
            int8_t *x_ = (int8_t*) x;
            int8_t *y_ = (int8_t*) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_abs_int8(x_[i]);
            }
            break;
        }
        case EXPR_MAX_INT8: {
            int8_t *x_ = (int8_t*) x;
            int8_t *y_ = (int8_t*) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_max_int8(x_[i]);
            }
            break;
        }
        case CONST_BOOL_: {
            bool *x_ = (bool*) x;
            bool *y_ = (bool*) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = const_bool_(x_[i]);
            }
            break;
        }
        case EXPR_MIN_BOOL: {
            bool *x_ = (bool*) x;
            bool *y_ = (bool*) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_min_bool(x_[i]);
            }
            break;
        }
        case EXPR_MAX_BOOL: {
            bool *x_ = (bool*) x;
            bool *y_ = (bool*) y;
            for (int i = 0; i < nelem; i++) {
                y_[i] = expr_max_bool(x_[i]);
            }
            break;
        }
    }
}


static ina_rc_t execute_iarray_eval(iarray_config_t *cfg, int8_t ndim, const int64_t *shape, const int64_t *cshape,
                                    const int64_t *bshape, enum test_func func, char* expr_str, bool contiguous,
                                    char *urlpath)
{
    iarray_context_t *ctx;
    iarray_expression_t* e;
    iarray_container_t* c_x;
    iarray_container_t* c_out;
    iarray_dtshape_t dtshape;
    switch (func) {
        case CONST_INT64_:
        case EXPR_INT64_:
        case EXPR_MIN_INT64:
        case EXPR_ABS_INT64:
        case EXPR_MAX_INT64:
            dtshape.dtype = IARRAY_DATA_TYPE_INT64;
            dtshape.dtype_size = sizeof(int64_t);
            break;
        case CONST_INT32_:
        case EXPR_INT32_:
        case EXPR_MIN_INT32:
        case EXPR_ABS_INT32:
        case EXPR_MAX_INT32:
            dtshape.dtype = IARRAY_DATA_TYPE_INT32;
            dtshape.dtype_size = sizeof(int32_t);
            break;
        case CONST_INT16_:
        case EXPR_INT16_:
        case EXPR_MIN_INT16:
        case EXPR_ABS_INT16:
        case EXPR_MAX_INT16:
            dtshape.dtype = IARRAY_DATA_TYPE_INT16;
            dtshape.dtype_size = sizeof(int16_t);
            break;
        case CONST_INT8_:
        case EXPR_INT8_:
        case EXPR_MIN_INT8:
        case EXPR_ABS_INT8:
        case EXPR_MAX_INT8:
            dtshape.dtype = IARRAY_DATA_TYPE_INT8;
            dtshape.dtype_size = sizeof(int8_t);
            break;
        case CONST_BOOL_:
        case EXPR_MIN_BOOL:
        case EXPR_MAX_BOOL:
            dtshape.dtype = IARRAY_DATA_TYPE_BOOL;
            dtshape.dtype_size = sizeof(bool);
            break;
        default:
            return INA_ERR_EXCEEDED;
    }
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

    void *buffer_x = ina_mem_alloc(nelem * dtshape.dtype_size);
    void *buffer_y = ina_mem_alloc(nelem * dtshape.dtype_size);

    fill_buf(dtshape.dtype, buffer_x, nelem);
    fill_y(buffer_x, buffer_y, nelem, func);

    INA_TEST_ASSERT_SUCCEED(iarray_context_new(cfg, &ctx));
    blosc2_remove_urlpath(store.urlpath);
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &dtshape, buffer_x, nelem * dtshape.dtype_size, &store, &c_x));

    INA_TEST_ASSERT_SUCCEED(iarray_expr_new(ctx, dtshape.dtype, &e));
    if (func >= 0) {
        INA_TEST_ASSERT_SUCCEED(iarray_expr_bind(e, "x", c_x));
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
    INA_TEST_ASSERT_SUCCEED(test_double_buffer_cmp(ctx, c_out, buffer_y, nelem * dtshape.dtype_size, 5e-15, 5e-14));

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

INA_TEST_DATA(expression_eval)
{
    iarray_config_t cfg;
    enum test_func func;
    char *expr_str;
};

INA_TEST_SETUP(expression_eval)
{
    iarray_init();

    data->cfg = IARRAY_CONFIG_DEFAULTS;
    data->cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    data->cfg.compression_level = 9;
    data->cfg.max_num_threads = NTHREADS;
}

INA_TEST_TEARDOWN(expression_eval)
{
    INA_UNUSED(data);
    iarray_destroy();
}


INA_TEST_FIXTURE(expression_eval, int32_iterblosc_constant)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = CONST_INT32_;
    data->expr_str = "2";

    int8_t ndim = 2;
    int64_t shape[] = {100, 40};
    int64_t cshape[] = {50, 20};
    int64_t bshape[] = {15, 20};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, false, NULL));
}


INA_TEST_FIXTURE(expression_eval, int32_iterblosc_superchunk)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = EXPR_INT32_;
    data->expr_str = "(x - 3) * (x - 1) * (x + 4)";

    int8_t ndim = 2;
    int64_t shape[] = {100, 40};
    int64_t cshape[] = {50, 20};
    int64_t bshape[] = {15, 20};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, false, "arr.iarr"));
}


INA_TEST_FIXTURE(expression_eval, int32_iterblosc2_superchunk)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = EXPR_INT32_;
    data->expr_str = "(x - 3) * (x - 1) * (x + 4)";

    int8_t ndim = 3;
    int64_t shape[] = {100, 230, 121};
    int64_t cshape[] = {31, 32, 17};
    int64_t bshape[] = {7, 12, 5};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, "arr.iarr"));
}


INA_TEST_FIXTURE(expression_eval, int16_iterblosc_constant)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = CONST_INT16_;
    data->expr_str = "2";

    int8_t ndim = 2;
    int64_t shape[] = {10, 40};
    int64_t cshape[] = {5, 20};
    int64_t bshape[] = {5, 20};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, false, NULL));
}


INA_TEST_FIXTURE(expression_eval, int16_iterblosc_superchunk)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = EXPR_INT16_;
    data->expr_str = "(x - 3) * (x - 1) * (x + 4)";

    int8_t ndim = 2;
    int64_t shape[] = {100, 4};
    int64_t cshape[] = {50, 2};
    int64_t bshape[] = {15, 2};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, false, "arr.iarr"));
}


INA_TEST_FIXTURE(expression_eval, int16_iterblosc2_superchunk)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = EXPR_INT16_;
    data->expr_str = "(x - 3) * (x - 1) * (x + 4)";

    int8_t ndim = 3;
    int64_t shape[] = {10, 23, 121};
    int64_t cshape[] = {10, 3, 17};
    int64_t bshape[] = {7, 2, 5};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, "arr.iarr"));
}


INA_TEST_FIXTURE(expression_eval, int64_iterblosc_constant)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = CONST_INT64_;
    data->expr_str = "2";

    int8_t ndim = 2;
    int64_t shape[] = {30, 40};
    int64_t cshape[] = {20, 20};
    int64_t bshape[] = {15, 20};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, false, NULL));
}


INA_TEST_FIXTURE(expression_eval, int64_iterblosc_superchunk)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = EXPR_INT64_;
    data->expr_str = "(x - 3) * (x - 1) * (x + 4)";

    int8_t ndim = 2;
    int64_t shape[] = {40, 40};
    int64_t cshape[] = {25, 20};
    int64_t bshape[] = {15, 20};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, false, "arr.iarr"));
}


INA_TEST_FIXTURE(expression_eval, int64_iterblosc2_superchunk)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = EXPR_INT64_;
    data->expr_str = "(x - 3) * (x - 1) * (x + 4)";

    int8_t ndim = 3;
    int64_t shape[] = {50, 23, 12};
    int64_t cshape[] = {13, 3, 7};
    int64_t bshape[] = {7, 2, 5};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, "arr.iarr"));
}


INA_TEST_FIXTURE(expression_eval, int8_iterblosc_constant)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = CONST_INT8_;
    data->expr_str = "2";

    int8_t ndim = 2;
    int64_t shape[] = {50, 2};
    int64_t cshape[] = {25, 2};
    int64_t bshape[] = {10, 2};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, false, NULL));
}


INA_TEST_FIXTURE(expression_eval, int8_iterblosc2_superchunk)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = EXPR_INT8_;
    data->expr_str = "(x - 3) * (x - 1) * (x + 4)";

    int8_t ndim = 3;
    int64_t shape[] = {10, 5, 2};
    int64_t cshape[] = {5, 3, 2};
    int64_t bshape[] = {3, 3, 2};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, "arr.iarr"));
}


INA_TEST_FIXTURE(expression_eval, bool_iterblosc_constant)
{
    data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    data->func = CONST_BOOL_;
    data->expr_str = "1";

    int8_t ndim = 2;
    int64_t shape[] = {100, 40};
    int64_t cshape[] = {50, 20};
    int64_t bshape[] = {15, 20};

    INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, false, NULL));
}


#ifndef INA_OS_WINDOWS
    /* Temporaly avoid these tests since we cannot use LLVM13 with windows */
    INA_TEST_FIXTURE(expression_eval, int32_iterblosc_superchunk_min)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_MIN_INT32;
        data->expr_str = "min(x, 35)";

        int8_t ndim = 2;
        int64_t shape[] = {100, 100};
        int64_t cshape[] = {25, 25};
        int64_t bshape[] = {10, 10};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }


    INA_TEST_FIXTURE(expression_eval, int32_iterblosc_superchunk_max)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_MAX_INT32;
        data->expr_str = "max(x, 35)";

        int8_t ndim = 2;
        int64_t shape[] = {100, 100};
        int64_t cshape[] = {25, 25};
        int64_t bshape[] = {10, 10};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }


    INA_TEST_FIXTURE(expression_eval, int32_iterblosc_superchunk_abs)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_ABS_INT32;
        data->expr_str = "abs(x) - 35";

        int8_t ndim = 2;
        int64_t shape[] = {100, 100};
        int64_t cshape[] = {25, 25};
        int64_t bshape[] = {10, 10};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }


    INA_TEST_FIXTURE(expression_eval, int16_iterblosc_superchunk_min)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_MIN_INT16;
        data->expr_str = "min(x, 35)";

        int8_t ndim = 2;
        int64_t shape[] = {10, 10};
        int64_t cshape[] = {5, 5};
        int64_t bshape[] = {3, 3};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }


    INA_TEST_FIXTURE(expression_eval, int16_iterblosc_superchunk_max)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_MAX_INT32;
        data->expr_str = "max(x, 35)";

        int8_t ndim = 2;
        int64_t shape[] = {50, 50};
        int64_t cshape[] = {25, 25};
        int64_t bshape[] = {10, 10};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }


    INA_TEST_FIXTURE(expression_eval, int64_iterblosc_superchunk_min)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_MIN_INT64;
        data->expr_str = "min(x, 35)";

        int8_t ndim = 2;
        int64_t shape[] = {50, 40};
        int64_t cshape[] = {25, 25};
        int64_t bshape[] = {10, 10};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }


    INA_TEST_FIXTURE(expression_eval, int16_iterblosc_superchunk_abs)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_ABS_INT16;
        data->expr_str = "abs(x) - 35";

        int8_t ndim = 2;
        int64_t shape[] = {50, 50};
        int64_t cshape[] = {25, 25};
        int64_t bshape[] = {10, 10};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }


    INA_TEST_FIXTURE(expression_eval, int64_iterblosc_superchunk_max)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_MAX_INT64;
        data->expr_str = "max(x, 35)";

        int8_t ndim = 2;
        int64_t shape[] = {40, 40};
        int64_t cshape[] = {25, 25};
        int64_t bshape[] = {10, 10};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }


    INA_TEST_FIXTURE(expression_eval, int64_iterblosc_superchunk_abs)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_ABS_INT64;
        data->expr_str = "abs(x) - 35";

        int8_t ndim = 2;
        int64_t shape[] = {100, 100};
        int64_t cshape[] = {25, 25};
        int64_t bshape[] = {10, 10};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }


    INA_TEST_FIXTURE(expression_eval, int8_iterblosc_superchunk_min)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_MIN_INT8;
        data->expr_str = "min(x, 35)";

        int8_t ndim = 2;
        int64_t shape[] = {50, 2};
        int64_t cshape[] = {25, 2};
        int64_t bshape[] = {10, 2};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }


    INA_TEST_FIXTURE(expression_eval, int8_iterblosc_superchunk_max)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_MAX_INT8;
        data->expr_str = "max(x, 35)";

        int8_t ndim = 2;
        int64_t shape[] = {100, 10};
        int64_t cshape[] = {25, 5};
        int64_t bshape[] = {10, 2};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }


    INA_TEST_FIXTURE(expression_eval, int8_iterblosc_superchunk_abs)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_ABS_INT8;
        data->expr_str = "abs(x) - 35";

        int8_t ndim = 2;
        int64_t shape[] = {50, 2};
        int64_t cshape[] = {25, 2};
        int64_t bshape[] = {10, 2};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }


    INA_TEST_FIXTURE(expression_eval, bool_iterblosc_superchunk_min)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_MIN_BOOL;
        data->expr_str = "min(x, 1)";

        int8_t ndim = 2;
        int64_t shape[] = {100, 100};
        int64_t cshape[] = {25, 25};
        int64_t bshape[] = {10, 10};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }


    INA_TEST_FIXTURE(expression_eval, bool_iterblosc_superchunk_max)
    {
        data->cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        data->func = EXPR_MAX_BOOL;
        data->expr_str = "max(x, 0)";

        int8_t ndim = 2;
        int64_t shape[] = {100, 100};
        int64_t cshape[] = {25, 25};
        int64_t bshape[] = {10, 10};

        INA_TEST_ASSERT_SUCCEED(execute_iarray_eval(&data->cfg, ndim, shape, cshape, bshape, data->func, data->expr_str, true, NULL));
    }
#endif
