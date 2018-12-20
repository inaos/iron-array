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

#define NCHUNKS  10
#define NITEMS_CHUNK (20 * 1000)
#define NELEM (((NCHUNKS - 1) * NITEMS_CHUNK) + 10)
#define NTHREADS 1

static double _poly(const double x)
{
    return (x - 1.35)*(x - 4.45)*(x - 8.5);
}

/* Compute and fill X values in a buffer */
static int _fill_x(double* x)
{
    double incx = 10. / NELEM;

    /* Fill even values between 0 and 10 */
    for (int i = 0; i<NELEM; i++) {
        x[i] = incx * i;
    }
    return 0;
}

/* Compute and fill Y values in a buffer */
static void _fill_y(const double* x, double* y)
{
    for (int i = 0; i < NELEM; i++) {
        y[i] = _poly(x[i]);
    }
}

static ina_rc_t _execute_iarray_eval(iarray_config_t *cfg, const double *buffer_x, const double *buffer_y, size_t buffer_len)
{
    iarray_context_t *ctx;
    iarray_expression_t* e;
    iarray_container_t* c_x;
    iarray_container_t* c_out;

    iarray_dtshape_t shape;
    shape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    shape.ndim = 1;
    shape.shape[0] = NELEM;
    shape.pshape[0] = NITEMS_CHUNK;

    INA_TEST_ASSERT_SUCCEED(iarray_context_new(cfg, &ctx));

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &shape, buffer_x, buffer_len, NULL, 0, &c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &shape, NULL, 0, &c_out));

    INA_TEST_ASSERT_SUCCEED(iarray_expr_new(ctx, &e));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind(e, "x", c_x));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_compile(e, "(x - 1.35) * (x - 4.45) * (x - 8.5)"));
    INA_TEST_ASSERT_SUCCEED(iarray_eval(e, c_out));

    INA_TEST_ASSERT_SUCCEED(_iarray_test_container_dbl_buffer_cmp(ctx, c_out, buffer_y, buffer_len));

    iarray_expr_free(ctx, &e);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_context_free(&ctx);

    return INA_SUCCESS;
}

INA_TEST_DATA(expression_eval)
{
    size_t buf_len;
    double *buffer_x;
    double *buffer_y;
    iarray_config_t cfg;
};

INA_TEST_SETUP(expression_eval)
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
    _fill_y(data->buffer_x, data->buffer_y);
}

INA_TEST_TEARDOWN(expression_eval)
{
    ina_mem_free(data->buffer_x);
    ina_mem_free(data->buffer_y);

    iarray_destroy();
}

INA_TEST_FIXTURE(expression_eval, chunk1)
{
    data->cfg.flags |= IARRAY_EXPR_EVAL_CHUNK;
 
    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, data->buffer_x, data->buffer_y, data->buf_len));
}

INA_TEST_FIXTURE(expression_eval, block1)
{
    data->cfg.flags |= IARRAY_EXPR_EVAL_BLOCK;
   
    INA_TEST_ASSERT_SUCCEED(_execute_iarray_eval(&data->cfg, data->buffer_x, data->buffer_y, data->buf_len));
}
