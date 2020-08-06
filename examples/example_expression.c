/*
 * Copyright INAOS GmbH, Thalwil, 2019.
 * Copyright Francesc Alted, 2019.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of INAOS GmbH
 * and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <libiarray/iarray.h>


int main(void)
{
    iarray_init();
    ina_stopwatch_t *w;

    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    char *expr = "2*x+1";

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_level = 0;
    cfg.eval_flags = IARRAY_EVAL_METHOD_ITERBLOSC2;
    cfg.max_num_threads = 4;
    iarray_context_new(&cfg, &ctx);


    int64_t shape[] = {4000, 4000};
    int8_t ndim = 2;
    int8_t typesize = sizeof(double);

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape.ndim = ndim;

    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }

    int32_t xchunkshape[] = {500, 500};
    int32_t xblockshape[] = {200, 100};
    iarray_storage_t xstorage;
    xstorage.backend = IARRAY_STORAGE_BLOSC;
    xstorage.enforce_frame = false;
    xstorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    int32_t ychunkshape[] = {400, 800};
    int32_t yblockshape[] = {100, 150};
    iarray_storage_t ystorage;
    ystorage.backend = IARRAY_STORAGE_BLOSC;
    ystorage.enforce_frame = false;
    ystorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        ystorage.chunkshape[i] = ychunkshape[i];
        ystorage.blockshape[i] = yblockshape[i];
    }
    
    iarray_container_t* c_x;
    iarray_container_t* c_y;
    iarray_linspace(ctx, &dtshape, nelem, 2.1, .1, &xstorage, 0, &c_x);
    iarray_linspace(ctx, &dtshape, nelem, 0.1, .1, &xstorage, 0, &c_y);

    iarray_expression_t* e;
    iarray_expr_new(ctx, &e);
    iarray_expr_bind(e, "x", c_x);
    iarray_expr_bind(e, "y", c_y);
    iarray_expr_bind_out_properties(e, &dtshape, &xstorage);

    iarray_expr_compile(e, expr);


    iarray_container_t* c_out;

    INA_STOPWATCH_START(w);
    IARRAY_RETURN_IF_FAILED(iarray_eval(e, &c_out));
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    int64_t nbytes = nelem * typesize;

    printf("Time for eval expression: %.3g s, %.1f GB/s\n",
           elapsed_sec, nbytes / (elapsed_sec * (1u << 20u)));

    iarray_expr_free(ctx, &e);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);

    return 0;
}
