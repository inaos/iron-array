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
#include <math.h>


double fexpr(double x, double y, double z) {
    return sin(x) * sin(y) + cos(z);
}


int main(void) {
    iarray_init();
    ina_stopwatch_t *w;

    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    char *expr = "sin(x) * sin(y) + cos(z)";

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_level = 0;
    cfg.eval_flags = IARRAY_EVAL_METHOD_ITERBLOSC;

    cfg.max_num_threads = 4;
    iarray_context_new(&cfg, &ctx);


    int64_t shape[] = {20 * 1000 * 1000};
    int8_t ndim = 1;
    int8_t typesize = sizeof(double);

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape.ndim = ndim;

    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }

    int32_t xchunkshape[] = {4000 * 1000};
    int32_t xblockshape[] = {32 * 1000};

    iarray_storage_t xstorage;
    xstorage.backend = IARRAY_STORAGE_BLOSC;
    xstorage.enforce_frame = false;
    xstorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    int32_t ychunkshape[] = {4000 * 1000};
    int32_t yblockshape[] = {32 * 1000};

    iarray_storage_t ystorage;
    ystorage.backend = IARRAY_STORAGE_BLOSC;
    ystorage.enforce_frame = false;
    ystorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        ystorage.chunkshape[i] = ychunkshape[i];
        ystorage.blockshape[i] = yblockshape[i];
    }

    int32_t zchunkshape[] = {4000 * 1000};
    int32_t zblockshape[] = {32 * 1000};
    iarray_storage_t zstorage;
    zstorage.backend = IARRAY_STORAGE_BLOSC;
    zstorage.enforce_frame = false;
    zstorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        zstorage.chunkshape[i] = zchunkshape[i];
        zstorage.blockshape[i] = zblockshape[i];
    }

    int32_t outchunkshape[] = {4000 * 1000};
    int32_t outblockshape[] = {32 * 1000};

    iarray_storage_t outstorage;
    outstorage.backend = IARRAY_STORAGE_BLOSC;
    outstorage.enforce_frame = false;
    outstorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        outstorage.chunkshape[i] = outchunkshape[i];
        outstorage.blockshape[i] = outblockshape[i];
    }

    iarray_container_t *c_x;
    iarray_container_t *c_y;
    iarray_container_t *c_z;

    iarray_linspace(ctx, &dtshape, nelem, 0, 1, &xstorage, 0, &c_x);
    iarray_linspace(ctx, &dtshape, nelem, 0, 1, &ystorage, 0, &c_y);
    iarray_linspace(ctx, &dtshape, nelem, 0, 1, &zstorage, 0, &c_z);


    iarray_expression_t *e;
    iarray_expr_new(ctx, &e);
    iarray_expr_bind(e, "x", c_x);
    iarray_expr_bind(e, "y", c_y);
    iarray_expr_bind(e, "z", c_z);

    iarray_expr_bind_out_properties(e, &dtshape, &outstorage);

    iarray_expr_compile(e, expr);


    iarray_container_t *c_out;

    int nrep = 5;
    INA_STOPWATCH_START(w);
    for (int i = 0; i < nrep; ++i) {
        IARRAY_RETURN_IF_FAILED(iarray_eval(e, &c_out));
    }
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    int64_t nbytes = nelem * typesize;

    printf("Time for eval expression: %.3g s, %.1f GB/s\n",
           elapsed_sec, nbytes * nrep / (elapsed_sec * (1u << 20u)));

    uint64_t b_size = nelem * typesize;
    uint8_t *b_x = ina_mem_alloc(b_size);
    uint8_t *b_y = ina_mem_alloc(b_size);
    uint8_t *b_z = ina_mem_alloc(b_size);

    uint8_t *b_out = ina_mem_alloc(b_size);

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_x, b_x, b_size));
    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_y, b_y, b_size));
    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_z, b_z, b_size));

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_out, b_out, b_size));

    for (int i = 1; i < nelem; ++i) {
        double d1 = ((double *) b_out)[i];
        double d2 = fexpr(((double *) b_x)[i],
                          ((double *) b_y)[i],
                          ((double *) b_z)[i]);


        double rerr = fabs((d1 - d2) / d1);
        if (rerr > 1e-15) {
            printf("ERROR at [%d]!\n", i);
            return -1;
        }
    }

    iarray_expr_free(ctx, &e);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_context_free(&ctx);

    INA_MEM_FREE_SAFE(b_x);
    INA_MEM_FREE_SAFE(b_y);
    INA_MEM_FREE_SAFE(b_z);

    INA_MEM_FREE_SAFE(b_out);

    INA_STOPWATCH_FREE(&w);

    return 0;
}
