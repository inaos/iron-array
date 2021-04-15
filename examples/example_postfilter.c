/*
 * Copyright ironArray S.L. 2021
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray S.L.
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <libiarray/iarray.h>
#include <math.h>


double fexpr(double x) {
    return cos(x);
}


int main(void) {
    iarray_init();
    ina_stopwatch_t *w;

    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    char *expr = "cos(x)";

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_level = 9;
    cfg.max_num_threads = 8;
    iarray_context_new(&cfg, &ctx);

    int64_t shape[] = {50 * 1000 * 1000};
    int32_t chunkshape[] = {500 * 1000};
    int32_t blockshape[] = {32 * 1000};
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

    iarray_storage_t xstorage = {0};
    xstorage.backend = IARRAY_STORAGE_BLOSC;
    xstorage.enforce_frame = false;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = chunkshape[i];
        xstorage.blockshape[i] = blockshape[i];
    }

    iarray_container_t *c_x;
    IARRAY_RETURN_IF_FAILED(iarray_linspace(ctx, &dtshape, 0, 1, &xstorage, 0, &c_x));
    //IARRAY_RETURN_IF_FAILED(iarray_zeros(ctx, &dtshape, &xstorage, 0, &c_x));

    uint64_t buffer_size = nelem * typesize;
    uint8_t *b_x = ina_mem_alloc(buffer_size);
    uint8_t *a_x = ina_mem_alloc(buffer_size);
    // Convert into a buffer *before* attaching the postfilter
    iarray_to_buffer(ctx, c_x, b_x, buffer_size);

    iarray_expression_t *e;
    iarray_expr_new(ctx, &e);
    iarray_expr_bind(e, "x", c_x);
    iarray_expr_bind_out_properties(e, &dtshape, NULL);
    iarray_expr_compile(e, expr);
    iarray_expr_register_as_postfilter(e, c_x);

    int nrep = 3;
    INA_STOPWATCH_START(w);
    for (int i = 0; i < nrep; ++i) {
        // Convert into a buffer *after* postfilter
        IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_x, a_x, buffer_size));
    }
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    int64_t nbytes = nelem * typesize;

    printf("Time for eval expression: %.3g s, %.1f GB/s\n",
           elapsed_sec / nrep, (1. * nbytes) * nrep / (elapsed_sec * (1u << 30u)));

    for (int i = 0; i < nelem; ++i) {
        double d1 = ((double *) a_x)[i];
        double d2 = fexpr(((double *) b_x)[i]);

        //printf("%d: %f, %f\n", i, d1, d2);
        double rerr = fabs((d1 - d2) / d1);
        if (rerr > 1e-15) {
            printf("ERROR at [%d]!  %f != %f\n", i, d1, d2);
            return -1;
        }
    }

    iarray_expr_free(ctx, &e);
    iarray_container_free(ctx, &c_x);
    iarray_context_free(&ctx);

    INA_MEM_FREE_SAFE(b_x);
    INA_MEM_FREE_SAFE(a_x);

    INA_STOPWATCH_FREE(&w);

    return 0;
}
