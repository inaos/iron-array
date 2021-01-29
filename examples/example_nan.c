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

ina_rc_t ia_eval(iarray_context_t *ctx, iarray_container_t **c) {
    iarray_container_t *c1 = *c;
    iarray_container_t *c2 = NULL;

    iarray_expression_t *expr;
    IARRAY_RETURN_IF_FAILED(iarray_expr_new(ctx, &expr));

    iarray_expr_bind(expr, "x", c1);

    iarray_storage_t storage;
    IARRAY_RETURN_IF_FAILED(iarray_get_storage(ctx, c1, &storage));
    iarray_dtshape_t dtshape;
    IARRAY_RETURN_IF_FAILED(iarray_get_dtshape(ctx, c1, &dtshape));
    IARRAY_RETURN_IF_FAILED(iarray_expr_bind_out_properties(expr, &dtshape, &storage));

    IARRAY_RETURN_IF_FAILED(iarray_expr_compile(expr, "x / 100"));

    IARRAY_RETURN_IF_FAILED(iarray_eval(expr, &c2));
    iarray_container_free(ctx, &c1);
    *c = c2;
    
    return INA_SUCCESS;
}

void eval(int32_t nelem, double *buf) {
    for (int i = 0; i < nelem; ++i) {
        buf[i] = buf[i] / 100;
    }
}

int main() {
    
    iarray_init();
    
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_t *ctx;
    
    IARRAY_RETURN_IF_FAILED(iarray_context_new(&cfg, &ctx));
    
    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int32_t chunkshape[] = {4, 5};
    int32_t blockshape[] = {2, 2};
    
    iarray_dtshape_t dtshape = {0};
    dtshape.ndim = ndim;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t typesize = sizeof(double);
    
    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }
    
    iarray_storage_t storage = {0};
    storage.backend = IARRAY_STORAGE_BLOSC;
    for (int i = 0; i < ndim; ++i) {
        storage.chunkshape[i] = chunkshape[i];
        storage.blockshape[i] = blockshape[i];
    }
    
    iarray_random_ctx_t *rctx;
    IARRAY_RETURN_IF_FAILED(iarray_random_ctx_new(ctx, 0, IARRAY_RANDOM_RNG_MERSENNE_TWISTER, &rctx));
    
    iarray_container_t *c;
    IARRAY_RETURN_IF_FAILED(iarray_random_dist_set_param_double(rctx, IARRAY_RANDOM_DIST_PARAM_A, -1));
    IARRAY_RETURN_IF_FAILED(iarray_random_dist_set_param_double(rctx, IARRAY_RANDOM_DIST_PARAM_B, 1));
    IARRAY_RETURN_IF_FAILED(iarray_random_uniform(ctx, &dtshape, rctx, &storage, 0, &c));
    
    int64_t buf_nbytes = nelem * typesize;
    double *buf = malloc(buf_nbytes);
    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c, buf, buf_nbytes));
    printf("%f\n", buf[0]);
    eval(nelem, buf);
    IARRAY_RETURN_IF_FAILED(ia_eval(ctx, &c));

    double *buf2 = malloc(buf_nbytes);
    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c, buf2, buf_nbytes));

    for (int i = 0; i < nelem; ++i) {
        printf("%3d:\t %.8f - %3.8f\n", i, buf[i], buf2[i]);
    }

    iarray_container_free(ctx, &c);
    iarray_random_ctx_free(ctx, &rctx);
    iarray_context_free(&ctx);
    
    iarray_destroy();
    
    return 0;
}