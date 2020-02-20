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


int main()
{
    iarray_init();

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_new(&cfg, &ctx);

    iarray_dtshape_t shape;
    shape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    shape.ndim = 2;
    shape.shape[0] = 10000; shape.shape[1] = 2000;
    shape.pshape[0] = 1000; shape.pshape[1] = 200;
    int nelem = shape.shape[0] * shape.shape[1];

    iarray_store_properties_t store;
    store.backend = IARRAY_STORAGE_BLOSC;
    store.enforce_frame = false;
    store.filename = NULL;
    
    iarray_container_t* c_x;
    iarray_linspace(ctx, &shape, nelem, 0., 10., &store, 0, &c_x);
    iarray_container_t* c_out;
    iarray_container_new(ctx, &shape, &store, 0, &c_out);

    iarray_expression_t* e;
    iarray_expr_new(ctx, &e);
    iarray_expr_bind(e, "x", c_x);
    iarray_expr_bind_out(e, c_out);
    iarray_expr_compile(e, "(x - 1.35) * (x - 4.45) * (x - 8.5)");
    iarray_eval(e);

    // Print some values of the outcome
    size_t buf_len = sizeof(double) * nelem;
    double *buff_out = malloc(buf_len);
    iarray_to_buffer(ctx, c_out, buff_out, buf_len);

    printf("First 10 elements of outcome: ");
    for (int i = 0; i < 10; i++) {
        printf("%.3f, ", buff_out[i]);
    }

    iarray_expr_free(ctx, &e);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_context_free(&ctx);
    free(buff_out);

    return 0;
}
