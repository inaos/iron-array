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

#define NCHUNKS  10
#define NITEMS_CHUNK (20 * 1000)
#define NELEM (NCHUNKS * NITEMS_CHUNK + 1)
#define NTHREADS 2


int main()
{
    iarray_init();

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_new(&cfg, &ctx);

    iarray_dtshape_t shape;
    shape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    shape.ndim = 1;
    shape.shape[0] = NELEM;
    shape.pshape[0] = NITEMS_CHUNK;

    iarray_container_t* c_x;
    iarray_linspace(ctx, &shape, NELEM, 0., 1., NULL, 0, &c_x);
    iarray_container_t* c_out;
    iarray_container_new(ctx, &shape, NULL, 0, &c_out);

    iarray_expression_t* e;
    iarray_expr_new(ctx, &e);
    iarray_expr_bind(e, "x", c_x);
    iarray_expr_compile(e, "(x - 1.35) * (x - 4.45) * (x - 8.5)");
    iarray_eval(e, c_out);

    // Print some values of the outcome
    size_t buf_len = sizeof(double) * NELEM;
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
