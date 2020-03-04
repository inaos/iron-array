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

    char *expr = "x+1";

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 3;
    int64_t shape[] = {7, 7, 7};
    int64_t pshape[] = {3, 3, 3};

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.eval_flags = IARRAY_EXPR_EVAL_METHOD_ITERBLOSC2 | (IARRAY_EXPR_EVAL_ENGINE_JUGGERNAUT << 3);
    cfg.blocksize = 0;
    cfg.max_num_threads = 1;
    iarray_context_new(&cfg, &ctx);

    iarray_dtshape_t dtshape;
    dtshape.dtype = dtype;
    dtshape.ndim = ndim;
    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        dtshape.pshape[i] = pshape[i];
        nelem *= shape[i];
    }


    iarray_store_properties_t store;
    store.backend = IARRAY_STORAGE_BLOSC;
    store.enforce_frame = false;
    store.filename = NULL;

    iarray_container_t* c_x;
    iarray_container_t* c_y;
    iarray_arange(ctx, &dtshape, 0, nelem, 1, &store, 0, &c_x);
    iarray_linspace(ctx, &dtshape, nelem, 0.1, .1, &store, 0, &c_y);

    iarray_expression_t* e;
    iarray_expr_new(ctx, &e);
    iarray_expr_bind(e, "x", c_x);
    iarray_expr_bind(e, "y", c_y);
    iarray_expr_bind_out_properties(e, &dtshape, &store);

    iarray_expr_compile(e, expr);

    iarray_container_t* c_out;
    iarray_eval(e, &c_out);

    // Print some values of the outcome
    size_t buf_len = sizeof(double) * nelem;
    double *buff_x = malloc(buf_len);
    iarray_to_buffer(ctx, c_x, buff_x, buf_len);
    double *buff_y = malloc(buf_len);
    iarray_to_buffer(ctx, c_y, buff_y, buf_len);
    double *buff_out = malloc(buf_len);
    iarray_to_buffer(ctx, c_out, buff_out, buf_len);


    bool success = true;
    for (int64_t i = 0; i < nelem; i++) {
        if (buff_out[i] != buff_x[i]+1) {
            printf("ERROR in pos %lld\n", i);
            success = false;
            break;
        }
    }
    if (success) {
      printf("Evaluation of '%s' expression is correct!\n", expr);
    }

    iarray_expr_free(ctx, &e);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_context_free(&ctx);
    free(buff_out);
    free(buff_x);
    free(buff_y);

    return 0;
}
