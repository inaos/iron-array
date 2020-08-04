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

    char *expr = "2*x+y";

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.eval_flags = IARRAY_EVAL_METHOD_ITERBLOSC2;
    cfg.max_num_threads = 1;
    iarray_context_new(&cfg, &ctx);

    int64_t shape[] = {10, 10, 10};
    int8_t ndim = 3;
    int8_t typesize = sizeof(double);

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape.ndim = ndim;
    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }

    int32_t x_chunkshape[] = {5, 10, 5};
    int32_t x_blockshape[] = {5, 5, 5};
    iarray_storage_t x_storage;
    x_storage.backend = IARRAY_STORAGE_BLOSC;
    x_storage.enforce_frame = false;
    x_storage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        x_storage.chunkshape[i] = x_chunkshape[i];
        x_storage.blockshape[i] = x_blockshape[i];
    }

    int32_t y_chunkshape[] = {5, 10, 5};
    int32_t y_blockshape[] = {5, 5, 5};
    iarray_storage_t y_storage;
    y_storage.backend = IARRAY_STORAGE_BLOSC;
    y_storage.enforce_frame = false;
    y_storage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        y_storage.chunkshape[i] = y_chunkshape[i];
        y_storage.blockshape[i] = y_blockshape[i];
    }


    iarray_container_t* c_x;
    iarray_container_t* c_y;
    IARRAY_RETURN_IF_FAILED(iarray_arange(ctx, &dtshape, 0, nelem, 1, &x_storage, 0, &c_x));
    IARRAY_RETURN_IF_FAILED(iarray_arange(ctx, &dtshape, 0, nelem, 1, &y_storage, 0, &c_y));

    iarray_expression_t* e;
    iarray_expr_new(ctx, &e);
    iarray_expr_bind(e, "x", c_x);
    iarray_expr_bind(e, "y", c_y);
    iarray_expr_bind_out_properties(e, &dtshape, &x_storage);

    iarray_expr_compile(e, expr);

    iarray_container_t* c_out;
    iarray_eval(e, &c_out);

    int64_t buffersize = nelem * typesize;
    uint8_t *buffer = ina_mem_alloc(buffersize);

    iarray_to_buffer(ctx, c_out, buffer, buffersize);

    for (int i = 0; i < nelem; ++i) {
        double res1 = ((double *) buffer)[i];
        double res2 = 2 * i + i;
        if (res1 != res2) {
            printf("ERROR!!!!\n");
            return -1;
        }
    }

    iarray_expr_free(ctx, &e);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_context_free(&ctx);

    return 0;
}
