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



int main(void) {
    iarray_init();
    ina_stopwatch_t *w;

    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);


    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_level = 0;
    cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;

    cfg.max_num_threads = 4;
    iarray_context_new(&cfg, &ctx);


    int64_t shape[] = {50, 100};
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

    int32_t xchunkshape[] = {25, 50};
    int32_t xblockshape[] = {5, 10};

    iarray_storage_t xstorage;
    xstorage.backend = IARRAY_STORAGE_BLOSC;
    xstorage.enforce_frame = false;
    xstorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }


    iarray_container_t *c_x;

    IARRAY_RETURN_IF_FAILED(iarray_arange(ctx, &dtshape, 0, nelem, 1, &xstorage, 0, &c_x));



    iarray_container_t *c_out;
    IARRAY_RETURN_IF_FAILED(iarray_linalg_transpose(ctx, c_x, true, NULL, &c_out));

    uint64_t b_size = nelem * typesize;
    uint8_t *b_x = ina_mem_alloc(b_size);
    uint8_t *b_out = ina_mem_alloc(b_size);

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_x, b_x, b_size));
    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_out, b_out, b_size));

    for (int i = 1; i < nelem; ++i) {
        double d1 = ((double *) b_out)[i];
        printf("%f\n", d1);
    }

    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);

    INA_MEM_FREE_SAFE(b_x);
    INA_MEM_FREE_SAFE(b_out);

    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);

    return 0;
}
