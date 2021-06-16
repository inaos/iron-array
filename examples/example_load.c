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
#include "iarray_private.h"



int main(void) {

    iarray_init();

    remove("example_copy.iarr");

    ina_stopwatch_t *w;
    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.btune = true;
    iarray_context_new(&cfg, &ctx);

    int64_t shape[] = {1024 * 1024};
    int8_t ndim = 1;

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t itemsize = sizeof(double);
    dtshape.ndim = ndim;

    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }

    int32_t xchunkshape[] = {512 * 1024};
    int32_t xblockshape[] = {128 * 1024};

    iarray_storage_t xstorage;
    xstorage.backend = IARRAY_STORAGE_BLOSC;
    xstorage.enforce_frame = false;
    xstorage.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    iarray_container_t *x;
    IARRAY_RETURN_IF_FAILED(iarray_linspace(ctx, &dtshape, -10, 10, &xstorage, 0, &x));


    IARRAY_RETURN_IF_FAILED(iarray_container_save(ctx, x, "example_copy.iarr"));
    iarray_container_free(ctx, &x);

    INA_STOPWATCH_START(w);
    IARRAY_RETURN_IF_FAILED(iarray_container_load(ctx, "example_copy.iarr", &x));
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time: %.4f s\n", elapsed_sec);

    iarray_container_free(ctx, &x);
    ctx->cfg->compression_level = 9;
    ctx->cfg->compression_codec = IARRAY_COMPRESSION_ZLIB;

    INA_STOPWATCH_START(w);
    IARRAY_RETURN_IF_FAILED(iarray_container_load(ctx, "example_copy.iarr", &x));
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time: %.4f s\n", elapsed_sec);

    int64_t buflen = nelem * itemsize;
    uint8_t *buf = malloc(buflen);

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, x, buf, buflen));

    free(buf);
    iarray_container_free(ctx, &x);
    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);
    return 0;
}
