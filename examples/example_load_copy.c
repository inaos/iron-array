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
#include "iarray_private.h"



int main(void) {

    iarray_init();

    blosc2_remove_urlpath("example_load_copy.iarr");

    ina_stopwatch_t *w;
    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    // Activating btune will override compression_level and compression_codec defaults later on
    // cfg.btune = true;
    iarray_context_new(&cfg, &ctx);

    int64_t shape[] = {1024 * 1024};
    int8_t ndim = 1;

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_FLOAT;
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
    xstorage.contiguous = false;
    xstorage.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    iarray_container_t *x;
    IARRAY_RETURN_IF_FAILED(iarray_linspace(ctx, &dtshape, -10, 10, &xstorage, 0, &x));

    float cratio = (float)x->catarr->sc->nbytes / (float)x->catarr->sc->cbytes;
    printf("orig cratio: %.3f\n", cratio);
    printf("orig contiguous: %d\n", x->storage->contiguous);

    iarray_storage_t xstorage2;
    xstorage2.contiguous = true;
    xstorage2.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage2.chunkshape[i] = xchunkshape[i] + 1;
        xstorage2.blockshape[i] = xblockshape[i] + 1;
    }

    iarray_container_t *out;
    IARRAY_RETURN_IF_FAILED(iarray_copy(ctx, x, false, &xstorage2, 0, &out));

    IARRAY_RETURN_IF_FAILED(iarray_container_save(ctx, out, "example_load_copy.iarr"));

    cratio = (float)out->catarr->sc->nbytes / (float)out->catarr->sc->cbytes;
    printf("copy cratio: %.3f\n", cratio);
    printf("copy contiguous: %d\n", out->storage->contiguous);

    iarray_container_free(ctx, &x);
    // Check whether ctx affects the load process (it does not look like it)
    ctx->cfg->compression_level = 0;
    ctx->cfg->compression_codec = IARRAY_COMPRESSION_ZLIB;

    INA_STOPWATCH_START(w);
    IARRAY_RETURN_IF_FAILED(iarray_container_load(ctx, "example_load_copy.iarr", &x));
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time: %.4f s\n", elapsed_sec);

    cratio = (float)x->catarr->sc->nbytes / (float)x->catarr->sc->cbytes;
    printf("load cratio: %.3f\n", cratio);
    printf("load contiguous: %d\n", x->storage->contiguous);

    iarray_container_free(ctx, &x);
    ctx->cfg->compression_level = 0;
    ctx->cfg->compression_codec = IARRAY_COMPRESSION_ZLIB;

    INA_STOPWATCH_START(w);
    IARRAY_RETURN_IF_FAILED(iarray_container_load(ctx, "example_load_copy.iarr", &x));
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time: %.4f s\n", elapsed_sec);

    cratio = (float)x->catarr->sc->nbytes / (float)x->catarr->sc->cbytes;
    printf("load2 cratio: %.3f\n", cratio);
    printf("load2 contiguous: %d\n", x->storage->contiguous);

    int64_t buflen = nelem * itemsize;
    uint8_t *buf = malloc(buflen);

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, x, buf, buflen));

    free(buf);
    iarray_container_free(ctx, &x);
    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);
    return 0;
}
