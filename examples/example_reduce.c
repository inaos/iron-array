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
    ina_stopwatch_t *w;


    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);


    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_level = 9;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;

    cfg.max_num_threads = 1;
    iarray_context_new(&cfg, &ctx);


    int64_t shape[] = {100, 100, 100};
    int8_t ndim = 3;
    int8_t naxis = 3;
    int8_t axis[] = {1, 1, 0};
    iarray_reduce_func_t func = IARRAY_REDUCE_SUM;

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape.ndim = ndim;

    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }

    int32_t xchunkshape[] = {40, 30, 40};
    int32_t xblockshape[] = {21, 10, 14};

    iarray_storage_t xstorage;
    xstorage.backend = IARRAY_STORAGE_BLOSC;
    xstorage.enforce_frame = false;
    xstorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    iarray_container_t *c_x;
    IARRAY_RETURN_IF_FAILED(iarray_ones(ctx, &dtshape, &xstorage, 0, &c_x));

    int32_t outchunkshape[] = {40};
    int32_t outblockshape[] = {14};

    iarray_storage_t outstorage;
    outstorage.backend = IARRAY_STORAGE_BLOSC;
    outstorage.enforce_frame = false;
    outstorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        outstorage.chunkshape[i] = outchunkshape[i];
        outstorage.blockshape[i] = outblockshape[i];
    }

    blosc_timestamp_t t0, t1;
    iarray_container_t *c_out;
    double *buff;

    blosc_set_timestamp(&t0);
    IARRAY_RETURN_IF_FAILED(iarray_reduce_multi(ctx, c_x, func, naxis, axis, &outstorage, &c_out));

    blosc_set_timestamp(&t1);
    printf("time 1: %f \n", blosc_elapsed_secs(t0, t1));
    buff = (double *) malloc(c_out->catarr->nitems * c_out->catarr->itemsize);
    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_out, buff, c_out->catarr->nitems * c_out->catarr->itemsize));
    for (int i = 0; i < c_out->catarr->nitems; ++i) {
        printf(" %f ", buff[i]);
    }
    printf("\n");
    free(buff);
    iarray_container_free(ctx, &c_out);


    iarray_container_free(ctx, &c_x);

    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);

    return 0;
}
