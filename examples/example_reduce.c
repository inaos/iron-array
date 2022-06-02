/*
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <libiarray/iarray.h>
#include "iarray_private.h"


int main(void) {
    iarray_init();
    ina_stopwatch_t *w;

    INA_STOPWATCH_NEW(-1, -1, &w);


    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_level = 5;
    cfg.compression_codec = IARRAY_COMPRESSION_BLOSCLZ;

    cfg.max_num_threads = 1;
    iarray_context_new(&cfg, &ctx);


    int64_t shape[] = {8, 8};
    int8_t ndim = 2;
    int8_t naxis = 1;
    int8_t axis[] = {1,0};
    iarray_reduce_func_t func = IARRAY_REDUCE_NAN_VAR;
    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_FLOAT;
    dtshape.ndim = ndim;

    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
    }

    int32_t xchunkshape[] = {4, 4};
    int32_t xblockshape[] = {2, 2};

    iarray_storage_t xstorage;
    xstorage.contiguous = false;
    xstorage.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    iarray_container_remove(xstorage.urlpath);
    iarray_container_t *c_x;
    IARRAY_RETURN_IF_FAILED(iarray_arange(ctx, &dtshape, 0, 1, &xstorage, &c_x));

    int32_t outchunkshape[] = {4, 4};
    int32_t outblockshape[] = {2, 2};

    iarray_storage_t outstorage;
    outstorage.contiguous = false;
    outstorage.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        outstorage.chunkshape[i] = outchunkshape[i];
        outstorage.blockshape[i] = outblockshape[i];
    }

    blosc_timestamp_t t0, t1;
    iarray_container_t *c_out;

    blosc_set_timestamp(&t0);
    IARRAY_RETURN_IF_FAILED(iarray_reduce_multi(ctx, c_x, func, naxis, axis, &outstorage, &c_out));
    blosc_set_timestamp(&t1);

    printf("time 2: %f \n", blosc_elapsed_secs(t0, t1));

    uint8_t *buff = malloc(c_out->catarr->nitems * c_out->catarr->itemsize);
    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_out, buff, c_out->catarr->nitems * c_out->catarr->itemsize));

    for (int i = 0; i < c_out->catarr->nitems; ++i) {
        printf(" %.10f ", ((float *) buff)[i]);
    }
    printf("\n");
    free(buff);

    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);
    iarray_destroy();
    return 0;
}
