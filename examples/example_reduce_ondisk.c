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

    cfg.max_num_threads = 4;
    iarray_context_new(&cfg, &ctx);


    int64_t shape[] = {200, 200};
    int8_t ndim = 2;
    int8_t axis = 0;

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape.ndim = ndim;

    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }

    int32_t xchunkshape[] = {200, 200};
    int32_t xblockshape[] = {10, 10};

    iarray_storage_t xstorage;
    xstorage.backend = IARRAY_STORAGE_BLOSC;
    xstorage.filename = "ia_reduce.iarray";
    xstorage.enforce_frame = true;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    iarray_random_ctx_t *rnd_ctx;
    iarray_random_ctx_new(ctx, 0, IARRAY_RANDOM_RNG_MERSENNE_TWISTER, &rnd_ctx);

    iarray_container_t *c_x;
    if (!INA_SUCCEED(iarray_container_open(ctx, "ia_reduce.iarray-dev", &c_x))) {
        IARRAY_RETURN_IF_FAILED(iarray_random_dist_set_param_double(rnd_ctx,
                                                                    IARRAY_RANDOM_DIST_PARAM_MU,
                                                                    0));
        IARRAY_RETURN_IF_FAILED(iarray_random_dist_set_param_double(rnd_ctx,
                                                                    IARRAY_RANDOM_DIST_PARAM_SIGMA,
                                                                    1));
        IARRAY_RETURN_IF_FAILED(iarray_ones(ctx, &dtshape, &xstorage, 0, &c_x));
    }

    int64_t buff_nitems = 1;
    for (int i = 0; i < ndim; ++i) {
        buff_nitems *= shape[i];
    }
    int64_t buff_size = buff_nitems * sizeof(double);

    double *buff = malloc(buff_size);

    iarray_to_buffer(ctx, c_x, buff, buff_size);

    blosc_timestamp_t t0;
    blosc_set_timestamp(&t0);
    iarray_container_t *c_out;
    IARRAY_RETURN_IF_FAILED(iarray_reduce(ctx, c_x, IARRAY_REDUCE_SUM, axis, &c_out));
    blosc_timestamp_t t1;
    blosc_set_timestamp(&t1);
    printf("time: %f s\n", blosc_elapsed_secs(t0, t1));

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_out, buff, buff_size));
    for (int i = 0; i < 100; ++i) {
        printf(" %f ", buff[i]);
    }
    iarray_container_free(ctx, &c_out);

    iarray_container_free(ctx, &c_x);

    iarray_random_ctx_free(ctx, &rnd_ctx);
    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);

    return 0;
}
