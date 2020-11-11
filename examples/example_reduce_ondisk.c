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
    cfg.compression_level = 9;
    cfg.fp_mantissa_bits = 10;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;

    cfg.max_num_threads = 1;
    iarray_context_new(&cfg, &ctx);


    int64_t shape[] = {2000, 16918};
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

    int32_t xchunkshape[] = {2000, 2000};
    int32_t xblockshape[] = {100, 100};

    iarray_storage_t xstorage;
    xstorage.backend = IARRAY_STORAGE_BLOSC;
    xstorage.filename = "ia_reduce.iarray";
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    iarray_random_ctx_t *rnd_ctx;
    iarray_random_ctx_new(ctx, 0, IARRAY_RANDOM_RNG_MERSENNE_TWISTER, &rnd_ctx);

    iarray_container_t *c_x;
    IARRAY_RETURN_IF_FAILED(iarray_random_dist_set_param_double(rnd_ctx,
                                                                IARRAY_RANDOM_DIST_PARAM_MU, 0));
    IARRAY_RETURN_IF_FAILED(iarray_random_dist_set_param_double(rnd_ctx,
                                                                IARRAY_RANDOM_DIST_PARAM_SIGMA, 1));
    IARRAY_RETURN_IF_FAILED(iarray_random_normal(ctx, &dtshape, rnd_ctx, &xstorage, 0, &c_x));


    iarray_container_t *c_out;
    IARRAY_RETURN_IF_FAILED(iarray_reduce2(ctx, c_x, IARRAY_REDUCE_SUM, axis, &c_out));

    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);

    iarray_random_ctx_free(ctx, &rnd_ctx);
    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);

    return 0;
}
