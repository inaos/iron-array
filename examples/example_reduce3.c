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


int8_t _first_divisor(int64_t n) {
    for (int i = 2; i < ceil(sqrt(n)); ++i)
        if (n % i == 0)
            return i;
    return n;
}


void _compute_block_cluster(int8_t nthreads,
                            int8_t ndim,
                            int64_t *shape,
                            int64_t *chunkshape,
                            int64_t *cluster_shape) {
    int8_t index[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < ndim; ++i) {
        index[i] = i;
        cluster_shape[i] = 1;
    }

    // Bubble sort
    for (int i = 0; i < ndim; ++i) {
        for (int j = 0; j < ndim - i - 1; ++j) {
            if (shape[index[j]] < shape[index[j + 1]]) {
                int8_t temp = index[j];
                index[j] = index[j + 1];
                index[j + 1] = temp;
            }
        }
    }

    // Asign threads to each dimension
    int8_t unused_threads = nthreads;
    for (int i = 0; i < ndim && unused_threads > 0; ++i) {
        cluster_shape[index[i]] = (int64_t) unused_threads;
        while (cluster_shape[index[i]] * chunkshape[index[i]] > shape[index[i]]) {
            int8_t div = _first_divisor(cluster_shape[index[i]]);
            cluster_shape[index[i]] = cluster_shape[index[i]] / div;
        }
        unused_threads /= cluster_shape[index[i]];
        cluster_shape[index[i]] *= chunkshape[index[i]];
    }
}


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

    cfg.max_num_threads = 16;
    iarray_context_new(&cfg, &ctx);

    int8_t ndiim = 2;
    int64_t shape1[] = {1000, 1500};
    int64_t cshape1[] = {100, 100};
    int64_t cshape2[] = {0, 0};

    _compute_block_cluster(cfg.max_num_threads, ndiim, shape1, cshape1, cshape2);

    for (int i = 0; i < ndiim; ++i) {
        printf(" %lld ", cshape2[i]);
    }
    printf("\n");

    int64_t shape[] = {30000, 30000};
    int8_t ndim = 2;
    int8_t typesize = sizeof(double);
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
    xstorage.enforce_frame = false;
    xstorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    iarray_container_t *c_x;
    // IARRAY_RETURN_IF_FAILED(iarray_arange(ctx, &dtshape, 0, nelem, 1, &xstorage, 0, &c_x));

    int32_t outchunkshape[] = {0};
    int32_t outblockshape[] = {0};

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
    // IARRAY_RETURN_IF_FAILED(iarray_reduce2(ctx, c_x, IARRAY_REDUCE_SUM, axis, &c_out));
    blosc_set_timestamp(&t1);
    printf("time: %f \n", blosc_elapsed_secs(t0, t1));
    // iarray_container_free(ctx, &c_out);

    blosc_set_timestamp(&t0);
    // IARRAY_RETURN_IF_FAILED(iarray_reduce(ctx, c_x, IARRAY_REDUCE_SUM, axis, &c_out));
    blosc_set_timestamp(&t1);
    // iarray_container_free(ctx, &c_out);

    // iarray_container_free(ctx, &c_x);

    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);

    return 0;
}
