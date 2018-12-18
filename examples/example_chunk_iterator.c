/*
 * Copyright INAOS GmbH, Thalwil, 2018.
 * Copyright Francesc Alted, 2018.
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
    printf("Starting iarray...\n");
    iarray_init();

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &ctx);

    iarray_container_t *c_x, *c_out;

    // Create x container
    uint8_t ndim = 3;
    uint64_t shape[] = {10, 10, 10};
    uint64_t pshape[] = {5, 5, 5};

    iarray_dtshape_t dtshape;
    dtshape.ndim = ndim;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    for (int i = 0; i < dtshape.ndim; ++i) {
        dtshape.shape[i] = shape[i];
        dtshape.partshape[i] = pshape[i];
    }

    printf("Initializing c_x container...\n");

    iarray_container_new(ctx, &dtshape, NULL, 0, &c_x);

    printf("Filling c_x with a chunk iterator...\n");

    iarray_itr_chunk_t *I;
    iarray_itr_chunk_new(ctx, c_x, &I);

    for (iarray_itr_chunk_init(I); !iarray_itr_chunk_finished(I); iarray_itr_chunk_next(I)) {

        iarray_itr_chunk_value_t val;
        iarray_itr_chunk_value(I, &val);

        uint64_t chunksize = 1;
        for (int i = 0; i < ndim; ++i) {
            chunksize *= val.shape[i];
        }

        double *chunkbuf = (double *) malloc(chunksize * sizeof(double));

        for (uint64_t i = 0; i < chunksize; ++i) {
            chunkbuf[i] = val.nelem * chunksize + i;
        }

        memcpy(val.pointer, &chunkbuf[0], chunksize * sizeof(double));

        free(chunkbuf);
    }

    printf("Storing data into a buffer...\n");

    uint64_t destsize = 1;
    for (int i = 0; i < ndim; ++i) {
        destsize *= shape[i];
    }

    double *destbuf = (double *) malloc(destsize * sizeof(double));

    iarray_to_buffer(ctx, c_x, destbuf, destsize);

    printf("Printing first 125 elements...\n");

    for (int i = 0; i < 125; ++i) {
        printf(" - Element %d: %.f\n", i, destbuf[i]);
    }

    printf("Destroying iarray...\n");
    iarray_destroy();

    return 0;
}
