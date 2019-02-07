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
#include <src/iarray_private.h>

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
    uint8_t xndim = 3;
    uint64_t xshape[] = {100, 100, 100};
    uint64_t xpshape[] = {10, 10, 10};

    iarray_dtshape_t xdtshape;
    xdtshape.ndim = xndim;
    xdtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    for (int i = 0; i < xdtshape.ndim; ++i) {
        xdtshape.shape[i] = xshape[i];
        xdtshape.partshape[i] = xpshape[i];
    }

    printf("Initializing c_x container...\n");
    printf("- c_x shape: ");
    for (int i = 0; i < xdtshape.ndim; ++i) {
        printf("%d ", (int)xdtshape.shape[i]);
    }
    printf("\n");

    iarray_fill_double(ctx, &xdtshape, 3.14, NULL, 0, &c_x);

    // Create out container (empty)
    uint8_t outndim = 3;
    uint64_t start[] = {10, 20, 30};
    uint64_t stop[] = {40, 21, 80};
    uint64_t outpshape[] = {5, 1, 20};

    printf("Defining start and stop for slicing...\n");
    printf("- start: ");
    for (int i = 0; i < outndim; ++i) {
        printf("%d ", (int)start[i]);
    }
    printf("\n");
    printf("- stop: ");
    for (int i = 0; i < outndim; ++i) {
        printf("%d ", (int)stop[i]);
    }
    printf("\n");

    // Slicing c_x into c_out
    printf("Slicing c_x into c_out container...\n");
    iarray_get_slice(ctx, c_x, start, stop, outpshape, NULL, 0, &c_out);

    printf("- c_out shape: ");
    for (int i = 0; i < c_out->dtshape->ndim; ++i) {
        printf("%d ", (int)c_out->dtshape->shape[i]);
    }
    printf("\n");

    //Squeezing c_out
    printf("Squeezing c_out...\n");
    iarray_squeeze(ctx, c_out);

    printf("- c_out shape: ");
    for (int i = 0; i < c_out->dtshape->ndim; ++i) {
        printf("%d ", (int)c_out->dtshape->shape[i]);
    }
    printf("\n");

    printf("Destroying iarray...\n");
    iarray_destroy();

    return 0;
}
