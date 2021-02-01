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

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;

    iarray_context_new(&cfg, &ctx);

    int64_t shape[] = {123};
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

    int32_t xchunkshape[] = {49};
    int32_t xblockshape[] = {20};

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
    IARRAY_RETURN_IF_FAILED(iarray_container_open(ctx, "example_copy.iarr", &x));

    int64_t buflen = nelem * itemsize;
    uint8_t *buf = malloc(buflen);
    printf("TO buffer\n");
    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, x, buf, buflen));

    free(buf);
    iarray_container_free(ctx, &x);
    iarray_context_free(&ctx);

    return 0;
}
