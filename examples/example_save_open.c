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

    int64_t shape[] = {10, 10};
    int8_t ndim = 2;

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape.ndim = ndim;

    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }

    int32_t xchunkshape[] = {8, 8};
    int32_t xblockshape[] = {8, 8};

    iarray_storage_t xstorage;
    xstorage.backend = IARRAY_STORAGE_BLOSC;
    xstorage.enforce_frame = false;
    xstorage.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    iarray_container_t *c_x;
    IARRAY_RETURN_IF_FAILED(iarray_linspace(ctx, &dtshape, -1, 1, &xstorage, 0, &c_x));

    int32_t outchunkshape[] = {5, 5};
    int32_t outblockshape[] = {2, 2};

    iarray_storage_t outstorage;
    outstorage.backend = IARRAY_STORAGE_BLOSC;
    outstorage.enforce_frame = false;
    outstorage.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        outstorage.chunkshape[i] = outchunkshape[i];
        outstorage.blockshape[i] = outblockshape[i];
    }
    
    iarray_container_t *out;
    IARRAY_RETURN_IF_FAILED(iarray_copy(ctx, c_x, false, &outstorage, 0, &out));
    IARRAY_RETURN_IF_FAILED(iarray_container_save(ctx, out, "example_copy.iarr"));
    iarray_container_free(ctx, &out);
    IARRAY_RETURN_IF_FAILED(iarray_container_open(ctx, "example_copy.iarr", &out));


    iarray_container_free(ctx, &out);
    iarray_context_free(&ctx);

    return 0;
}
