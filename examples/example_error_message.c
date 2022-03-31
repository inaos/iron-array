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

int main(void) {
    printf("This example should fail. Set 'INAC_TRACE=*' to see the error log.\n");

    IARRAY_RETURN_IF_FAILED(iarray_init());
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;

    iarray_context_t *ctx;
    IARRAY_RETURN_IF_FAILED(iarray_context_new(&cfg, &ctx));

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t ndim = 2;
    int64_t shape[] = {1024, 1024};
    int64_t chunkshape[] = {2048, 128};
    int64_t blockshape[] = {128, 128};

    iarray_dtshape_t dtshape = {0};
    dtshape.ndim = ndim;
    dtshape.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
    }

    iarray_storage_t storage = {0};
    for (int i = 0; i < ndim; ++i) {
        storage.chunkshape[i] = chunkshape[i];
        storage.blockshape[i] = blockshape[i];
    }

    iarray_container_t *container;
    IARRAY_RETURN_IF_FAILED(iarray_linspace(ctx, &dtshape, 0, 1, &storage, &container));

    iarray_container_free(ctx, &container);
    iarray_context_free(&ctx);

    iarray_destroy();

    return INA_SUCCESS;
}
