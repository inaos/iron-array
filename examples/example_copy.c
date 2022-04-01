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

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_level = 0;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;

    cfg.max_num_threads = 4;
    iarray_context_new(&cfg, &ctx);

    int64_t shape[] = {2000, 200, 100};
    int8_t ndim = 3;

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape.ndim = ndim;

    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }

    int32_t xchunkshape[] = {500, 50, 10};
    int32_t xblockshape[] = {50, 20, 5};

    iarray_storage_t xstorage;
    xstorage.contiguous = false;
    xstorage.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    iarray_container_t *c_x;
    IARRAY_RETURN_IF_FAILED(iarray_zeros(ctx, &dtshape, &xstorage, &c_x));

    iarray_container_t *out;
    IARRAY_RETURN_IF_FAILED(iarray_copy(ctx, c_x, false, &xstorage, &out));

    cfg.compression_codec = 5;

    iarray_container_t *out2;
    IARRAY_RETURN_IF_FAILED(iarray_copy(ctx, c_x, false, &xstorage, &out2));

    IARRAY_RETURN_IF_FAILED(iarray_container_almost_equal(out, out2, 1e-6));
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &out);
    iarray_container_free(ctx, &out2);

    iarray_context_free(&ctx);

    return 0;
}
