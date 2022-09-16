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
    cfg.compression_level = 5;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;

    cfg.max_num_threads = 4;
    iarray_context_new(&cfg, &ctx);

    int64_t shape[] = {100, 100};
    int8_t ndim = 2;

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape.ndim = ndim;

    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
    }

    int32_t xchunkshape[] = {50, 50};
    int32_t xblockshape[] = {10, 10};

    iarray_storage_t xstorage;
    xstorage.contiguous = false;
    xstorage.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    iarray_container_t *x;
    IARRAY_RETURN_IF_FAILED(iarray_arange(ctx, &dtshape, 0, 1, &xstorage, &x));

    uint8_t *cframe;
    int64_t cframe_len;
    bool needs_free;

    IARRAY_RETURN_IF_FAILED(iarray_to_cframe(ctx, x, &cframe, &cframe_len, &needs_free));

    iarray_container_t *y;
    IARRAY_RETURN_IF_FAILED(iarray_from_cframe(ctx, cframe, cframe_len, false, &y));

    IARRAY_RETURN_IF_FAILED(iarray_container_almost_equal(x, y, 1e-12));

    iarray_container_free(ctx, &x);
    iarray_container_free(ctx, &y);

    iarray_context_free(&ctx);

    return 0;
}
