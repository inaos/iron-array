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

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.btune = false;
    cfg.max_num_threads = 1;

    iarray_context_t *ctx;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &ctx));

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t xndim = 2;
    int64_t shape[] = {10, 10};
    int64_t cshape[] = {4, 6};
    int64_t bshape[] = {3, 3};

    //Define iarray container x
    iarray_dtshape_t dtshape_a;
    dtshape_a.ndim = xndim;
    dtshape_a.dtype = dtype;
    for (int i = 0; i < dtshape_a.ndim; ++i) {
        dtshape_a.shape[i] = shape[i];
    }

    iarray_storage_t storage_a;
    storage_a.urlpath = NULL;
    storage_a.contiguous = true;
    for (int i = 0; i < dtshape_a.ndim; ++i) {
        storage_a.chunkshape[i] = cshape[i];
        storage_a.blockshape[i] = bshape[i];
    }
    int64_t nelem = 1;
    for (int i = 0; i < xndim; ++i) {
        nelem *= shape[i];
    }

    iarray_container_t *a;
    INA_RETURN_IF_FAILED(iarray_tri(ctx, &dtshape_a, -1, &storage_a, &a));

    iarray_split_container_t *split;
    INA_RETURN_IF_FAILED(iarray_split_new(ctx, a, &split));

    iarray_container_t *b;
    iarray_storage_t storage = {0};
    memcpy(&storage, &storage_a, sizeof(iarray_storage_t));
    storage.urlpath = NULL;
    storage.contiguous = true;

    INA_RETURN_IF_FAILED(iarray_concatenate(ctx, split, &storage, &b));
    iarray_split_free(ctx, &split);

    int64_t size = nelem * (int64_t)  sizeof(double);

    double *buf = ina_mem_alloc(size);

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, b, buf, size));

    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            printf("%8.4f", buf[j + i * shape[1]]);
        }
        printf("\n");
    }

    INA_MEM_FREE_SAFE(buf);

    iarray_container_free(ctx, &a);
    iarray_container_free(ctx, &b);

    iarray_context_free(&ctx);

    iarray_destroy();

    return INA_SUCCESS;
}