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


int main(void)
{
    iarray_init();

    int8_t ndim = 2;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int64_t shape[] = {100, 100};
    int64_t cshape[] = {20, 20};
    int64_t bshape[] = {5, 5};
    ina_rc_t rc;

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_t *ctx;
    IARRAY_FAIL_IF_ERROR(iarray_context_new(&cfg, &ctx));

    iarray_dtshape_t dtshape;
    dtshape.ndim = ndim;
    dtshape.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
    }

    iarray_storage_t store;
    store.contiguous = false;
    store.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        store.chunkshape[i] = cshape[i];
        store.blockshape[i] = bshape[i];
    }
    iarray_container_t *cont;
    IARRAY_FAIL_IF_ERROR(iarray_empty(ctx, &dtshape, &store, &cont));


    iarray_iter_write_t *iter_w;
    iarray_iter_write_value_t val_w;
    IARRAY_FAIL_IF_ERROR(iarray_iter_write_new(ctx, &iter_w, cont, &val_w));

    while (INA_SUCCEED(iarray_iter_write_has_next(iter_w))) {
        IARRAY_FAIL_IF_ERROR(iarray_iter_write_next(iter_w));
        ((double *) val_w.elem_pointer)[0] = (double) val_w.elem_flat_index;
        printf("%.lld - ", val_w.elem_flat_index);
    }
    printf("\n");
    iarray_iter_write_free(&iter_w);
    IARRAY_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    iarray_iter_read_block_t *iter;
    iarray_iter_read_block_value_t val;
    IARRAY_FAIL_IF(iarray_iter_read_block_new(ctx, &iter, cont, bshape, &val, false));
    while (INA_SUCCEED(iarray_iter_read_block_has_next(iter))) {
        IARRAY_FAIL_IF(iarray_iter_read_block_next(iter, NULL, 0));
        for (int64_t i = 0; i < val.block_size; ++i) {
            double value = ((double *) val.block_pointer)[i];
            printf("%.f - ", value);
        }
        printf("\n");
    }
    iarray_iter_read_block_free(&iter);
    IARRAY_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    iarray_iter_write_free(&iter_w);
    iarray_iter_read_block_free(&iter);
    iarray_container_free(ctx, &cont);
    iarray_context_free(&ctx);
    iarray_destroy();

    return (int)rc;
}
