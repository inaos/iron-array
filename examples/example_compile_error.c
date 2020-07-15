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
#include <iarray_private.h>


int main(void)
{
    iarray_init();

    int8_t ndim = 1;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int64_t shape[] = {400 * 1000};
    int64_t pshape[] = {200 * 1000};
    int64_t bshape[] = {16 * 1000};
    ina_rc_t rc;

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_level = 5;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.max_num_threads = 1;
    cfg.eval_flags = IARRAY_EVAL_METHOD_ITERBLOSC2 | (IARRAY_EVAL_ENGINE_COMPILER << 3);
    iarray_context_t *ctx;
    IARRAY_FAIL_IF_ERROR(iarray_context_new(&cfg, &ctx));

    iarray_dtshape_t dtshape;
    dtshape.ndim = ndim;
    dtshape.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
    }

    iarray_storage_t store;
    store.backend = IARRAY_STORAGE_BLOSC;
    store.enforce_frame = false;
    store.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        store.chunkshape[i] = pshape[i];
        store.blockshape[i] = bshape[i];
    }

    iarray_container_t *data;
    IARRAY_FAIL_IF_ERROR(iarray_linspace(ctx, &dtshape, shape[0], 0, 1, &store, 0, &data));

    iarray_expression_t* e;
    iarray_expr_new(ctx, &e);

    iarray_expr_bind(e, "x", data);
    iarray_expr_bind_out_properties(e, &dtshape, &store);
    char* expr_str = "(sin(x) - 3.2) * (cos(x) + 1.2)";
    iarray_expr_compile(e, expr_str);

    iarray_container_t* res1;
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &dtshape, &store, 0, &res1));
    iarray_eval(e, &res1);


    iarray_iter_read_block_t *iter;
    iarray_iter_read_block_value_t val;
    IARRAY_FAIL_IF(iarray_iter_read_block_new(ctx, &iter, data, pshape, &val, false));
    while (INA_SUCCEED(iarray_iter_read_block_has_next(iter))) {
        IARRAY_FAIL_IF(iarray_iter_read_block_next(iter, NULL, 0));
        for (int64_t i = 0; i < val.block_size; ++i) {
            //printf("Next\n");
        }
    }
    iarray_iter_read_block_free(&iter);
    IARRAY_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    iarray_iter_read_block_free(&iter);
    iarray_container_free(ctx, &data);
    iarray_context_free(&ctx);

    return rc;
}
