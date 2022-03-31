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


int main(void)
{
    // The initialization is commented out on purpose to trigger a compilation issue
    // iarray_init();

    int8_t ndim = 1;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int64_t shape[] = {400 * 1000};
    int64_t cshape[] = {200 * 1000};
    int64_t bshape[] = {16 * 1000};
    ina_rc_t rc;

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_level = 5;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.max_num_threads = 1;
    cfg.eval_method = IARRAY_EVAL_METHOD_AUTO;
    iarray_context_t *ctx;
    //IARRAY_FAIL_IF_ERROR(iarray_context_new(&cfg, &ctx));
    // Call iarray_context_new without any protection to make this crash heavyly.
    iarray_context_new(&cfg, &ctx);

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

    iarray_container_t *data;
    IARRAY_FAIL_IF_ERROR(iarray_linspace(ctx, &dtshape, 0, 1, &store, &data));

    iarray_expression_t* e;
    iarray_expr_new(ctx, dtshape.dtype, &e);

    iarray_expr_bind(e, "x", data);
    iarray_expr_bind_out_properties(e, &dtshape, &store);
    char* expr_str = "(sin(x) - 3.2) * (cos(x) + 1.2)";
    iarray_expr_compile(e, expr_str);

    iarray_container_t* res1;
    INA_TEST_ASSERT_SUCCEED(iarray_empty(ctx, &dtshape, &store, &res1));
    iarray_eval(e, &res1);

    iarray_iter_read_block_t *iter;
    iarray_iter_read_block_value_t val;
    IARRAY_FAIL_IF_ERROR(iarray_iter_read_block_new(ctx, &iter, data, cshape, &val, false));
    while (INA_SUCCEED(iarray_iter_read_block_has_next(iter))) {
        IARRAY_FAIL_IF_ERROR(iarray_iter_read_block_next(iter, NULL, 0));
        for (int64_t i = 0; i < val.block_size; ++i) {
            printf("Next\n");
        }
    }
    iarray_iter_read_block_free(&iter);
    IARRAY_ITER_FINISH();

    rc = INA_SUCCESS;

    iarray_iter_read_block_free(&iter);
    iarray_container_free(ctx, &data);
    iarray_context_free(&ctx);

    return rc;

fail:
    printf("%s\n", ina_err_strerror(ina_err_get_rc()));
    return EXIT_FAILURE;
}
