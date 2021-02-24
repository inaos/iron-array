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

ina_rc_t ia_eval(iarray_context_t *ctx, iarray_container_t **c) {
    iarray_container_t *c1 = *c;
    iarray_container_t *c2 = NULL;

    iarray_expression_t *expr;
    INA_TEST_ASSERT_SUCCEED(iarray_expr_new(ctx, &expr));

    iarray_expr_bind(expr, "x", c1);

    iarray_storage_t storage;
    INA_TEST_ASSERT_SUCCEED(iarray_get_storage(ctx, c1, &storage));
    iarray_dtshape_t dtshape;
    INA_TEST_ASSERT_SUCCEED(iarray_get_dtshape(ctx, c1, &dtshape));
    INA_TEST_ASSERT_SUCCEED(iarray_expr_bind_out_properties(expr, &dtshape, &storage));

    INA_TEST_ASSERT_SUCCEED(iarray_expr_compile(expr, "x - 1"));

    INA_TEST_ASSERT_SUCCEED(iarray_eval(expr, &c2));
    iarray_container_free(ctx, &c1);
    *c = c2;

    return INA_SUCCESS;
}

void eval(int32_t nelem, double *buf) {
    for (int i = 0; i < nelem; ++i) {
        buf[i] = buf[i] - 1;
    }
}

INA_TEST(expression_eval_memcpyed, case1) {

    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_level = 5;
    cfg.max_num_threads = 2;
    cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    cfg.compression_codec = IARRAY_COMPRESSION_BLOSCLZ;
    iarray_context_t *ctx;

    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &ctx));

    int8_t ndim = 2;
    int64_t shape[] = {5, 5};
    int32_t chunkshape[] = {4, 4};
    int32_t blockshape[] = {2, 2};

    iarray_dtshape_t dtshape = {0};
    dtshape.ndim = ndim;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    int8_t typesize = sizeof(double);

    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }

    iarray_storage_t storage = {0};
    storage.backend = IARRAY_STORAGE_BLOSC;
    for (int i = 0; i < ndim; ++i) {
        storage.chunkshape[i] = chunkshape[i];
        storage.blockshape[i] = blockshape[i];
    }

    iarray_container_t *c;
    INA_TEST_ASSERT_SUCCEED(iarray_ones(ctx, &dtshape, &storage, 0, &c));

    int64_t buf_nbytes = nelem * typesize;
    double *buf = malloc(buf_nbytes);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c, buf, buf_nbytes));
    eval(nelem, buf);

    iarray_context_free(&ctx);
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &ctx));

    INA_TEST_ASSERT_SUCCEED(ia_eval(ctx, &c));

    double *buf2 = malloc(buf_nbytes);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c, buf2, buf_nbytes));

    for (int i = 0; i < nelem; ++i) {
        INA_TEST_ASSERT_EQUAL_FLOATING(buf[i], buf2[i]);
    }

    iarray_container_free(ctx, &c);
    iarray_context_free(&ctx);

    iarray_destroy();
}