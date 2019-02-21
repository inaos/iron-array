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

#include <tests/iarray_test.h>

static ina_rc_t test_rand(iarray_context_t *ctx, iarray_random_ctx_t *rnd_ctx, iarray_data_type_t dtype, 
                          uint8_t ndim, const uint64_t *shape, const uint64_t *pshape) {

    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    uint64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        xdtshape.pshape[i] = pshape[i];
        size *= shape[i];
    }

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_random_rand(ctx, &xdtshape, rnd_ctx, NULL, 0, &c_x));

    // Assert iterator reading it

    iarray_iter_read_t *iter;
    iarray_iter_read_new(ctx, c_x, &iter);
    for (iarray_iter_read_init(iter); !iarray_iter_read_finished(iter); iarray_iter_read_next(iter)) {

        iarray_iter_read_value_t val;
        iarray_iter_read_value(iter, &val);

        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            double v = *((double*)val.pointer);
            INA_TEST_ASSERT_TRUE(v > .0 && v < 1.);
        } 
        else {
            float v = *((float*)val.pointer);
            INA_TEST_ASSERT_TRUE(v > .0 && v < 1.);
        }
    }

    iarray_iter_read_free(iter);
    iarray_container_free(ctx, &c_x);

    return INA_SUCCESS;
}

INA_TEST_DATA(random_mt) {
    iarray_context_t *ctx;
    iarray_random_ctx_t *rnd_ctx;
};

INA_TEST_SETUP(random_mt) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.eval_flags = IARRAY_EXPR_EVAL_CHUNK;

    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));

    INA_TEST_ASSERT_SUCCEED(iarray_random_ctx_new(
        data->ctx, 777, IARRAY_RANDOM_RNG_MERSENNE_TWISTER, &data->rnd_ctx));
}

INA_TEST_TEARDOWN(random_mt) {
    iarray_random_ctx_free(data->ctx, &data->rnd_ctx);
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(random_mt, rand_double) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    uint8_t ndim = 2;
    uint64_t shape[] = {223, 456};
    uint64_t pshape[] = { 31, 43 };

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape));
}

