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


static ina_rc_t test_rand(iarray_context_t *ctx, iarray_random_ctx_t *rnd_ctx,
                          iarray_data_type_t dtype, int8_t ndim, const int64_t *shape,
                          const int64_t *pshape, iarray_store_properties_t store_y,
                          ina_rc_t (*random_fun)(iarray_context_t*, iarray_dtshape_t*,
                              iarray_random_ctx_t*, iarray_store_properties_t*, int, iarray_container_t**))
{

    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        xdtshape.pshape[i] = pshape[i];
        size *= shape[i];
    }

    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(random_fun(ctx, &xdtshape, rnd_ctx, NULL, 0, &c_x));

    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_from_file(ctx, &store_y, &c_y));

    bool res = false;

    INA_TEST_ASSERT_SUCCEED(iarray_random_kstest(ctx, c_x, c_y, &res));

    if (!res) {
        return INA_ERROR(INA_ERR_FAILED);
    }

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);

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

INA_TEST_FIXTURE(random_mt, rand) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_store_properties_t store_y;
    store_y.id = "test_rand.iarray";


    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_rand));
}

INA_TEST_FIXTURE(random_mt, rand_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_store_properties_t store_y;
    store_y.id = "test_rand_f.iarray";


    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_rand));
}

INA_TEST_FIXTURE(random_mt, randn) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_store_properties_t store_y;
    store_y.id = "test_randn.iarray";


    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_randn));
}

INA_TEST_FIXTURE(random_mt, randn_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_store_properties_t store_y;
    store_y.id = "test_randn_f.iarray";


    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_randn));
}

INA_TEST_FIXTURE(random_mt, beta) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_ALPHA, 2.0);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 4.0);

    iarray_store_properties_t store_y;
    store_y.id = "test_beta_2_4.iarray";


    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_beta));
}

INA_TEST_FIXTURE(random_mt, beta_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_ALPHA, 4.0);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 5.0);

    iarray_store_properties_t store_y;
    store_y.id = "test_beta_f_4_5.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_beta));
}

INA_TEST_FIXTURE(random_mt, lognormal) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 0.0);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 0.4);

    iarray_store_properties_t store_y;
    store_y.id = "test_lognormal_0_04.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_lognormal));
}

INA_TEST_FIXTURE(random_mt, lognormal_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 4.0);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 0.7);

    iarray_store_properties_t store_y;
    store_y.id = "test_lognormal_f_4_07.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_lognormal));
}

INA_TEST_FIXTURE(random_mt, exponential) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 6.0);

    iarray_store_properties_t store_y;
    store_y.id = "test_exponential_6.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_exponential));
}

INA_TEST_FIXTURE(random_mt, exponential_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 0.5);


    iarray_store_properties_t store_y;
    store_y.id = "test_exponential_f_05.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_exponential));
}

INA_TEST_FIXTURE(random_mt, uniform) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_A, -1.0);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_B, 4.0);

    iarray_store_properties_t store_y;
    store_y.id = "test_uniform_-1_4.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_uniform));
}

INA_TEST_FIXTURE(random_mt, uniform_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_A, 0.3);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_B, 0.5);

    iarray_store_properties_t store_y;
    store_y.id = "test_uniform_f_03_05.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_uniform));
}

INA_TEST_FIXTURE(random_mt, normal) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, -2.0);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 4.0);

    iarray_store_properties_t store_y;
    store_y.id = "test_normal_-2_4.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_normal));
}

INA_TEST_FIXTURE(random_mt, normal_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {100};

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 3);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 0.5);

    iarray_store_properties_t store_y;
    store_y.id = "test_normal_f_3_05.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, dtype, ndim, shape, pshape, store_y,
                                      &iarray_random_normal));
}
