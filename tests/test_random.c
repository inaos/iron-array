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
                          char* filename,
                          ina_rc_t (*random_fun)(iarray_context_t*, iarray_dtshape_t*,
                                                 iarray_random_ctx_t*, iarray_storage_t*, int, iarray_container_t**))
{

    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_container_load(ctx, filename, true, &c_y));

    iarray_dtshape_t xdtshape;
    iarray_get_dtshape(ctx, c_y, &xdtshape);

    iarray_storage_t xstore;
    xstore.backend = IARRAY_STORAGE_BLOSC;
    xstore.enforce_frame = false;
    xstore.filename = NULL;
    
    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(random_fun(ctx, &xdtshape, rnd_ctx, &xstore, 0, &c_x));


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
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));

    INA_TEST_ASSERT_SUCCEED(iarray_random_ctx_new(
        data->ctx, 777, IARRAY_RANDOM_RNG_MERSENNE_TWISTER, &data->rnd_ctx));
}

INA_TEST_TEARDOWN(random_mt) {
    iarray_random_ctx_free(data->ctx, &data->rnd_ctx);
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE_SKIP(random_mt, rand) {

    char *filename = "test_rand_d.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_rand));
}

INA_TEST_FIXTURE_SKIP(random_mt, rand_f) {

    char* filename = "test_rand_f.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_rand));
}

INA_TEST_FIXTURE_SKIP(random_mt, randn) {

    char* filename = "test_randn_d.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_randn));
}

INA_TEST_FIXTURE_SKIP(random_mt, randn_f) {

    char* filename = "test_randn_f.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_randn));
}

INA_TEST_FIXTURE_SKIP(random_mt, beta) {


    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_ALPHA, 3.);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 4.);

    char* filename = "test_beta_d_3_4.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_beta));
}

INA_TEST_FIXTURE_SKIP(random_mt, beta_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_ALPHA, 0.1f);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 5.0f);

    char* filename = "test_beta_f_01_5.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_beta));
}

INA_TEST_FIXTURE_SKIP(random_mt, lognormal) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 3.);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 4.);

    char* filename = "test_lognormal_d_3_4.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_lognormal));
}

INA_TEST_FIXTURE_SKIP(random_mt, lognormal_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 0.1f);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 5.f);

    char* filename = "test_lognormal_f_01_5.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_lognormal));
}

INA_TEST_FIXTURE_SKIP(random_mt, exponential) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 3.0f);
    
    char* filename = "test_exponential_d_3.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_exponential));
}

INA_TEST_FIXTURE_SKIP(random_mt, exponential_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 0.1f);

    char* filename = "test_exponential_f_01.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_exponential));
}

INA_TEST_FIXTURE_SKIP(random_mt, uniform) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_A, 3.);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_B, 5.);

    char* filename = "test_uniform_d_3_5.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_uniform));
}

INA_TEST_FIXTURE_SKIP(random_mt, uniform_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_A, 0.1f);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_B, 0.2f);

    char* filename = "test_uniform_f_01_02.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_uniform));
}

INA_TEST_FIXTURE_SKIP(random_mt, normal) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 3.);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 5.);

    char* filename = "test_normal_d_3_5.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_normal));
}

INA_TEST_FIXTURE_SKIP(random_mt, normal_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 0.1f);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 0.2f);

    char* filename = "test_normal_f_01_02.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_normal));
}

INA_TEST_FIXTURE_SKIP(random_mt, binomial) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_M, 3.f);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_P, 0.7f);

    char* filename = "test_binomial_d_3_07.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_binomial));
}

INA_TEST_FIXTURE_SKIP(random_mt, binomial_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_M, 10.f);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_P, 0.01f);

    char* filename = "test_binomial_f_10_001.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_binomial));
}


INA_TEST_FIXTURE_SKIP(random_mt, poisson) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_LAMBDA, 3.0f);

    char* filename = "test_poisson_d_3.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename, &iarray_random_poisson));
}


INA_TEST_FIXTURE_SKIP(random_mt, poisson_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_LAMBDA, 0.001f);

    char* filename = "test_poisson_f_001.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename, &iarray_random_poisson));
}

INA_TEST_FIXTURE_SKIP(random_mt, bernouilli) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_P, 0.7f);

    char* filename = "test_bernoulli_d_07.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename, &iarray_random_bernoulli));
}


INA_TEST_FIXTURE_SKIP(random_mt, bernoulli_f) {

    iarray_random_ctx_free(data->ctx, &data->rnd_ctx);

    INA_TEST_ASSERT_SUCCEED(iarray_random_ctx_new(
        data->ctx, 777, IARRAY_RANDOM_RNG_SOBOL, &data->rnd_ctx));
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_P, 0.01f);

    char* filename = "test_bernoulli_f_001.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename, &iarray_random_bernoulli));
}
