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
    INA_TEST_ASSERT_SUCCEED(iarray_container_open(ctx, filename, &c_y));

    iarray_dtshape_t xdtshape;
    iarray_get_dtshape(ctx, c_y, &xdtshape);

    iarray_storage_t xstorage;
    iarray_get_storage(ctx, c_y, &xstorage);
    
    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(random_fun(ctx, &xdtshape, rnd_ctx, &xstorage, 0, &c_x));


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
        data->ctx, 1234, IARRAY_RANDOM_RNG_MERSENNE_TWISTER, &data->rnd_ctx));
}

INA_TEST_TEARDOWN(random_mt) {
    iarray_random_ctx_free(data->ctx, &data->rnd_ctx);
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(random_mt, rand) {

    char *filename = "test_rand_float64.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_rand));
}

INA_TEST_FIXTURE(random_mt, rand_f) {

    char* filename = "test_rand_float32.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_rand));
}

INA_TEST_FIXTURE(random_mt, randn) {

    char* filename = "test_randn_float64.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_randn));
}

INA_TEST_FIXTURE(random_mt, randn_f) {

    char* filename = "test_randn_float32.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_randn));
}


INA_TEST_FIXTURE(random_mt, beta) {


    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_ALPHA, 3.);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 4.);

    char* filename = "test_beta_float64_a3_b4.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_beta));
}

INA_TEST_FIXTURE(random_mt, beta_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_ALPHA, 0.1f);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 5.0f);

    char* filename = "test_beta_float32_a0.1_b5.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_beta));
}

INA_TEST_FIXTURE(random_mt, lognormal) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 3.);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 4.);

    char* filename = "test_lognormal_float64_mean3_sigma4.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_lognormal));
}

INA_TEST_FIXTURE(random_mt, lognormal_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 0.1f);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 5.f);

    char* filename = "test_lognormal_float32_mean0.1_sigma5.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_lognormal));
}

INA_TEST_FIXTURE(random_mt, exponential) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 3.0f);
    
    char* filename = "test_exponential_float64_scale3.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_exponential));
}

INA_TEST_FIXTURE(random_mt, exponential_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 0.1f);

    char* filename = "test_exponential_float32_scale0.1.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_exponential));
}

INA_TEST_FIXTURE(random_mt, uniform) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_A, -3.);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_B, 5.);

    char* filename = "test_uniform_float64_low-3_high5.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_uniform));
}

INA_TEST_FIXTURE(random_mt, uniform_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_A, -0.1f);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_B, 0.2f);

    char* filename = "test_uniform_float32_low-0.1_high0.2.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_uniform));
}

INA_TEST_FIXTURE(random_mt, normal) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 3.);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 5.);

    char* filename = "test_normal_float64_loc3_scale5.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_normal));
}

INA_TEST_FIXTURE(random_mt, normal_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 0.1f);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 0.2f);

    char* filename = "test_normal_float32_loc0.1_scale0.2.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_normal));
}

INA_TEST_FIXTURE(random_mt, binomial) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_M, 3.f);
    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_P, 0.7f);

    char* filename = "test_binomial_float64_n3_p0.7.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_binomial));
}

INA_TEST_FIXTURE(random_mt, binomial_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_M, 10.f);
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_P, 0.01f);

    char* filename = "test_binomial_float32_n10_p0.01.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename,
                                      &iarray_random_binomial));
}


INA_TEST_FIXTURE(random_mt, poisson) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_LAMBDA, 3.0f);

    char* filename = "test_poisson_float64_lam3.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename, &iarray_random_poisson));
}


INA_TEST_FIXTURE(random_mt, poisson_f) {

    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_LAMBDA, 0.001f);

    char* filename = "test_poisson_float32_lam0.001.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename, &iarray_random_poisson));
}


INA_TEST_FIXTURE(random_mt, bernouilli) {

    iarray_random_dist_set_param_double(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_P, 0.7f);

    char* filename = "test_binomial_float64_n1_p0.7.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename, &iarray_random_bernoulli));
}


INA_TEST_FIXTURE(random_mt, bernoulli_f) {

    iarray_random_ctx_free(data->ctx, &data->rnd_ctx);

    INA_TEST_ASSERT_SUCCEED(iarray_random_ctx_new(
        data->ctx, 777, IARRAY_RANDOM_RNG_SOBOL, &data->rnd_ctx));
    iarray_random_dist_set_param_float(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_P, 0.01f);

    char* filename = "test_binomial_float32_n1_p0.01.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, filename, &iarray_random_bernoulli));
}
