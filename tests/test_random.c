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

#include <src/iarray_private.h>
#include <libiarray/iarray.h>


static ina_rc_t test_rand(iarray_context_t *ctx, iarray_random_ctx_t *rnd_ctx,
                          bool xcontiguous, char* xurlpath, char* yurlpath,
                          ina_rc_t (*random_fun)(iarray_context_t*, iarray_dtshape_t*,
                                                 iarray_random_ctx_t*, iarray_storage_t*, iarray_container_t**))
{

    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_container_open(ctx, yurlpath, &c_y));

    iarray_dtshape_t xdtshape;
    iarray_get_dtshape(ctx, c_y, &xdtshape);

    iarray_storage_t xstorage;
    iarray_get_storage(ctx, c_y, &xstorage);
    xstorage.urlpath = xurlpath;
    xstorage.contiguous = xcontiguous;
    blosc2_remove_urlpath(xstorage.urlpath);

    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(random_fun(ctx, &xdtshape, rnd_ctx, &xstorage, &c_x));


    bool res = false;

    INA_TEST_ASSERT_SUCCEED(iarray_random_kstest(ctx, c_x, c_y, &res));

    if (!res) {
        return INA_ERROR(INA_ERR_FAILED);
    }

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    blosc2_remove_urlpath(xstorage.urlpath);

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
            data->ctx, 123, IARRAY_RANDOM_RNG_MRG32K3A, &data->rnd_ctx));
}

INA_TEST_TEARDOWN(random_mt) {
    iarray_random_ctx_free(data->ctx, &data->rnd_ctx);
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(random_mt, rand) {

    char *urlpath = "test_rand_float64.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, false, "xarr.iarr", urlpath,
                                      &iarray_random_rand));
}

INA_TEST_FIXTURE(random_mt, rand_f) {

    char* urlpath = "test_rand_float32.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, true, "xarr.iarr", urlpath,
                                      &iarray_random_rand));
}

INA_TEST_FIXTURE(random_mt, randn) {

    char* urlpath = "test_randn_float64.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, false, NULL, urlpath,
                                      &iarray_random_randn));
}

INA_TEST_FIXTURE(random_mt, randn_f) {

    char* urlpath = "test_randn_float32.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, true, "xarr.iarr", urlpath,
                                      &iarray_random_randn));
}


INA_TEST_FIXTURE(random_mt, beta) {


    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_ALPHA, 3.);
    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 4.);

    char* urlpath = "test_beta_float64_a3_b4.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, true, NULL, urlpath,
                                      &iarray_random_beta));
}

INA_TEST_FIXTURE(random_mt, beta_f) {

    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_ALPHA, 0.1);
    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 5.0);

    char* urlpath = "test_beta_float32_a0.1_b5.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, false, NULL, urlpath,
                                      &iarray_random_beta));
}

INA_TEST_FIXTURE(random_mt, lognormal) {

    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 3.);
    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 4.);

    char* urlpath = "test_lognormal_float64_mean3_sigma4.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, false, "xarr.iarr", urlpath,
                                      &iarray_random_lognormal));
}

INA_TEST_FIXTURE(random_mt, lognormal_f) {

    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 0.1);
    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 5.);

    char* urlpath = "test_lognormal_float32_mean0.1_sigma5.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, false, "xarr.iarr", urlpath,
                                      &iarray_random_lognormal));
}

INA_TEST_FIXTURE(random_mt, exponential) {

    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 3.0);

    char* urlpath = "test_exponential_float64_scale3.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, true, NULL, urlpath,
                                      &iarray_random_exponential));
}

INA_TEST_FIXTURE(random_mt, exponential_f) {

    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_BETA, 0.1);

    char* urlpath = "test_exponential_float32_scale0.1.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, true, NULL, urlpath,
                                      &iarray_random_exponential));
}

INA_TEST_FIXTURE(random_mt, uniform) {

    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_A, -3.);
    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_B, 5.);

    char* urlpath = "test_uniform_float64_low-3_high5.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, false, NULL, urlpath,
                                      &iarray_random_uniform));
}

INA_TEST_FIXTURE(random_mt, uniform_f) {

    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_A, -0.1);
    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_B, 0.2);

    char* urlpath = "test_uniform_float32_low-0.1_high0.2.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, true, "xarr.iarr", urlpath,
                                      &iarray_random_uniform));
}

INA_TEST_FIXTURE(random_mt, normal) {

    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 3.);
    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 5.);

    char* urlpath = "test_normal_float64_loc3_scale5.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, false, "xarr.iarr", urlpath,
                                      &iarray_random_normal));
}

INA_TEST_FIXTURE(random_mt, normal_f) {

    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 0.1);
    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 0.2);

    char* urlpath = "test_normal_float32_loc0.1_scale0.2.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, true, NULL, urlpath,
                                      &iarray_random_normal));
}

INA_TEST_FIXTURE(random_mt, binomial) {

    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_M, 3.);
    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_P, 0.7);

    char* urlpath = "test_binomial_int32_n3_p0.7.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, false, NULL, urlpath,
                                      &iarray_random_binomial));
}


INA_TEST_FIXTURE(random_mt, poisson) {

    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_LAMBDA, 3.0);

    char* urlpath = "test_poisson_int32_lam3.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, false, "xarr.iarr", urlpath, &iarray_random_poisson));
}


INA_TEST_FIXTURE(random_mt, bernouilli) {

    iarray_random_dist_set_param(data->rnd_ctx, IARRAY_RANDOM_DIST_PARAM_P, 0.7);

    char* urlpath = "test_binomial_int32_n1_p0.7.iarray";

    INA_TEST_ASSERT_SUCCEED(test_rand(data->ctx, data->rnd_ctx, true, NULL, urlpath, &iarray_random_bernoulli));
}


