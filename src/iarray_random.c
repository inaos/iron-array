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
#include "iarray_constructor.h"
#include <mkl_vsl.h>


typedef enum _iarray_random_method_e {
    _IARRAY_RANDOM_METHOD_UNIFORM,
    _IARRAY_RANDOM_METHOD_GAUSSIAN,
    _IARRAY_RANDOM_METHOD_BETA,
    _IARRAY_RANDOM_METHOD_LOGNORMAL,
    _IARRAY_RANDOM_METHOD_EXPONENTIAL,
    // _IARRAY_RANDOM_METHOD_CHISQUARE, // TODO: It is in documentation but not found in mkl.h ; bug?
    _IARRAY_RANDOM_METHOD_BERNOUILLI,
    _IARRAY_RANDOM_METHOD_BINOMIAL,
    _IARRAY_RANDOM_METHOD_POISSON,
} _iarray_random_method_t;

struct iarray_random_ctx_s {
    iarray_random_rng_t rng;
    uint32_t seed;
    VSLStreamStatePtr stream;
    double dparams[IARRAY_RANDOM_DIST_PARAM_SENTINEL];
    float fparams[IARRAY_RANDOM_DIST_PARAM_SENTINEL];
};

INA_API(ina_rc_t) iarray_random_ctx_new(iarray_context_t *ctx,
    uint32_t seed,
    iarray_random_rng_t rng,
    iarray_random_ctx_t **rng_ctx)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(rng_ctx);
    *rng_ctx = (iarray_random_ctx_t*)ina_mem_alloc(sizeof(iarray_random_ctx_t));
    (*rng_ctx)->seed = seed;
    (*rng_ctx)->rng = rng;

    int mkl_rng;
    switch (rng) {
        case IARRAY_RANDOM_RNG_MERSENNE_TWISTER:
            mkl_rng = VSL_BRNG_SFMT19937;
            break;
        case IARRAY_RANDOM_RNG_SOBOL:
            mkl_rng = VSL_BRNG_SOBOL;
            break;
        default:
            IARRAY_TRACE1(iarray.error, "The random generator method is invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RNG_METHOD));
    }

    vslNewStream(&(*rng_ctx)->stream, mkl_rng, seed);

    ina_mem_set((*rng_ctx)->dparams, 0, sizeof(double)*(IARRAY_RANDOM_DIST_PARAM_SENTINEL));
    ina_mem_set((*rng_ctx)->fparams, 0, sizeof(float)*(IARRAY_RANDOM_DIST_PARAM_SENTINEL));

    return INA_SUCCESS;

fail:
    iarray_random_ctx_free(ctx, rng_ctx);
    return ina_err_get_rc();
}

INA_API(void) iarray_random_ctx_free(iarray_context_t *ctx, iarray_random_ctx_t **rng_ctx)
{
    INA_VERIFY_FREE(rng_ctx);
    INA_UNUSED(ctx);
    vslDeleteStream(&((*rng_ctx)->stream));
    INA_MEM_FREE_SAFE(*rng_ctx);
}

INA_API(ina_rc_t) iarray_random_dist_set_param_float(iarray_random_ctx_t *ctx,
    iarray_random_dist_parameter_t key,
    float value)
{
    INA_VERIFY_NOT_NULL(ctx);
    ctx->fparams[key] = value;
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_random_dist_set_param_double(iarray_random_ctx_t *ctx,
    iarray_random_dist_parameter_t key,
    double value)
{
    INA_VERIFY_NOT_NULL(ctx);
    ctx->dparams[key] = value;
    return INA_SUCCESS;
}

static ina_rc_t _iarray_rand_internal(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *random_ctx,
    iarray_container_t *container,
    _iarray_random_method_t method)
{
    
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(container);

    ina_rc_t rc;

    int status = VSL_ERROR_OK;
    iarray_iter_write_block_t *iter;
    iarray_iter_write_block_value_t val;

    int64_t max_part_size = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        max_part_size *= container->dtshape->pshape[i];
    }
    void *buffer_mem = ina_mem_alloc(max_part_size * sizeof(double));

    IARRAY_FAIL_IF_ERROR(iarray_iter_write_block_new(ctx, &iter, container, container->dtshape->pshape, &val, false));

    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter))) {
        IARRAY_FAIL_IF_ERROR(iarray_iter_write_block_next(iter, NULL, 0));

        int64_t block_size = val.block_size;

        if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
            float *r = (float*)buffer_mem;
            switch (method) {
                case _IARRAY_RANDOM_METHOD_UNIFORM: {
                    float a = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_A];
                    float b = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_B];
                    status = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, random_ctx->stream,
                                          (int) block_size, r, a, b);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_GAUSSIAN: {
                    float mu = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_MU];
                    float sigma = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_SIGMA];
                    status = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, random_ctx->stream,
                                           (int) block_size, r, mu, sigma);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_BETA: {
                    float alpha = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_ALPHA];
                    float beta = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_BETA];
                    status = vsRngBeta(VSL_RNG_METHOD_BETA_CJA, random_ctx->stream, (int) block_size, r, alpha, beta, 0, 1);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_LOGNORMAL: {
                    float mu = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_MU];
                    float sigma = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_SIGMA];
                    status = vsRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2, random_ctx->stream, (int) block_size, r, mu, sigma, 0, 1);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_EXPONENTIAL: {
                    float beta = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_BETA];
                    status = vsRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF, random_ctx->stream, (int) block_size, r, 0, beta);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_BERNOUILLI: {
                    float p = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_P];
                    status = viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, random_ctx->stream, (int) block_size, (int *) r, p);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_BINOMIAL: {
                    float p = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_P];
                    int m = (int) random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_M];
                    status = viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, random_ctx->stream, (int) block_size, (int *) r, m, p);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_POISSON: {
                    float lambda = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_LAMBDA];
                    status = viRngPoisson(VSL_RNG_METHOD_POISSON_PTPE, random_ctx->stream, (int) block_size, (int *) r, lambda);
                    break;
                }
                default:
                    IARRAY_TRACE1(iarray.error, "The random generator method is invalid");
                    IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_METHOD));
            }
            if (status != VSL_ERROR_OK) {
                IARRAY_TRACE1(iarray.error, "The random generator method failed");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_RAND_METHOD_FAILED));
            }

            for (int64_t i = 0; i < block_size; ++i) {
                if ((method == _IARRAY_RANDOM_METHOD_BERNOUILLI) ||
                    (method == _IARRAY_RANDOM_METHOD_POISSON) ||
                    (method == _IARRAY_RANDOM_METHOD_BINOMIAL)) {
                    ((float *) val.block_pointer)[i] = (float) ((int *) r)[i];
                } else {
                    ((float *) val.block_pointer)[i] = r[i];
                }
            }
        }
        else {
            double *r = (double*)buffer_mem;
            switch (method) {
                case _IARRAY_RANDOM_METHOD_UNIFORM: {
                    double a = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_A];
                    double b = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_B];
                    status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, random_ctx->stream,
                                          (int) block_size, r, a, b);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_GAUSSIAN: {
                    double mu = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_MU];
                    double sigma = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_SIGMA];
                    status = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, random_ctx->stream,
                                           (int) block_size, r, mu, sigma);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_BETA: {
                    double alpha = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_ALPHA];
                    double beta = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_BETA];
                    status = vdRngBeta(VSL_RNG_METHOD_BETA_CJA, random_ctx->stream, (int) block_size, r, alpha, beta, 0, 1);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_LOGNORMAL: {
                    double mu = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_MU];
                    double sigma = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_SIGMA];
                    status = vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2, random_ctx->stream, (int) block_size, r, mu, sigma, 0, 1);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_EXPONENTIAL: {
                    double beta = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_BETA];
                    status = vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF, random_ctx->stream, (int) block_size, r, 0, beta);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_BERNOUILLI: {
                    double p = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_P];
                    status = viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, random_ctx->stream, (int) block_size, (int *) r, p);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_BINOMIAL: {
                    double p = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_P];
                    int m = (int) random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_M];
                    status = viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, random_ctx->stream, (int) block_size, (int *) r, m, p);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_POISSON: {
                    double lambda = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_LAMBDA];
                    status = viRngPoisson(VSL_RNG_METHOD_POISSON_PTPE, random_ctx->stream, (int) block_size, (int *) r, lambda);
                    break;
                }
                default:
                    IARRAY_TRACE1(iarray.error, "The random gneerator method is invalid");
                    IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_METHOD));
            }
            if (status != VSL_ERROR_OK) {
                IARRAY_TRACE1(iarray.error, "The random generator method failed");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_RAND_METHOD_FAILED));
            }

            for (int64_t i = 0; i < block_size; ++i) {
                if ((method == _IARRAY_RANDOM_METHOD_BERNOUILLI) ||
                    (method == _IARRAY_RANDOM_METHOD_POISSON) ||
                    (method == _IARRAY_RANDOM_METHOD_BINOMIAL)) {
                    ((double *) val.block_pointer)[i] = (double) ((int *) r)[i];
                } else {
                    ((double *) val.block_pointer)[i] = r[i];
                }
            }
        }
    }
    IARRAY_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    rc = INA_SUCCESS;
    goto cleanup;
fail:
    rc = ina_err_get_rc();
cleanup:
    INA_MEM_FREE_SAFE(buffer_mem);
    iarray_iter_write_block_free(&iter);
    return rc;
}

INA_API(ina_rc_t) iarray_random_rand(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *random_ctx,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        IARRAY_FAIL_IF_ERROR(iarray_random_dist_set_param_float(random_ctx, IARRAY_RANDOM_DIST_PARAM_A, 0.0f));
        IARRAY_FAIL_IF_ERROR(iarray_random_dist_set_param_float(random_ctx, IARRAY_RANDOM_DIST_PARAM_B, 1.0f));
    }
    else {
        IARRAY_FAIL_IF_ERROR(iarray_random_dist_set_param_double(random_ctx, IARRAY_RANDOM_DIST_PARAM_A, 0.0));
        IARRAY_FAIL_IF_ERROR(iarray_random_dist_set_param_double(random_ctx, IARRAY_RANDOM_DIST_PARAM_B, 1.0));
    }

    IARRAY_FAIL_IF_ERROR(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_UNIFORM);

    fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_random_randn(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *random_ctx,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        IARRAY_FAIL_IF_ERROR(iarray_random_dist_set_param_float(random_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 0.0f));
        IARRAY_FAIL_IF_ERROR(iarray_random_dist_set_param_float(random_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 1.0f));
    }
    else {
        IARRAY_FAIL_IF_ERROR(iarray_random_dist_set_param_double(random_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 0.0));
        IARRAY_FAIL_IF_ERROR(iarray_random_dist_set_param_double(random_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 1.0));
    }

    IARRAY_FAIL_IF_ERROR(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_GAUSSIAN);

    fail:
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_random_beta(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *random_ctx,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        if (random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_ALPHA] <= 0 ||
            random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_BETA] <= 0) {
            IARRAY_TRACE1(iarray.error, "The parameters for the beta distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }
    else {
        if (random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_ALPHA] <= 0 ||
            random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_BETA] <= 0) {
            IARRAY_TRACE1(iarray.error, "The parameters for the beta distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }

    IARRAY_FAIL_IF_ERROR(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_BETA);
fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_random_lognormal(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *random_ctx,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        if (random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_SIGMA] <= 0) {
            IARRAY_TRACE1(iarray.error, "The parameters for the lognormal distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }
    else {
        if (random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_SIGMA] <= 0) {
            IARRAY_TRACE1(iarray.error, "The parameters for the lognormal distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }

    IARRAY_FAIL_IF_ERROR(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_LOGNORMAL);

    fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_random_exponential(iarray_context_t *ctx,
                                           iarray_dtshape_t *dtshape,
                                           iarray_random_ctx_t *random_ctx,
                                           iarray_store_properties_t *store,
                                           int flags,
                                           iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        if (random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_BETA] <= 0) {
            IARRAY_TRACE1(iarray.error, "The parameters for the exponential distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }
    else {
        if (random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_BETA] <= 0) {
            IARRAY_TRACE1(iarray.error, "The parameters for the exponential distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }

    IARRAY_FAIL_IF_ERROR(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_EXPONENTIAL);

    fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_random_uniform(iarray_context_t *ctx,
                                        iarray_dtshape_t *dtshape,
                                        iarray_random_ctx_t *random_ctx,
                                        iarray_store_properties_t *store,
                                        int flags,
                                        iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        if (random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_A] >= random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_B]) {
            IARRAY_TRACE1(iarray.error, "The parameters for the uniform distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }
    else {
        if (random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_A] >= random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_B]) {
            IARRAY_TRACE1(iarray.error, "The parameters for the uniform distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }

    IARRAY_FAIL_IF_ERROR(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_UNIFORM);

    fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_random_normal(iarray_context_t *ctx,
                                       iarray_dtshape_t *dtshape,
                                       iarray_random_ctx_t *random_ctx,
                                       iarray_store_properties_t *store,
                                       int flags,
                                       iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        if (random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_SIGMA] <= 0) {
            IARRAY_TRACE1(iarray.error, "The parameters for the normal distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }
    else {
        if (random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_SIGMA] <= 0) {
            IARRAY_TRACE1(iarray.error, "The parameters for the normal distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }

    IARRAY_FAIL_IF_ERROR(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_GAUSSIAN);

    fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_random_bernoulli(iarray_context_t *ctx,
                                          iarray_dtshape_t *dtshape,
                                          iarray_random_ctx_t *random_ctx,
                                          iarray_store_properties_t *store,
                                          int flags,
                                          iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        if (random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_P] < 0 ||
            random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_P] > 1) {
            IARRAY_TRACE1(iarray.error, "The parameters for the bernoulli distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }
    else {
        if (random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_P] < 0 ||
            random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_P] > 1) {
            IARRAY_TRACE1(iarray.error, "The parameters for the bernoulli distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }

    IARRAY_FAIL_IF_ERROR(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_BERNOUILLI);

    fail:
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_random_binomial(iarray_context_t *ctx,
                                         iarray_dtshape_t *dtshape,
                                         iarray_random_ctx_t *random_ctx,
                                         iarray_store_properties_t *store,
                                         int flags,
                                         iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        if (random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_P] < 0 ||
            random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_P] > 1 ||
            random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_M] <= 0) {
            IARRAY_TRACE1(iarray.error, "The parameters for the binomial distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }
    else {
        if (random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_P] < 0 ||
            random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_P] > 1 ||
            random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_M] <= 0) {
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }

    IARRAY_FAIL_IF_ERROR(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_BINOMIAL);

    fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_random_poisson(iarray_context_t *ctx,
                                        iarray_dtshape_t *dtshape,
                                        iarray_random_ctx_t *random_ctx,
                                        iarray_store_properties_t *store,
                                        int flags,
                                        iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        if (random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_LAMBDA] <= 0) {
            IARRAY_TRACE1(iarray.error, "The parameters for the poisson distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }
    else {
        if (random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_LAMBDA] <= 0) {
            IARRAY_TRACE1(iarray.error, "The parameters for the poisson distribution are invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }

    IARRAY_FAIL_IF_ERROR(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_POISSON);

    fail:
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_random_kstest(iarray_context_t *ctx,
                                       iarray_container_t *container1,
                                       iarray_container_t *container2,
                                       bool *res)
{

    IARRAY_FAIL_IF(container1->catarr->size != container2->catarr->size);
    int64_t size = container1->catarr->size;

    int nbins = 100;
    double bins[100];
    double hist1[100];
    double hist2[100];

    double max = -INFINITY;
    double min = INFINITY;

    iarray_iter_read_t *iter;
    iarray_iter_read_value_t val;
    IARRAY_FAIL_IF_ERROR(iarray_iter_read_new(ctx, &iter, container1, &val));

    while (INA_SUCCEED(iarray_iter_read_has_next(iter))) {
        IARRAY_FAIL_IF_ERROR(iarray_iter_read_next(iter));

        double data;
        switch(container1->dtshape->dtype){
            case IARRAY_DATA_TYPE_DOUBLE:
                data = ((double *) val.elem_pointer)[0];
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                data = ((float *) val.elem_pointer)[0];
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }

        max = (data > max) ? data : max;
        min = (data < min) ? data : min;
    }
    IARRAY_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    iarray_iter_read_free(&iter);

    IARRAY_FAIL_IF_ERROR(iarray_iter_read_new(ctx, &iter, container2, &val));
    while (INA_SUCCEED(iarray_iter_read_has_next(iter))) {
        IARRAY_FAIL_IF_ERROR(iarray_iter_read_next(iter));

        double data;
        switch(container1->dtshape->dtype){
            case IARRAY_DATA_TYPE_DOUBLE:
                data = ((double *) val.elem_pointer)[0];
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                data = ((float *) val.elem_pointer)[0];
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }

        max = (data > max) ? data : max;
        min = (data < min) ? data : min;
    }
    iarray_iter_read_free(&iter);
    IARRAY_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    for (int i = 0; i < nbins; ++i) {
        bins[i] = min + (max-min)/nbins * (i+1);
        hist1[i] = 0;
        hist2[i] = 0;
    }

    IARRAY_FAIL_IF_ERROR(iarray_iter_read_new(ctx, &iter, container1, &val));

    while (INA_SUCCEED(iarray_iter_read_has_next(iter))) {
        IARRAY_FAIL_IF_ERROR(iarray_iter_read_next(iter));

        double data;
        switch(container1->dtshape->dtype){
            case IARRAY_DATA_TYPE_DOUBLE:
                data = ((double *) val.elem_pointer)[0];
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                data = ((float *) val.elem_pointer)[0];
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }

        for (int i = 0; i < nbins; ++i) {
            if (data <= bins[i]) {
                hist1[i] += 1;
                break;
            }
        }
    }
    iarray_iter_read_free(&iter);
    IARRAY_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    IARRAY_FAIL_IF_ERROR(iarray_iter_read_new(ctx, &iter, container2, &val));

    while (INA_SUCCEED(iarray_iter_read_has_next(iter))) {
        IARRAY_FAIL_IF_ERROR(iarray_iter_read_next(iter));

        double data;
        switch(container1->dtshape->dtype){
            case IARRAY_DATA_TYPE_DOUBLE:
                data = ((double *) val.elem_pointer)[0];
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                data = ((float *) val.elem_pointer)[0];
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }
        for (int i = 0; i < nbins; ++i) {
            if (data <= bins[i]) {
                hist2[i] += 1;
                break;
            }
        }
    }
    iarray_iter_read_free(&iter);
    IARRAY_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    for (int i = 1; i < nbins; ++i) {
        hist1[i] += hist1[i-1];
        hist2[i] += hist2[i-1];
    }

    double max_dif = -INFINITY;
    for (int i = 0; i < nbins; ++i) {
        max_dif = (fabs(hist1[i] - hist2[i]) / size > max_dif) ? fabs(hist1[i] - hist2[i]) / size : max_dif;
    }

    double a = 0.001;
    double threshold = sqrt(- log(a) / 2) * sqrt(2 * ((double) size) / (size * size));

    *res = (max_dif < threshold);
    return INA_SUCCESS;

    fail:
    iarray_iter_read_free(&iter);
    return ina_err_get_rc();

}
