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

#include <mkl_vsl.h>

#include "iarray_constructor.h"

#define _IARRAY_RNG_CHUNK_SIZE 1024

typedef enum _iarray_random_method_e {
    _IARRAY_RANDOM_METHOD_UNIFORM,
    _IARRAY_RANDOM_METHOD_GAUSSIAN,
    _IARRAY_RANDOM_METHOD_BETA,
    _IARRAY_RANDOM_METHOD_LOGNORMAL,
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
            INA_FAIL_IF(1);
    }

    vslNewStream(&(*rng_ctx)->stream, mkl_rng, seed);

    ina_mem_set((*rng_ctx)->dparams, 0, sizeof(double)*(IARRAY_RANDOM_DIST_PARAM_SENTINEL));
    ina_mem_set((*rng_ctx)->fparams, 0, sizeof(float)*(IARRAY_RANDOM_DIST_PARAM_SENTINEL));

    return INA_SUCCESS;

fail:
    iarray_random_ctx_free(ctx, rng_ctx);
    return INA_ERR_ILLEGAL;
}

INA_API(void) iarray_random_ctx_free(iarray_context_t *ctx, iarray_random_ctx_t **rng_ctx)
{
    INA_ASSERT_NOT_NULL(ctx);
    INA_VERIFY_FREE(rng_ctx);
    vslDeleteStream(&((*rng_ctx)->stream));
    INA_MEM_FREE_SAFE(*rng_ctx);
}

INA_API(ina_rc_t) iarray_random_dist_set_param_float(iarray_random_ctx_t *ctx,
    iarray_random_dist_parameter_t key,
    float value)
{
    INA_ASSERT_NOT_NULL(ctx);
    ctx->fparams[key] = value;
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_random_dist_set_param_double(iarray_random_ctx_t *ctx,
    iarray_random_dist_parameter_t key,
    double value)
{
    INA_ASSERT_NOT_NULL(ctx);
    ctx->dparams[key] = value;
    return INA_SUCCESS;
}

static ina_rc_t _iarray_rand_internal(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *random_ctx,
    iarray_container_t *container,
    _iarray_random_method_t method)
{
    int status = VSL_ERROR_OK;
    iarray_iter_write_part_t *iter;
    iarray_iter_write_part_new(ctx, container, &iter);

    uint64_t max_part_size = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        max_part_size *= dtshape->pshape[i];
    }
    void *buffer_mem = ina_mem_alloc(max_part_size * sizeof(double));

    for (iarray_iter_write_part_init(iter);
        !iarray_iter_write_part_finished(iter);
        iarray_iter_write_part_next(iter)) {

        iarray_iter_write_part_value_t val;
        iarray_iter_write_part_value(iter, &val);

        uint64_t part_size = 1;
        for (int i = 0; i < dtshape->ndim; ++i) {
            part_size *= val.part_shape[i];
        }

        if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
            float *r = (float*)buffer_mem;
            switch (method) {
                case _IARRAY_RANDOM_METHOD_UNIFORM:
                    status = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, random_ctx->stream, (int)part_size, r, 0.0, 1.0);
                    break;
                case _IARRAY_RANDOM_METHOD_GAUSSIAN:
                    status = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, random_ctx->stream, (int)part_size, r, 0.0, 1.0);
                    break;
                case _IARRAY_RANDOM_METHOD_BETA:
                {
                    float alpha = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_ALPHA];
                    float beta = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_BETA];
                    status = vsRngBeta(VSL_RNG_METHOD_BETA_CJA, random_ctx->stream, (int) part_size, r, alpha, beta, 0, 1);
                }
                case _IARRAY_RANDOM_METHOD_LOGNORMAL:
                {
                    float mu = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_MU];
                    float sigma = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_SIGMA];
                    status = vsRngLognormal(method, random_ctx->stream, (int) part_size, r, mu, sigma, 0, 1);
                }
                    break;
            }
            INA_FAIL_IF(status != VSL_ERROR_OK);

            for (uint64_t i = 0; i < part_size; ++i) {
                ((float *)val.pointer)[i] = r[i];
            }
        }
        else {
            double *r = (double*)buffer_mem;
            switch (method) {
                case _IARRAY_RANDOM_METHOD_UNIFORM:
                    status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, random_ctx->stream, (int)part_size, r, 0.0, 1.0);
                    break;
                case _IARRAY_RANDOM_METHOD_GAUSSIAN:
                    status = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, random_ctx->stream, (int)part_size, r, 0.0, 1.0);
                    break;
                case _IARRAY_RANDOM_METHOD_BETA:
                {
                    double alpha = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_ALPHA];
                    double beta = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_BETA];
                    status = vdRngBeta(VSL_RNG_METHOD_BETA_CJA, random_ctx->stream, (int) part_size, r, alpha, beta, 0, 1);
                }
                    break;
                case _IARRAY_RANDOM_METHOD_LOGNORMAL:
                {
                    double mu = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_MU];
                    double sigma = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_SIGMA];
                    status = vdRngLognormal(method, random_ctx->stream, (int) part_size, r, mu, sigma, 0, 1);
                }
                    break;
            }
            INA_FAIL_IF(status != VSL_ERROR_OK);

            for (uint64_t i = 0; i < part_size; ++i) {
                ((double *)val.pointer)[i] = r[i];
            }
        }
    }

    return INA_SUCCESS;

fail:
    return INA_ERR_ILLEGAL;
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
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(container);
    
    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_UNIFORM);
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
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_GAUSSIAN);
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
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        INA_FAIL_IF(random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_ALPHA] == 0);
        /* FIXME: add more validations */
    }
    else {
        INA_FAIL_IF(random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_ALPHA] == 0);
        /* FIXME: add more validations */
    }

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_BETA);

fail:
    return INA_ERR_MISSING;
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
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_LOGNORMAL);
}
