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

struct iarray_random_ctx_s {
    iarray_random_rng_t rng;
    uint32_t seed;
    VSLStreamStatePtr stream;
};

INA_API(ina_rc_t) iarray_random_ctx_new(iarray_context_t *ctx,
    uint32_t seed,
    iarray_random_rng_t rng,
    iarray_random_ctx_t **rng_ctx)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(*rng_ctx);
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

    return INA_SUCCESS;

fail:
    iarray_random_ctx_free(ctx, rng_ctx);
}

INA_API(void) iarray_random_ctx_free(iarray_context_t *ctx, iarray_random_ctx_t **rng_ctx)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_FREE(rng_ctx);
    vslDeleteStream((*rng_ctx)->stream);
    INA_MEM_FREE_SAFE(*rng_ctx);
}

INA_API(ina_rc_t) iarray_rand(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *random_ctx,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    //vRngUniform

    /* implement rand */

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_random_randn(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *rand_ctx,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_random_beta(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *rand_ctx,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_random_lognormal(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *rand_ctx,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    return INA_SUCCESS;
}