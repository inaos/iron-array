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

#include "iarray_private.h"
#include <libiarray/iarray.h>


int iarray_random_lognormal_fn(iarray_random_ctx_t *random_ctx,
                            VSLStreamStatePtr stream,
                            uint8_t itemsize,
                            int32_t blocksize,
                            uint8_t *buffer) {
    double mu = random_ctx->params[IARRAY_RANDOM_DIST_PARAM_MU];
    double sigma = random_ctx->params[IARRAY_RANDOM_DIST_PARAM_SIGMA];
    if (itemsize == 4) {
        return vsRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2, stream,
                              (int) blocksize, (float *) buffer, (float) mu, (float) sigma, 0., 1.);
    } else {
        return vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2, stream,
                             (int) blocksize, (double *) buffer, mu, sigma, 0, 1);
    }
}


INA_API(ina_rc_t) iarray_random_lognormal(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     iarray_random_ctx_t *random_ctx,
                                     iarray_storage_t *storage,
                                     iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    if (dtshape->dtype != IARRAY_DATA_TYPE_FLOAT && dtshape->dtype != IARRAY_DATA_TYPE_DOUBLE) {
        IARRAY_TRACE1(iarray.error, "Dtype is not supported");
        return (INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    /* validate distribution parameters */
    if (random_ctx->params[IARRAY_RANDOM_DIST_PARAM_SIGMA] <= 0) {
        IARRAY_TRACE1(iarray.error, "The parameters for the lognormal distribution are invalid");
        return (INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
    }

    return iarray_random_prefilter(ctx, dtshape, random_ctx, iarray_random_lognormal_fn, storage, container);
}

