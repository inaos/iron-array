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


int iarray_random_randn_fn(iarray_random_ctx_t *random_ctx,
                            VSLStreamStatePtr stream,
                            uint8_t itemsize,
                            int32_t blocksize,
                            uint8_t *buffer) {
    if (itemsize == 4) {
        float mu = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_MU];
        float sigma = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_SIGMA];
        return vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream,
                               (int) blocksize, (float *) buffer, mu, sigma);
    } else {
        double mu = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_MU];
        double sigma = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_SIGMA];
        return vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream,
                             (int) blocksize, (double *) buffer, mu, sigma);
    }
}


INA_API(ina_rc_t) iarray_random_randn(iarray_context_t *ctx,
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
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        IARRAY_RETURN_IF_FAILED(iarray_random_dist_set_param_float(random_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 0.0f));
        IARRAY_RETURN_IF_FAILED(iarray_random_dist_set_param_float(random_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 1.0f));
    }
    else {
        IARRAY_RETURN_IF_FAILED(iarray_random_dist_set_param_double(random_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 0.0));
        IARRAY_RETURN_IF_FAILED(iarray_random_dist_set_param_double(random_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 1.0));
    }

    return iarray_random_prefilter(ctx, dtshape, random_ctx, iarray_random_randn_fn, storage, container);
}

