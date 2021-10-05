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


int iarray_random_uniform_fn(iarray_random_ctx_t *random_ctx,
                            VSLStreamStatePtr stream,
                            uint8_t itemsize,
                            int32_t blocksize,
                            uint8_t *buffer) {
    if (itemsize == 4) {
        float a = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_A];
        float b = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_B];
        return vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream,
                              (int) blocksize, (float *) buffer, a, b);
    } else {
        double a = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_A];
        double b = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_B];
        return vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream,
                            (int) blocksize, (double *) buffer, a, b);
    }
}


INA_API(ina_rc_t) iarray_random_uniform(iarray_context_t *ctx,
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

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        if (random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_A] >= random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_B]) {
            IARRAY_TRACE1(iarray.error, "The parameters for the uniform distribution are invalid");
            return (INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }
    else {
        if (random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_A] >= random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_B]) {
            IARRAY_TRACE1(iarray.error, "The parameters for the uniform distribution are invalid");
            return (INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }

    return iarray_random_prefilter(ctx, dtshape, random_ctx, iarray_random_uniform_fn, storage, container);
}
