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


int iarray_random_poisson_fn(iarray_random_ctx_t *random_ctx,
                            VSLStreamStatePtr stream,
                            uint8_t itemsize,
                            int32_t blocksize,
                            uint8_t *buffer) {

    int *tmp = ina_mem_alloc(blocksize * sizeof(int));

    double lambda;
    if (itemsize == 4) {
        lambda = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_LAMBDA];
    } else {
        lambda = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_LAMBDA];
    }
    int state = viRngPoisson(VSL_RNG_METHOD_POISSON_PTPE, stream,
                         (int) blocksize, tmp, lambda);

    for (int i = 0; i < blocksize; ++i) {
        if (itemsize == 4) {
            ((float *) buffer)[i] = (float) tmp[i];
        } else {
            ((double *) buffer)[i] = (double) tmp[i];
        }
    }
    INA_MEM_FREE_SAFE(tmp);
    return state;
}


INA_API(ina_rc_t) iarray_random_poisson(iarray_context_t *ctx,
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
        if (random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_LAMBDA] <= 0) {
            IARRAY_TRACE1(iarray.error, "The parameters for the poisson distribution are invalid");
            return (INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }
    else {
        if (random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_LAMBDA] <= 0) {
            IARRAY_TRACE1(iarray.error, "The parameters for the poisson distribution are invalid");
            return (INA_ERROR(IARRAY_ERR_INVALID_RAND_PARAM));
        }
    }

    return iarray_random_prefilter(ctx, dtshape, random_ctx, iarray_random_poisson_fn, storage, container);
}
