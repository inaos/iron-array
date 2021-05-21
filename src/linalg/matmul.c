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
#include "matmul/gemm.h"
#include "matmul/gemv.h"


ina_rc_t iarray_linalg_matmul(iarray_context_t *ctx,
                               iarray_container_t *a,
                               iarray_container_t *b,
                               iarray_storage_t *storage,
                               iarray_container_t **c) {
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(b);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(c);

    // Inputs checking
    if (a->dtshape->ndim != 2 || (b->dtshape->ndim != 1 && b->dtshape->ndim != 2)) {
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }
    if (a->dtshape->shape[1] != b->dtshape->shape[0]) {
        return INA_ERROR(IARRAY_ERR_INVALID_SHAPE);
    }
    if (a->dtshape->dtype != b->dtshape->dtype) {
        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }


    // C parameters
    iarray_dtshape_t dtshape = {0};
    dtshape.dtype = a->dtshape->dtype;
    dtshape.ndim = b->dtshape->ndim;
    dtshape.shape[0] = a->dtshape->shape[0];
   if (dtshape.ndim > 1) {
       dtshape.shape[1] = b->dtshape->shape[1];
   }

    if (storage->backend == IARRAY_STORAGE_BLOSC) {
        for (int i = 0; i < dtshape.ndim; ++i) {
            if (dtshape.shape[i] < storage->chunkshape[i]) {
                return INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE);
            }
        }
    }

    // Create output array
    IARRAY_RETURN_IF_FAILED(iarray_empty(ctx, &dtshape, storage, 0, c));

    if (b->dtshape->ndim == 2) {
        IARRAY_RETURN_IF_FAILED(iarray_gemm(ctx, a, b, *c));
    } else {
        IARRAY_RETURN_IF_FAILED(iarray_gemv(ctx, a, b, *c));
    }

    return INA_SUCCESS;
}