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
#include "iarray_constructor.h"


ina_rc_t iarray_linalg_transpose(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t **c) {
    INA_VERIFY_NOT_NULL(ctx);
    if (a->dtshape->ndim != 2) {
        IARRAY_TRACE1(iarray.error, "The container dimension is not 2");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }

    int64_t offset[IARRAY_DIMENSION_MAX] = {0};
    _iarray_view_new(ctx, a, a->dtshape, offset, c);

    if (a->transposed == 0) {
        (*c)->transposed = 1;
    }
    else {
        (*c)->transposed = 0;
    }

    int64_t aux[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        aux[i] = a->dtshape->shape[i];
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        a->dtshape->shape[i] = aux[a->dtshape->ndim - 1 - i];
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        aux[i] = a->storage->chunkshape[i];
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        a->storage->chunkshape[i] = aux[a->dtshape->ndim - 1 - i];
    }
    return INA_SUCCESS;
}
