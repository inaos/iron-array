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
#include "../iarray_constructor.h"


ina_rc_t iarray_linalg_transpose(iarray_context_t *ctx, iarray_container_t *a, bool view,
                                 iarray_storage_t *storage, iarray_container_t **b) {
    if (a->view) {
        IARRAY_TRACE1(iarray.error, "Transposing views is not supported yet");
        return INA_ERROR(INA_ERR_NOT_SUPPORTED);
    }
    if (a->dtshape->ndim != 2) {
        IARRAY_TRACE1(iarray.error, "It is only supported for two dimensions");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }
    if (!view) {
        IARRAY_TRACE1(iarray.error, "The result must be a view (storage=NULL)");
        return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
    } else {
        int64_t offset[IARRAY_DIMENSION_MAX] = {0};
        iarray_dtshape_t view_dtshape = {0};
        view_dtshape.ndim = a->dtshape->ndim;
        view_dtshape.dtype = a->dtshape->dtype;
        for (int i = 0; i < view_dtshape.ndim; ++i) {
            view_dtshape.shape[i] = a->dtshape->shape[(a->dtshape->ndim - 1) - i];
        }
        _iarray_view_new(ctx, a, a->dtshape, offset, b);
    }
    return INA_SUCCESS;
}
