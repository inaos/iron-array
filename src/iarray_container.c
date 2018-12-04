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

INA_API(ina_rc_t) iarray_container_dtshape_equal(iarray_dtshape_t *a, iarray_dtshape_t *b)
{
    if (a->dtype != b->dtype) {
        return -1;
    }
    if (a->ndim != b->ndim) {
        return -1;
    }
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return -1;
        }
    }
    return 0;
}

INA_API(ina_rc_t) iarray_container_new(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    return _iarray_container_new(ctx, dtshape, store, flags, container);
}

INA_API(ina_rc_t) iarray_slice(iarray_context_t *ctx,
    iarray_container_t *c,
    uint64_t *start_,
    uint64_t *stop_,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start_);
    INA_VERIFY_NOT_NULL(stop_);
    INA_VERIFY_NOT_NULL(container);

    caterva_dims_t start = caterva_new_dims(start_, c->dtshape->ndim);
    caterva_dims_t stop = caterva_new_dims(stop_, c->dtshape->ndim);

    iarray_dtshape_t dtshape;
    for (int i = 0; i < c->dtshape->ndim; ++i) {
        dtshape.shape[i] = (stop_[i] - start_[i]);
        dtshape.partshape[i] = c->dtshape->partshape[i];
    }
    dtshape.ndim = c->dtshape->ndim;
    dtshape.dtype = c->dtshape->dtype;
    INA_RETURN_IF_FAILED(iarray_container_new(ctx, &dtshape, store, flags, container));

    INA_FAIL_IF(caterva_get_slice((*container)->catarr, c->catarr, start, stop) != 0);

    return INA_SUCCESS;

fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_container_info(iarray_container_t *c,
    uint64_t *nbytes,
    uint64_t *cbytes)
{
    INA_VERIFY_NOT_NULL(c);

    *nbytes = (uint64_t) c->catarr->sc->nbytes;
    *cbytes = (uint64_t) c->catarr->sc->cbytes;

    return INA_SUCCESS;
}

INA_API(void) iarray_container_free(iarray_context_t *ctx, iarray_container_t **container)
{
    INA_VERIFY_FREE(container);
    if ((*container)->catarr != NULL) {
        caterva_free_array((*container)->catarr);
    }
    INA_MEM_FREE_SAFE((*container)->frame);
    INA_MEM_FREE_SAFE((*container)->cparams);
    INA_MEM_FREE_SAFE((*container)->dparams);
    INA_MEM_FREE_SAFE((*container)->shape);
    INA_MEM_FREE_SAFE((*container)->pshape);
    INA_MEM_FREE_SAFE((*container)->dtshape);
    INA_MEM_FREE_SAFE(*container);
}
