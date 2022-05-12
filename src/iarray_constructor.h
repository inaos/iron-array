/*
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#ifndef _IARRAY_CONSTRUCTOR_H_
#define _IARRAY_CONSTRUCTOR_H_

#include "iarray_private.h"
#include <libiarray/iarray.h>


// TODO: clang complains about unused function.  provide a test using this.
static ina_rc_t _iarray_container_new(iarray_context_t *ctx,
                                      iarray_dtshape_t *dtshape,
                                      iarray_storage_t *storage,
                                      iarray_container_t **c)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(c);

    /* validation */
    if (dtshape->ndim > CATERVA_MAX_DIM) {
        IARRAY_TRACE1(iarray.error, "The container dimension is larger than caterva maximum dimension");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }

    *c = (iarray_container_t*)ina_mem_alloc(sizeof(iarray_container_t));
    if ((*c) == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iarray container");
        return INA_ERROR(INA_ERR_FAILED);
    }

    (*c)->dtshape = (iarray_dtshape_t*)ina_mem_alloc(sizeof(iarray_dtshape_t));
    if ((*c)->dtshape == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iarray dtshape");
        return INA_ERROR(INA_ERR_FAILED);
    }
    IARRAY_RETURN_IF_FAILED(iarray_set_dtype_size(dtshape));
    ina_mem_cpy((*c)->dtshape, dtshape, sizeof(iarray_dtshape_t));

    iarray_auxshape_t auxshape;
    for (int i = 0; i < dtshape->ndim; ++i) {
        auxshape.shape_wos[i] = dtshape->shape[i];
        auxshape.chunkshape_wos[i] = storage->chunkshape[i];
        auxshape.blockshape_wos[i] = storage->blockshape[i];
        auxshape.offset[i] = 0;
        auxshape.index[i] = (int8_t) i;
    }
    (*c)->auxshape = (iarray_auxshape_t*)ina_mem_alloc(sizeof(iarray_auxshape_t));
    if ((*c)->auxshape == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iarray auxshape");
        return INA_ERROR(INA_ERR_FAILED);
    }
    ina_mem_cpy((*c)->auxshape, &auxshape, sizeof(iarray_auxshape_t));

    (*c)->storage = ina_mem_alloc(sizeof(iarray_storage_t));
    if ((*c)->storage == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the store parameters");
        return INA_ERROR(INA_ERR_FAILED);
    }
    ina_mem_cpy((*c)->storage, storage, sizeof(iarray_storage_t));

    (*c)->catarr = NULL;
    (*c)->container_viewed = NULL;
    (*c)->transposed = false;

    return INA_SUCCESS;
}

// TODO: clang complains about unused function.  provide a test using this.
inline static ina_rc_t _iarray_view_new(iarray_context_t *ctx,
                                        iarray_container_t *pred,
                                        iarray_dtshape_t *dtshape,
                                        const int64_t *offset,
                                        iarray_container_t **c)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(pred);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(offset);
    INA_VERIFY_NOT_NULL(c);

    /* validation */
    if (dtshape->ndim > CATERVA_MAX_DIM) {
        IARRAY_TRACE1(iarray.error, "The container dimension is larger than the caterva maximum dimension");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }

    *c = (iarray_container_t*)ina_mem_alloc(sizeof(iarray_container_t));
    if (*c == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iarray container");
        return INA_ERROR(INA_ERR_FAILED);
    }

    (*c)->dtshape = (iarray_dtshape_t*)ina_mem_alloc(sizeof(iarray_dtshape_t));
    if ((*c)->dtshape == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iarray dtshape");
        return INA_ERROR(INA_ERR_FAILED);
    }
    ina_mem_cpy((*c)->dtshape, dtshape, sizeof(iarray_dtshape_t));

    iarray_auxshape_t auxshape;
    for (int i = 0; i < dtshape->ndim; ++i) {
        auxshape.shape_wos[i] = dtshape->shape[i];
        auxshape.chunkshape_wos[i] = pred->storage->chunkshape[i];
        auxshape.blockshape_wos[i] = pred->storage->blockshape[i];
        auxshape.offset[i] = offset[i];
        auxshape.index[i] = (int8_t) i;
    }
    (*c)->auxshape = (iarray_auxshape_t*)ina_mem_alloc(sizeof(iarray_auxshape_t));
    if ((*c)->auxshape == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iarray auxdtshape");
        return INA_ERROR(INA_ERR_FAILED);
    }
    ina_mem_cpy((*c)->auxshape, &auxshape, sizeof(iarray_auxshape_t));

    (*c)->container_viewed = pred;
    (*c)->transposed = false;

    (*c)->storage = pred->storage;
    (*c)->catarr = pred->catarr;

    return INA_SUCCESS;
}

#endif
