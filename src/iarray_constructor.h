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

#ifndef _IARRAY_CONSTRUCTOR_H_
#define _IARRAY_CONSTRUCTOR_H_

#include <libiarray/iarray.h>

#include <iarray_private.h>

static ina_rc_t _iarray_container_new(iarray_context_t *ctx, iarray_dtshape_t *dtshape,
                                      iarray_store_properties_t *store,
                                      int flags,
                                      iarray_container_t **c)
{
    blosc2_cparams cparams = BLOSC_CPARAMS_DEFAULTS;
    blosc2_dparams dparams = BLOSC_DPARAMS_DEFAULTS;
    caterva_dims_t pshape;
    caterva_dims_t shape;
    int blosc_filter_idx = 0;

    /* validation */
    if (dtshape->ndim > CATERVA_MAXDIM) {
        return INA_ERROR(INA_ERR_EXCEEDED);
    }
    if (flags & IARRAY_CONTAINER_PERSIST && store == NULL) {
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }
    for (int i = 0; i < dtshape->ndim; ++i) {
        if (dtshape->shape[i] < dtshape->partshape[i]) {
            return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
        }
    }

    *c = (iarray_container_t*)ina_mem_alloc(sizeof(iarray_container_t));
    INA_RETURN_IF_NULL(c);

    (*c)->dtshape = (iarray_dtshape_t*)ina_mem_alloc(sizeof(iarray_dtshape_t));
    INA_FAIL_IF((*c)->dtshape == NULL);
    ina_mem_cpy((*c)->dtshape, dtshape, sizeof(iarray_dtshape_t));

    (*c)->frame = (blosc2_frame*)ina_mem_alloc(sizeof(blosc2_frame));
    INA_FAIL_IF((*c)->frame == NULL);
    ina_mem_cpy((*c)->frame, &BLOSC_EMPTY_FRAME, sizeof(blosc2_frame));

    (*c)->cparams = (blosc2_cparams*)ina_mem_alloc(sizeof(blosc2_cparams));
    INA_FAIL_IF((*c)->cparams == NULL);

    (*c)->dparams = (blosc2_dparams*)ina_mem_alloc(sizeof(blosc2_dparams));
    INA_FAIL_IF((*c)->dparams == NULL);

    (*c)->shape = (caterva_dims_t*)ina_mem_alloc(sizeof(caterva_dims_t));
    INA_FAIL_IF((*c)->shape == NULL);

    (*c)->pshape = (caterva_dims_t*)ina_mem_alloc(sizeof(caterva_dims_t));
    INA_FAIL_IF((*c)->pshape == NULL);

    (*c)->transposed = 0;

    if (flags & IARRAY_CONTAINER_PERSIST) {
        (*c)->store = ina_mem_alloc(sizeof(_iarray_container_store_t));
        INA_FAIL_IF((*c)->store == NULL);
        (*c)->store->id = ina_str_new_fromcstr(store->id);
        (*c)->frame->fname = (char*)ina_str_cstr((*c)->store->id); /* FIXME: shouldn't fname be a const char? */
    }

    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            cparams.typesize = sizeof(double);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            cparams.typesize = sizeof(float);
            break;
    }
    cparams.compcode = ctx->cfg->compression_codec;
    cparams.clevel = (uint8_t)ctx->cfg->compression_level; /* Since its just a mapping, we know the cast is ok */
    cparams.blocksize = ctx->cfg->blocksize;
    cparams.nthreads = (uint16_t)ctx->cfg->max_num_threads; /* Since its just a mapping, we know the cast is ok */
    if (dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE && ctx->cfg->flags & IARRAY_COMP_TRUNC_PREC) {
        cparams.filters[blosc_filter_idx] = BLOSC_TRUNC_PREC;
        cparams.filters_meta[blosc_filter_idx] = ctx->cfg->fp_mantissa_bits;
        blosc_filter_idx++;
    }
    if (ctx->cfg->flags & IARRAY_COMP_BITSHUFFLE) {
        cparams.filters[blosc_filter_idx] = BLOSC_BITSHUFFLE;
        blosc_filter_idx++;
    }
    if (ctx->cfg->flags & IARRAY_COMP_SHUFFLE) {
        cparams.filters[blosc_filter_idx] = BLOSC_SHUFFLE;
        blosc_filter_idx++;
    }
    if (ctx->cfg->flags & IARRAY_COMP_DELTA) {
        cparams.filters[blosc_filter_idx] = BLOSC_DELTA;
        blosc_filter_idx++;
    }
    ina_mem_cpy((*c)->cparams, &cparams, sizeof(blosc2_cparams));

    dparams.nthreads = (uint16_t)ctx->cfg->max_num_threads; /* Since its just a mapping, we know the cast is ok */
    ina_mem_cpy((*c)->dparams, &dparams, sizeof(blosc2_dparams));

    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        shape.dims[i] = 1;
        pshape.dims[i] = 1;
    }
    for (int i = 0; i < dtshape->ndim; ++i) { // FIXME: 1's at the beginning should be removed
        shape.dims[i] = dtshape->shape[i];
        pshape.dims[i] = dtshape->partshape[i];
    }
    shape.ndim = dtshape->ndim;
    pshape.ndim = dtshape->ndim;

    ina_mem_cpy((*c)->shape, &shape, sizeof(caterva_dims_t));
    ina_mem_cpy((*c)->pshape, &pshape, sizeof(caterva_dims_t));

    caterva_ctx_t *cat_ctx = caterva_new_ctx(NULL, NULL, cparams, dparams);

    (*c)->catarr = caterva_empty_array(cat_ctx, (*c)->frame, *(*c)->pshape);
    INA_FAIL_IF((*c)->catarr == NULL);

    return INA_SUCCESS;

fail:
    iarray_container_free(ctx, c);
    caterva_free_ctx(cat_ctx);
    return ina_err_get_rc();
}

#endif
