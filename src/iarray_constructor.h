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


static int32_t serialize_meta(iarray_data_type_t dtype, uint8_t **smeta)
{
    if (smeta == NULL) {
        return -1;
    }
    if (dtype > IARRAY_DATA_TYPE_MAX) {
        return -1;
    }
    int32_t smeta_len = 3;  // the dtype should take less than 7-bit, so 1 byte is enough to store it
    *smeta = malloc((size_t)smeta_len);

    // version
    **smeta = 0;

    // dtype entry
    *(*smeta + 1) = (uint8_t)dtype;  // positive fixnum (7-bit positive integer)

    // flags (initialising all the entries to 0)
    *(*smeta + 2) = 0;  // positive fixnum (7-bit for flags)

    return smeta_len;
}

// TODO: clang complains about unused function.  provide a test using this.
static ina_rc_t _iarray_container_new(iarray_context_t *ctx,
                                      iarray_dtshape_t *dtshape,
                                      iarray_storage_t *storage,
                                      int flags,
                                      iarray_container_t **c)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(c);

    blosc2_cparams cparams = {0};
    blosc2_dparams dparams = {0};

    ina_rc_t rc;
    int blosc_filter_idx = 0;

    /* validation */
    if (dtshape->ndim > CATERVA_MAX_DIM) {
        IARRAY_TRACE1(iarray.error, "The container dimension is larger than caterva maximum dimension");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
    }
    if (flags & IARRAY_CONTAINER_PERSIST && storage->filename == NULL) {
        IARRAY_TRACE1(iarray.error, "Error with persistency flags");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
    }
    if (storage->backend == IARRAY_STORAGE_BLOSC) {
        for (int i = 0; i < dtshape->ndim; ++i) {
            if (dtshape->shape[i] < storage->pshape[i]) {
                IARRAY_TRACE1(iarray.error, "The pshape is larger than the shape");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
            }
        }
    }

    *c = (iarray_container_t*)ina_mem_alloc(sizeof(iarray_container_t));
    if ((*c) == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iarray container");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }

    (*c)->dtshape = (iarray_dtshape_t*)ina_mem_alloc(sizeof(iarray_dtshape_t));
    if ((*c)->dtshape == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iarray dtshape");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }
    ina_mem_cpy((*c)->dtshape, dtshape, sizeof(iarray_dtshape_t));

    if (storage->backend == IARRAY_STORAGE_PLAINBUFFER) {
        for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
            storage->pshape[i] = dtshape->shape[i];
            storage->bshape[i] = dtshape->shape[i];
        }
    }

    (*c)->cparams = (blosc2_cparams*)ina_mem_alloc(sizeof(blosc2_cparams));
    if ((*c)->cparams == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the blosc cparams");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }

    (*c)->dparams = (blosc2_dparams*)ina_mem_alloc(sizeof(blosc2_dparams));
    if ((*c)->dparams == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the blosc dparams");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }

    iarray_auxshape_t auxshape;
    for (int i = 0; i < dtshape->ndim; ++i) {
        auxshape.shape_wos[i] = dtshape->shape[i];
        auxshape.pshape_wos[i] = storage->pshape[i];
        auxshape.bshape_wos[i] = storage->bshape[i];
        auxshape.offset[i] = 0;
        auxshape.index[i] = (uint8_t) i;
    }
    (*c)->auxshape = (iarray_auxshape_t*)ina_mem_alloc(sizeof(iarray_auxshape_t));
    if ((*c)->auxshape == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iarray auxshape");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }
    ina_mem_cpy((*c)->auxshape, &auxshape, sizeof(iarray_auxshape_t));

    (*c)->transposed = false;
    (*c)->view = false;

    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            cparams.typesize = sizeof(double);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            cparams.typesize = sizeof(float);
            break;
        default:
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
            break;
    }
    cparams.compcode = ctx->cfg->compression_codec;
    cparams.use_dict = ctx->cfg->use_dict;
    cparams.clevel = (uint8_t)ctx->cfg->compression_level; /* Since its just a mapping, we know the cast is ok */
    cparams.blocksize = ctx->cfg->blocksize;
    cparams.nthreads = (uint16_t)ctx->cfg->max_num_threads; /* Since its just a mapping, we know the cast is ok */
    if ((ctx->cfg->filter_flags & IARRAY_COMP_TRUNC_PREC) &&
        (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT || dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE)) {
        cparams.filters[blosc_filter_idx] = BLOSC_TRUNC_PREC;
        cparams.filters_meta[blosc_filter_idx] = ctx->cfg->fp_mantissa_bits;
        blosc_filter_idx++;
    }
    if (ctx->cfg->filter_flags & IARRAY_COMP_BITSHUFFLE) {
        cparams.filters[blosc_filter_idx] = BLOSC_BITSHUFFLE;
        blosc_filter_idx++;
    }
    if (ctx->cfg->filter_flags & IARRAY_COMP_SHUFFLE) {
        cparams.filters[blosc_filter_idx] = BLOSC_SHUFFLE;
        blosc_filter_idx++;
    }
    if (ctx->cfg->filter_flags & IARRAY_COMP_DELTA) {
        cparams.filters[blosc_filter_idx] = BLOSC_DELTA;
        blosc_filter_idx++;
    }
    ina_mem_cpy((*c)->cparams, &cparams, sizeof(blosc2_cparams));

    dparams.nthreads = (uint16_t)ctx->cfg->max_num_threads; /* Since its just a mapping, we know the cast is ok */
    ina_mem_cpy((*c)->dparams, &dparams, sizeof(blosc2_dparams));

    (*c)->storage = ina_mem_alloc(sizeof(iarray_storage_t));
    if ((*c)->storage == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the store parameters");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }
    ina_mem_cpy((*c)->storage, storage, sizeof(iarray_storage_t));

    caterva_config_t cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
    caterva_context_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

    caterva_params_t cat_params = {0};
    iarray_create_caterva_params(dtshape, &cat_params);

    caterva_storage_t cat_storage = {0};
    iarray_create_caterva_storage(dtshape, storage, &cat_storage);

    IARRAY_ERR_CATERVA(caterva_array_empty(cat_ctx, &cat_params, &cat_storage, &(*c)->catarr));

    if (cat_ctx != NULL) caterva_context_free(&cat_ctx);

    if ((*c)->catarr == NULL) {
        IARRAY_TRACE1(iarray.error, "Error creating the caterva container");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
    }

    if ((*c)->catarr->storage == CATERVA_STORAGE_BLOSC) {
        uint8_t *smeta;
        int32_t smeta_len = serialize_meta(dtshape->dtype, &smeta);
        if (smeta_len < 0) {
            IARRAY_TRACE1(iarray.error, "Error serializing the meta-information");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
        }
        // And store it in iarray metalayer
        if (blosc2_add_metalayer((*c)->catarr->sc, "iarray", smeta, (uint32_t) smeta_len) < 0) {
            IARRAY_TRACE1(iarray.error, "Error adding a metalayer to blosc");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
        }
        free(smeta);
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_container_free(ctx, c);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
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

    ina_rc_t rc;

    /* validation */
    if (dtshape->ndim > CATERVA_MAX_DIM) {
        IARRAY_TRACE1(iarray.error, "The container dimension is larger than the caterva maximum dimension");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_NDIM));
    }

    *c = (iarray_container_t*)ina_mem_alloc(sizeof(iarray_container_t));
    if (*c == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iarray container");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }

    (*c)->dtshape = (iarray_dtshape_t*)ina_mem_alloc(sizeof(iarray_dtshape_t));
    if ((*c)->dtshape == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iarray dtshape");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }
    ina_mem_cpy((*c)->dtshape, dtshape, sizeof(iarray_dtshape_t));

    iarray_auxshape_t auxshape;
    for (int i = 0; i < dtshape->ndim; ++i) {
        auxshape.shape_wos[i] = dtshape->shape[i];
        auxshape.pshape_wos[i] = pred->storage->pshape[i];
        auxshape.bshape_wos[i] = pred->storage->bshape[i];
        auxshape.offset[i] = offset[i];
        auxshape.index[i] = (uint8_t) i;
    }
    (*c)->auxshape = (iarray_auxshape_t*)ina_mem_alloc(sizeof(iarray_auxshape_t));
    if ((*c)->auxshape == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iarray auxdtshape");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }
    ina_mem_cpy((*c)->auxshape, &auxshape, sizeof(iarray_auxshape_t));

    (*c)->cparams = pred->cparams;
    (*c)->dparams = pred->dparams;
    (*c)->transposed = pred->transposed;
    (*c)->view = true;
    (*c)->storage = pred->storage;
    (*c)->catarr = pred->catarr;

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_container_free(ctx, c);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}

#endif
