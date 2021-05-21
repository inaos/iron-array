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

#include "iarray_private.h"
#include <libiarray/iarray.h>


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


static ina_rc_t iarray_create_caterva_structs(iarray_context_t *ctx,
                                              iarray_dtshape_t *dtshape,
                                              iarray_storage_t *storage,
                                              caterva_ctx_t **cat_ctx,
                                              caterva_params_t *cat_params,
                                              caterva_storage_t *cat_storage) {

    caterva_config_t cfg = {0};
    IARRAY_RETURN_IF_FAILED(iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg));

    blosc2_btune iabtune = {0};
    btune_config iabtune_config = {0};
    memcpy(&iabtune_config, &BTUNE_CONFIG_DEFAULTS, sizeof(btune_config));
    switch(ctx->cfg->compression_favor) {
        case IARRAY_COMPRESSION_FAVOR_CRATIO:
            iabtune_config.comp_mode = BTUNE_COMP_HCR;
            break;
        case IARRAY_COMPRESSION_FAVOR_SPEED:
            iabtune_config.comp_mode = BTUNE_COMP_HSP;
            break;
        default:
            iabtune_config.comp_mode = BTUNE_COMP_BALANCED;
    }
    if (ctx->cfg->btune) {
        iabtune.btune_config = &iabtune_config;
        iabtune.btune_init = iabtune_init;
        iabtune.btune_next_blocksize = iabtune_next_blocksize;
        iabtune.btune_next_cparams = iabtune_next_cparams;
        iabtune.btune_update = iabtune_update;
        iabtune.btune_free = iabtune_free;
        cfg.udbtune = &iabtune;
    }

    if (storage->backend == IARRAY_STORAGE_BLOSC) {
        uint8_t *smeta;
        int32_t smeta_len = serialize_meta(dtshape->dtype, &smeta);
        if (smeta_len < 0) {
            IARRAY_TRACE1(iarray.error, "Error serializing the meta-information");
            return INA_ERROR(INA_ERR_FAILED);
        }

        IARRAY_ERR_CATERVA(caterva_ctx_new(&cfg, cat_ctx));
        IARRAY_RETURN_IF_FAILED(iarray_create_caterva_params(dtshape, cat_params));
        IARRAY_RETURN_IF_FAILED(iarray_create_caterva_storage(dtshape, storage, cat_storage));
        cat_storage->properties.blosc.nmetalayers = 1;
        caterva_metalayer_t metalayer = cat_storage->properties.blosc.metalayers[0];
        metalayer.name = strdup("iarray");
        metalayer.sdata = smeta;
        metalayer.size = smeta_len;
    }
    return INA_SUCCESS;
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

    /* validation */
    if (dtshape->ndim > CATERVA_MAX_DIM) {
        IARRAY_TRACE1(iarray.error, "The container dimension is larger than caterva maximum dimension");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }
    if (flags & IARRAY_CONTAINER_PERSIST && storage->urlpath == NULL) {
        IARRAY_TRACE1(iarray.error, "Error with persistency flags");
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }
    if (storage->backend == IARRAY_STORAGE_BLOSC) {
        for (int i = 0; i < dtshape->ndim; ++i) {
            if (dtshape->shape[i] < storage->chunkshape[i]) {
                IARRAY_TRACE1(iarray.error, "The chunkshape is larger than the shape");
                return INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE);
            }
        }
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
    ina_mem_cpy((*c)->dtshape, dtshape, sizeof(iarray_dtshape_t));

    if (storage->backend == IARRAY_STORAGE_PLAINBUFFER) {
        for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
            storage->chunkshape[i] = dtshape->shape[i];
            storage->blockshape[i] = dtshape->shape[i];
        }
    }

    iarray_auxshape_t auxshape;
    for (int i = 0; i < dtshape->ndim; ++i) {
        auxshape.shape_wos[i] = dtshape->shape[i];
        auxshape.chunkshape_wos[i] = storage->chunkshape[i];
        auxshape.blockshape_wos[i] = storage->blockshape[i];
        auxshape.offset[i] = 0;
        auxshape.index[i] = (uint8_t) i;
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
    (*c)->view = false;
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
        auxshape.index[i] = (uint8_t) i;
    }
    (*c)->auxshape = (iarray_auxshape_t*)ina_mem_alloc(sizeof(iarray_auxshape_t));
    if ((*c)->auxshape == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iarray auxdtshape");
        return INA_ERROR(INA_ERR_FAILED);
    }
    ina_mem_cpy((*c)->auxshape, &auxshape, sizeof(iarray_auxshape_t));

    (*c)->view = true;
    (*c)->transposed = false;

    (*c)->storage = pred->storage;
    (*c)->catarr = pred->catarr;

    return INA_SUCCESS;
}

#endif
