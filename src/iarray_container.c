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


static ina_rc_t deserialize_meta(uint8_t *smeta, uint32_t smeta_len, iarray_data_type_t *dtype, bool *transposed) {
    INA_UNUSED(smeta_len);
    INA_VERIFY_NOT_NULL(smeta);
    INA_VERIFY_NOT_NULL(dtype);
    INA_VERIFY_NOT_NULL(transposed);
    ina_rc_t rc;

    uint8_t *pmeta = smeta;

    //version
    uint8_t version = *pmeta;
    INA_UNUSED(version);
    pmeta +=1;

    // We only have an entry with the datatype (enumerated < 128)
    *dtype = *pmeta;
    pmeta += 1;

    *transposed = false;
    if ((*pmeta & 64ULL) != 0) {
        *transposed = true;
    }
    pmeta += 1;

    assert(pmeta - smeta == smeta_len);

    if (*dtype >= IARRAY_DATA_TYPE_MAX) {
        IARRAY_TRACE1(iarray.error, "The data type is invalid");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}

INA_API(ina_rc_t) iarray_container_dtshape_equal(iarray_dtshape_t *a, iarray_dtshape_t *b)
{
    ina_rc_t rc;
    if (a->dtype != b->dtype) {
        IARRAY_TRACE1(iarray.error, "The data types are not equal");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }
    if (a->ndim != b->ndim) {
        IARRAY_TRACE1(iarray.error, "The dimensions are not equal");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_NDIM));
    }
    for (int i = 0; i < CATERVA_MAX_DIM; ++i) {
        if (a->shape[i] != b->shape[i]) {
            IARRAY_TRACE1(iarray.error, "The shapes are not equal\n");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_SHAPE));
        }
    }
    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}


INA_API(ina_rc_t) iarray_container_new(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    return _iarray_container_new(ctx, dtshape, store, flags, container);
}


INA_API(ina_rc_t) iarray_container_save(iarray_context_t *ctx,
                                        iarray_container_t *container,
                                        char *filename) {
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(filename);

    if (container->catarr->storage != CATERVA_STORAGE_BLOSC) {
        IARRAY_TRACE1(iarray.error, "Container must be stored on a blosc schunk");
        return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
    }

    if (container->catarr->sc->frame == NULL) {
        blosc2_frame *frame = blosc2_new_frame(filename);
        if (frame == NULL) {
            IARRAY_TRACE1(iarray.error, "Error creating blosc2 frame");
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }
        int err = blosc2_schunk_to_frame(container->catarr->sc, frame);

        if (err < 0) {
            IARRAY_TRACE1(iarray.error, "Error converting a blosc schunk to a blosc frame");
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }
        free(frame);
    } else {
        if (container->catarr->sc->frame->fname != NULL) {
            IARRAY_TRACE1(iarray.error, "Container is already on disk");
            return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
        } else {
            blosc2_frame_to_file(container->catarr->sc->frame, filename);
        }
    }
    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_container_load(iarray_context_t *ctx, char *filename, bool enforce_frame,
                                        iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(filename);
    INA_VERIFY_NOT_NULL(container);

    ina_rc_t rc;

    caterva_config_t cfg;
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
    caterva_context_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));
    if (cat_ctx == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the caterva context");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
    }

    caterva_array_t *catarr;
    IARRAY_ERR_CATERVA(caterva_array_from_file(cat_ctx, filename, enforce_frame, &catarr));

    if (catarr == NULL) {
        IARRAY_TRACE1(iarray.error, "Error creating the caterva array from a file");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
    }

    uint8_t *smeta;
    uint32_t smeta_len;
    if (blosc2_get_metalayer(catarr->sc, "iarray", &smeta, &smeta_len) < 0) {
        IARRAY_TRACE1(iarray.error, "Error getting a blosc metalayer");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
    }
    iarray_data_type_t dtype;
    bool transposed;
    if (deserialize_meta(smeta, smeta_len, &dtype, &transposed) != 0) {
        IARRAY_TRACE1(iarray.error, "Error deserializing a sframe");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
    }

    *container = (iarray_container_t*)ina_mem_alloc(sizeof(iarray_container_t));
    (*container)->catarr = catarr;

    // Build the dtshape
    (*container)->dtshape = (iarray_dtshape_t*)ina_mem_alloc(sizeof(iarray_dtshape_t));
    iarray_dtshape_t* dtshape = (*container)->dtshape;
    dtshape->dtype = dtype;
    dtshape->ndim = catarr->ndim;
    for (int i = 0; i < catarr->ndim; ++i) {
        dtshape->shape[i] = catarr->shape[i];
        dtshape->pshape[i] = catarr->chunkshape[i];
    }

    // Build the auxshape
    (*container)->auxshape = (iarray_auxshape_t*)ina_mem_alloc(sizeof(iarray_auxshape_t));
    iarray_auxshape_t* auxshape = (*container)->auxshape;
    for (int8_t i = 0; i < catarr->ndim; ++i) {
        auxshape->index[i] = i;
        auxshape->offset[i] = 0;
        auxshape->shape_wos[i] = catarr->shape[i];
        auxshape->pshape_wos[i] = catarr->chunkshape[i];
    }

    // Populate compression parameters
    blosc2_cparams *cparams;
    if (blosc2_schunk_get_cparams(catarr->sc, &cparams) < 0) {
        IARRAY_TRACE1(iarray.error, "Error getting the cparams from blosc2 schunk");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
    }
    blosc2_cparams *cparams2 = (blosc2_cparams*)ina_mem_alloc(sizeof(blosc2_cparams));
    memcpy(cparams2, cparams, sizeof(blosc2_cparams));
    free(cparams);
    (*container)->cparams = cparams2;  // we need an INA-allocated struct (to match INA_MEM_FREE_SAFE)
    blosc2_dparams *dparams;
    if (blosc2_schunk_get_dparams(catarr->sc, &dparams) < 0) {
        IARRAY_TRACE1(iarray.error, "Error getting the dparams from blosc2 schunk");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
    }
    blosc2_dparams *dparams2 = (blosc2_dparams*)ina_mem_alloc(sizeof(blosc2_dparams));
    memcpy(dparams2, dparams, sizeof(blosc2_dparams));
    free(dparams);
    (*container)->dparams = dparams2;  // we need an INA-allocated struct (to match INA_MEM_FREE_SAFE)

    (*container)->transposed = transposed;  // TODO: complete this
    if (transposed) {
        int64_t aux[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < (*container)->dtshape->ndim; ++i) {
            aux[i] = (*container)->dtshape->shape[i];
        }
        for (int i = 0; i < (*container)->dtshape->ndim; ++i) {
            (*container)->dtshape->shape[i] = aux[(*container)->dtshape->ndim - 1 - i];
        }
        for (int i = 0; i < (*container)->dtshape->ndim; ++i) {
            aux[i] = (*container)->dtshape->pshape[i];
        }
        for (int i = 0; i < (*container)->dtshape->ndim; ++i) {
            (*container)->dtshape->pshape[i] = aux[(*container)->dtshape->ndim - 1 - i];
        }
    }
    (*container)->view = false;

    (*container)->store = ina_mem_alloc(sizeof(iarray_store_properties_t));
    if ((*container)->store == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the store parameter");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }
    (*container)->store->filename = filename;
    (*container)->store->backend = IARRAY_STORAGE_BLOSC;
    (*container)->store->enforce_frame = enforce_frame;

    free(smeta);

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_container_free(ctx, container);
    rc = ina_err_get_rc();
    cleanup:
    caterva_context_free(&cat_ctx);
    return rc;
}


INA_API(ina_rc_t) iarray_get_slice(iarray_context_t *ctx,
                                   iarray_container_t *src,
                                   int64_t *start,
                                   int64_t *stop,
                                   bool view,
                                   const int64_t *pshape,
                                   iarray_store_properties_t *store,
                                   int flags,
                                   iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(src);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    ina_rc_t rc;

    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t stop_[IARRAY_DIMENSION_MAX];

    int64_t *offset = src->auxshape->offset;

    for (int i = 0; i < src->dtshape->ndim; ++i) {
        if (start[i] < 0) {
            start_[i] =  offset[i] + start[i] + src->dtshape->shape[i];
        } else{
            start_[i] = offset[i] + (int64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[i] =  offset[i] + stop[i] + src->dtshape->shape[i];
        } else {
            stop_[i] = offset[i] + (int64_t) stop[i];
        }
    }

    for (int i = 0; i < src->dtshape->ndim; ++i) {
        if (start_[i] >= stop_[i]) {
            IARRAY_TRACE1(iarray.error, "Start is bigger than stop");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
        }
        if (store->backend == IARRAY_STORAGE_BLOSC && pshape[i] > stop_[i] - start_[i]){
            IARRAY_TRACE1(iarray.error, "The pshape is bigger than shape");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_PSHAPE));
        }
    }

    if (view) {
        iarray_dtshape_t dtshape;
        dtshape.ndim = src->dtshape->ndim;
        dtshape.dtype = src->dtshape->dtype;

        for (int i = 0; i < dtshape.ndim; ++i) {
            dtshape.shape[i] = stop_[i] - start_[i];
            dtshape.pshape[i] = pshape ? pshape[i] : 0;
        }

        IARRAY_FAIL_IF_ERROR(_iarray_view_new(ctx, src, &dtshape, start_, container));

        (*container)->view = 1;
        if (src->transposed == 1) {
            (*container)->transposed = 1;
        }

    } else {
        iarray_dtshape_t dtshape;

        dtshape.ndim = src->dtshape->ndim;
        dtshape.dtype = src->dtshape->dtype;

        for (int i = 0; i < dtshape.ndim; ++i) {
            dtshape.shape[i] = stop_[i] - start_[i];
            if (pshape)
                dtshape.pshape[i] = pshape[i];
        }

        iarray_container_new(ctx, &dtshape, store, flags, container);

        // Check if matrix is transposed
        if (src->transposed) {
            int64_t aux_stop[IARRAY_DIMENSION_MAX];
            int64_t aux_start[IARRAY_DIMENSION_MAX];

            for (int i = 0; i < src->dtshape->ndim; ++i) {
                aux_start[i] = start_[i];
                aux_stop[i] = stop_[i];
            }

            for (int i = 0; i < src->dtshape->ndim; ++i) {
                start_[i] = aux_start[src->dtshape->ndim - 1 - i];
                stop_[i] = aux_stop[src->dtshape->ndim - 1 - i];
            }
        }

        if (src->transposed) {
            (*container)->transposed = true;
        }

        caterva_config_t cfg = {0};
        iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
        caterva_context_t *cat_ctx;
        IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

        caterva_storage_t storage = {0};
        iarray_create_caterva_storage(&dtshape, store, &storage);

        caterva_array_free(cat_ctx, &(*container)->catarr);

        IARRAY_ERR_CATERVA(caterva_array_get_slice(cat_ctx, src->catarr, start_, stop_, &storage, &(*container)->catarr));
        IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_container_free(ctx, container);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}

INA_API(ina_rc_t) iarray_set_slice(iarray_context_t *ctx,
                                   iarray_container_t *container,
                                   const int64_t *start,
                                   const int64_t *stop,
                                   iarray_container_t *slice)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(slice);

    ina_rc_t rc;

    if (container->dtshape->dtype != slice->dtshape->dtype) {
        IARRAY_TRACE1(iarray.error, "The data types are different");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }
    if (container->dtshape->ndim != slice->dtshape->ndim) {
        IARRAY_TRACE1(iarray.error, "The dimensions are different");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_NDIM));
    }

    int typesize = slice->catarr->itemsize;
    int64_t buflen = slice->catarr->size;

    uint8_t *buffer;
    if (slice->catarr->storage == CATERVA_STORAGE_BLOSC) {
        buffer = ina_mem_alloc(buflen * typesize);
        IARRAY_FAIL_IF_ERROR(iarray_to_buffer(ctx, slice, buffer, buflen * typesize));
    } else {
        buffer = slice->catarr->buf;
    }

    IARRAY_FAIL_IF_ERROR(iarray_set_slice_buffer(ctx, container, start, stop, buffer, buflen * typesize));

    if (slice->catarr->storage == CATERVA_STORAGE_BLOSC) {
        INA_MEM_FREE_SAFE(buffer);
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    if (slice->catarr->storage == CATERVA_STORAGE_BLOSC) {
        INA_MEM_FREE_SAFE(buffer);
    }
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}


INA_API(ina_rc_t) iarray_get_slice_buffer(iarray_context_t *ctx,
                                          iarray_container_t *container,
                                          const int64_t *start,
                                          const int64_t *stop,
                                          void *buffer,
                                          const int64_t buflen)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(buffer);

    ina_rc_t rc;

    int8_t ndim = container->dtshape->ndim;
    int64_t *offset = container->auxshape->offset;
    int8_t *index = container->auxshape->index;

    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t stop_[IARRAY_DIMENSION_MAX];

    for (int i = 0; i < container->catarr->ndim; ++i) {
        start_[i] = 0 + offset[i];
        stop_[i] = 1 + offset[i];
    }

    for (int i = 0; i < ndim; ++i) {
        if (start[i] < 0) {
            start_[index[i]] += start[i] + container->dtshape->shape[i];
        } else{
            start_[index[i]] += (int64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[index[i]] += stop[i] + container->dtshape->shape[i] - 1;
        } else {
            stop_[index[i]] += (int64_t) stop[i] - 1;
        }
    }

    for (int i = 0; i < container->dtshape->ndim; ++i) {
        if (start_[i] >= stop_[i]) {
            IARRAY_TRACE1(iarray.error, "Start is bigger than stop");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
        }
    }


    if (container->transposed) {
        int64_t aux_stop[IARRAY_DIMENSION_MAX];
        int64_t aux_start[IARRAY_DIMENSION_MAX];

        for (int i = 0; i < container->dtshape->ndim; ++i) {
            aux_start[i] = start_[i];
            aux_stop[i] = stop_[i];
        }

        for (int i = 0; i < container->dtshape->ndim; ++i) {
            start_[i] = aux_start[container->dtshape->ndim - 1 - i];
            stop_[i] = aux_stop[container->dtshape->ndim - 1 - i];
        }
    }

    int64_t pshape[IARRAY_DIMENSION_MAX];
    int64_t psize = 1;
    for (int i = 0; i < container->catarr->ndim; ++i) {
        pshape[i] = stop_[i] - start_[i];
        psize *= pshape[i];
    }

    if (container->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        if (psize * (int64_t)sizeof(double) > buflen) {
            IARRAY_TRACE1(iarray.error, "The buffer size is not enough\n");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
        }
    } else {
        if (psize * (int64_t)sizeof(float) > buflen) {
            IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
        }
    }

    caterva_config_t cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
    caterva_context_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

    IARRAY_ERR_CATERVA(caterva_array_get_slice_buffer(cat_ctx, container->catarr, start_, stop_, pshape, buffer, buflen));

    IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));

    size_t rows = (size_t)stop_[0] - start_[0];
    size_t cols = (size_t)stop_[1] - start_[1];
    if (container->transposed) {
        switch (container->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_dimatcopy('R', 'T', rows, cols, 1.0, (double *) buffer, cols, rows);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_simatcopy('R', 'T', rows, cols, 1.0f, (float *) buffer, cols, rows);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}


INA_API(ina_rc_t) iarray_set_slice_buffer(iarray_context_t *ctx,
                                          iarray_container_t *c,
                                          const int64_t *start,
                                          const int64_t *stop,
                                          void *buffer,
                                          const int64_t buflen)
{
    // TODO: make use of buflen so as to avoid exceeding the buffer boundaries
    INA_UNUSED(ctx);
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(buffer);

    ina_rc_t rc;

    if (c->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
        IARRAY_TRACE1(iarray.error, "The container is not backed by a plainbuffer");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_STORAGE));
    }

    int8_t ndim = c->dtshape->ndim;
    int64_t *offset = c->auxshape->offset;
    int8_t *index = c->auxshape->index;

    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t stop_[IARRAY_DIMENSION_MAX];

    for (int i = 0; i < c->catarr->ndim; ++i) {
        start_[i] = 0 + offset[i];
        stop_[i] = 1 + offset[i];
    }

    for (int i = 0; i < ndim; ++i) {
        if (start[i] < 0) {
            start_[index[i]] += start[i] + c->dtshape->shape[i];
        } else{
            start_[index[i]] += (int64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[index[i]] += stop[i] + c->dtshape->shape[i] - 1;
        } else {
            stop_[index[i]] += (int64_t) stop[i] - 1;
        }
    }

    for (int i = 0; i < c->catarr->ndim; ++i) {
        if (start_[i] < 0) {
            IARRAY_TRACE1(iarray.error, "Start is negative");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
        }
        if (stop_[i] < start_[i]) {
            IARRAY_TRACE1(iarray.error, "Start is larger than stop");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
        }
        if (c->catarr->shape[i] < stop_[i]) {
            IARRAY_TRACE1(iarray.error, "Stop is larger than the container shape");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
        }
    }

    if (c->transposed) {
        size_t rows = (size_t)stop_[0] - start_[0];
        size_t cols = (size_t)stop_[1] - start_[1];
        switch (c->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_dimatcopy('R', 'T', rows, cols, 1.0, (double *) buffer, cols, rows);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_simatcopy('R', 'T', rows, cols, 1.0f, (float *) buffer, cols, rows);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }
    }

    if (c->transposed) {
        int64_t aux_stop[IARRAY_DIMENSION_MAX];
        int64_t aux_start[IARRAY_DIMENSION_MAX];

        for (int i = 0; i < c->dtshape->ndim; ++i) {
            aux_start[i] = start_[i];
            aux_stop[i] = stop_[i];
        }

        for (int i = 0; i < c->dtshape->ndim; ++i) {
            start_[i] = aux_start[c->dtshape->ndim - 1 - i];
            stop_[i] = aux_stop[c->dtshape->ndim - 1 - i];
        }
    }

    int64_t psize = 1;
    for (int i = 0; i < c->catarr->ndim; ++i) {
        psize *= stop_[i] - start_[i];
    }

    switch (c->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if (psize * (int64_t)sizeof(double) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if (psize * (int64_t)sizeof(float) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            }
            break;
        default:
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    caterva_config_t cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
    caterva_context_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

    IARRAY_ERR_CATERVA(caterva_array_set_slice_buffer(cat_ctx, buffer, buflen, start_, stop_, c->catarr));

    IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));
    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}


int _caterva_get_slice_buffer_no_copy(void **dest, caterva_array_t *src, int64_t *start,
                                      int64_t *stop, int64_t *d_pshape) {
    CATERVA_UNUSED_PARAM(d_pshape);
    int64_t start_[CATERVA_MAX_DIM];
    int64_t stop_[CATERVA_MAX_DIM];
    int8_t s_ndim = src->ndim;

    int64_t *shape = src->shape;
    int64_t s_shape[CATERVA_MAX_DIM];
    for (int i = 0; i < CATERVA_MAX_DIM; ++i) {
        start_[(CATERVA_MAX_DIM - s_ndim + i) % CATERVA_MAX_DIM] = i < s_ndim ? start[i] : 1;
        stop_[(CATERVA_MAX_DIM - s_ndim + i) % CATERVA_MAX_DIM] = i < s_ndim ? stop[i] : 1;
        s_shape[(CATERVA_MAX_DIM - s_ndim + i) % CATERVA_MAX_DIM] = i < s_ndim ? shape[i] : 1;
    }
    for (int j = 0; j < CATERVA_MAX_DIM - s_ndim; ++j) {
        start_[j] = 0;
    }

    int64_t chunk_pointer = 0;
    int64_t chunk_pointer_inc = 1;
    for (int i = CATERVA_MAX_DIM - 1; i >= 0; --i) {
        chunk_pointer += start_[i] * chunk_pointer_inc;
        chunk_pointer_inc *= s_shape[i];
    }
    *dest = &src->buf[chunk_pointer * src->itemsize];

    return 0;
}



INA_API(ina_rc_t) _iarray_get_slice_buffer_no_copy(iarray_context_t *ctx,
                                                   iarray_container_t *c,
                                                   int64_t *start,
                                                   int64_t *stop,
                                                   void **buffer,
                                                   int64_t buflen)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(buffer);

    ina_rc_t rc;

    int8_t ndim = c->dtshape->ndim;
    int64_t *offset = c->auxshape->offset;
    int8_t *index = c->auxshape->index;

    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t stop_[IARRAY_DIMENSION_MAX];

    for (int i = 0; i < c->catarr->ndim; ++i) {
        start_[i] = 0 + offset[i];
        stop_[i] = 1 + offset[i];
    }

    for (int i = 0; i < ndim; ++i) {
        if (start[i] < 0) {
            start_[index[i]] += start[i] + c->dtshape->shape[i];
        } else{
            start_[index[i]] += (int64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[index[i]] += stop[i] + c->dtshape->shape[i] - 1;
        } else {
            stop_[index[i]] += (int64_t) stop[i] - 1;
        }
    }

    if (c->transposed) {
        int64_t aux_stop[IARRAY_DIMENSION_MAX];
        int64_t aux_start[IARRAY_DIMENSION_MAX];

        for (int i = 0; i < c->dtshape->ndim; ++i) {
            aux_start[i] = start_[i];
            aux_stop[i] = stop_[i];
        }

        for (int i = 0; i < c->dtshape->ndim; ++i) {
            start_[i] = aux_start[c->dtshape->ndim - 1 - i];
            stop_[i] = aux_stop[c->dtshape->ndim - 1 - i];
        }
    }

    int64_t pshape[IARRAY_DIMENSION_MAX];
    int64_t psize = 1;
    for (int i = 0; i < c->catarr->ndim; ++i) {
        pshape[i] = stop_[i] - start_[i];
        psize *= pshape[i];
    }

    switch (c->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if (psize * (int64_t)sizeof(double) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if (psize * (int64_t)sizeof(float) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            }
            break;
        default:
            IARRAY_TRACE1(iarray.error, "the data type is invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }



    IARRAY_ERR_CATERVA(_caterva_get_slice_buffer_no_copy(buffer, c->catarr, start_, stop_, pshape));

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}

ina_rc_t _iarray_get_slice_buffer(iarray_context_t *ctx,
                                  iarray_container_t *c,
                                  int64_t *start,
                                  int64_t *stop,
                                  int64_t *pshape,
                                  void *buffer,
                                  int64_t buflen)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(pshape);

    ina_rc_t rc;

    int8_t ndim = c->dtshape->ndim;
    int64_t *offset = c->auxshape->offset;
    int8_t *index = c->auxshape->index;

    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t stop_[IARRAY_DIMENSION_MAX];
    int64_t pshape_[IARRAY_DIMENSION_MAX];

    for (int i = 0; i < c->catarr->ndim; ++i) {
        start_[i] = 0 + offset[i];
        stop_[i] = 1 + offset[i];
        pshape_[i] = 1;
    }

    for (int i = 0; i < ndim; ++i) {
        pshape_[i] = pshape[i];
        if (start[i] < 0) {
            start_[index[i]] += start[i] + c->dtshape->shape[i];
        } else{
            start_[index[i]] += (int64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[index[i]] += stop[i] + c->dtshape->shape[i] - 1;
        } else {
            stop_[index[i]] += (int64_t) stop[i] - 1;
        }
    }

    if (c->transposed) {
        int64_t aux_stop[IARRAY_DIMENSION_MAX];
        int64_t aux_start[IARRAY_DIMENSION_MAX];
        int64_t aux_pshape[IARRAY_DIMENSION_MAX];

        for (int i = 0; i < c->catarr->ndim; ++i) {
            aux_start[i] = start_[i];
            aux_stop[i] = stop_[i];
            aux_pshape[i] = pshape_[i];
        }

        for (int i = 0; i < c->catarr->ndim; ++i) {
            start_[i] = aux_start[c->catarr->ndim - 1 - i];
            stop_[i] = aux_stop[c->catarr->ndim - 1 - i];
            pshape_[i] = aux_pshape[c->catarr->ndim - 1 - i];
        }
    }

    int64_t psize = 1;
    for (int i = 0; i < ndim; ++i) {
        psize *= pshape[i];
    }

    switch (c->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if (psize * (int64_t)sizeof(double) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if (psize * (int64_t)sizeof(float) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            }
            break;
        default:
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }



    memset(buffer, 0, buflen);

    caterva_config_t cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
    caterva_context_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

    IARRAY_ERR_CATERVA(caterva_array_get_slice_buffer(cat_ctx, c->catarr, start_, stop_, pshape_, buffer, buflen));

    IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}

INA_API(ina_rc_t) iarray_squeeze(iarray_context_t *ctx,
                                 iarray_container_t *container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);

    uint8_t inc = 0;

    if (!container->view) {
        caterva_config_t cfg = {0};
        iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
        caterva_context_t *cat_ctx;
        IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

        IARRAY_ERR_CATERVA(caterva_array_squeeze(cat_ctx, container->catarr));

        IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));
        if (container->dtshape->ndim != container->catarr->ndim) {
            container->dtshape->ndim = (uint8_t) container->catarr->ndim;
            for (int i = 0; i < container->catarr->ndim; ++i) {
                if (container->dtshape->shape[i] != container->catarr->shape[i]) {
                    inc += 1;
                }
                container->dtshape->shape[i] = container->catarr->shape[i];
                container->dtshape->pshape[i] = container->catarr->chunkshape[i];
                container->auxshape->shape_wos[i] = container->catarr->shape[i];
                container->auxshape->pshape_wos[i] = container->catarr->chunkshape[i];
                container->auxshape->offset[i] = container->auxshape->offset[i + inc];
            }
        }
    } else {
        inc = 0;
        for (int i = 0; i < container->dtshape->ndim; ++i) {
            if (container->dtshape->shape[i] == 1) {
                inc ++;
            } else {
                container->dtshape->shape[i - inc] = container->dtshape->shape[i];
                container->dtshape->pshape[i - inc] = container->dtshape->pshape[i];
                container->auxshape->index[i - inc] = (uint8_t) i;
            }
        }
        container->dtshape->ndim -= inc;
    }

    return INA_SUCCESS;
    fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_get_dtshape(iarray_context_t *ctx,
                                     iarray_container_t *c,
                                     iarray_dtshape_t *dtshape)
{
    INA_UNUSED(ctx);
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(dtshape);

    dtshape->ndim = c->dtshape->ndim;
    dtshape->dtype = c->dtshape->dtype;
    for (int i = 0; i < c->dtshape->ndim; ++i) {
        dtshape->shape[i] = c->dtshape->shape[i];
        dtshape->pshape[i] = c->dtshape->pshape[i];
    }
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_container_info(iarray_container_t *container, int64_t *nbytes, int64_t *cbytes)
{
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(nbytes);
    INA_VERIFY_NOT_NULL(cbytes);

    if (container->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        *nbytes = container->catarr->size * container->catarr->itemsize;
        *cbytes = *nbytes;
    }
    else {
        *nbytes = container->catarr->sc->nbytes;
        *cbytes = container->catarr->sc->cbytes;
    }

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_container_almost_equal(iarray_container_t *a, iarray_container_t *b, double tol)
{
    ina_rc_t rc;

    if (a->dtshape->dtype != b->dtshape->dtype){
        IARRAY_TRACE1(iarray.error, "The data types are not equals");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }
    if (a->dtshape->ndim != b->dtshape->ndim) {
        IARRAY_TRACE1(iarray.error, "The dimensions are not equals");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_NDIM));
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        if (a->dtshape->shape[i] != b->dtshape->shape[i]) {
            IARRAY_TRACE1(iarray.error, "The shapes are not equals");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_SHAPE));
        }
    }

    int dtype = a->dtshape->dtype;
    int ndim = a->dtshape->ndim;

    // For the blocksize, choose the maximum of the partition shapes
    int64_t blocksize[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ndim; ++i) {
        blocksize[i] = INA_MAX(a->dtshape->pshape[i], b->dtshape->pshape[i]);
    }

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_t *ctx = NULL;
    IARRAY_FAIL_IF_ERROR(iarray_context_new(&cfg, &ctx));
    iarray_iter_read_block_t *iter_a;
    iarray_iter_read_block_value_t val_a;
    IARRAY_FAIL_IF_ERROR(iarray_iter_read_block_new(ctx, &iter_a, a, blocksize, &val_a, false));
    iarray_iter_read_block_t *iter_b;
    iarray_iter_read_block_value_t val_b;
    IARRAY_FAIL_IF_ERROR(iarray_iter_read_block_new(ctx, &iter_b, b, blocksize, &val_b, false));

    while (INA_SUCCEED(iarray_iter_read_block_has_next(iter_a))) {
        IARRAY_FAIL_IF_ERROR(iarray_iter_read_block_next(iter_a, NULL, 0));
        IARRAY_FAIL_IF_ERROR(iarray_iter_read_block_next(iter_b, NULL, 0));

        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (int64_t i = 0; i < val_a.block_size; ++i) {
                double adiff = fabs(((double *)val_a.block_pointer)[i] - ((double *)val_b.block_pointer)[i]);
                double rdiff = fabs(((double *)val_a.block_pointer)[i] - ((double *)val_b.block_pointer)[i]) /
                    ((double *)val_a.block_pointer)[i];
                if (rdiff > tol) {
                    printf("%f, %f (adiff: %f, rdiff: %f)\n", ((double *)val_a.block_pointer)[i],
                        ((double *)val_b.block_pointer)[i], adiff, rdiff);
                    IARRAY_TRACE1(iarray.error, "Values are different");
                    IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_ASSERTION_FAILED));
                }
            }
        }
        else {
            for (int64_t i = 0; i < val_a.block_size; ++i) {
                float adiff = fabsf(((float *)val_a.block_pointer)[i] - ((float *)val_b.block_pointer)[i]);
                float rdiff = fabsf(((float *)val_a.block_pointer)[i] - ((float *)val_b.block_pointer)[i]) /
                    ((float *)val_a.block_pointer)[i];
                if (rdiff > tol) {
                    printf("%f, %f (adiff: %f, rdiff: %f)\n", ((float *)val_a.block_pointer)[i],
                           ((float *)val_b.block_pointer)[i], adiff, rdiff);
                    IARRAY_TRACE1(iarray.error, "Values are different");
                    IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_ASSERTION_FAILED));
                }
            }
        }
    }

    IARRAY_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    iarray_iter_read_block_free(&iter_a);
    iarray_iter_read_block_free(&iter_b);
    iarray_context_free(&ctx);
    return rc;
}


INA_API(void) iarray_container_free(iarray_context_t *ctx, iarray_container_t **container)
{
    INA_UNUSED(ctx);
    INA_VERIFY_FREE(container);

    if (!(*container)->view) {
        if ((*container)->catarr != NULL) {
            caterva_config_t cfg = {0};
            iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
            caterva_context_t *cat_ctx;
            caterva_context_new(&cfg, &cat_ctx);
            caterva_array_free(cat_ctx, &(*container)->catarr);
            caterva_context_free(&cat_ctx);
        }
        INA_MEM_FREE_SAFE((*container)->cparams);
        INA_MEM_FREE_SAFE((*container)->dparams);
        INA_MEM_FREE_SAFE((*container)->store);

    }
    INA_MEM_FREE_SAFE((*container)->dtshape);
    INA_MEM_FREE_SAFE((*container)->auxshape);

    INA_MEM_FREE_SAFE(*container);
}

INA_API(ina_rc_t) iarray_container_gt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}

INA_API(ina_rc_t) iarray_container_lt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}

INA_API(ina_rc_t) iarray_container_gte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}

INA_API(ina_rc_t) iarray_container_lte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}

INA_API(ina_rc_t) iarray_container_eq(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}
