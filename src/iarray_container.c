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
#include "iarray_constructor.h"


static ina_rc_t deserialize_meta(uint8_t *smeta, uint32_t smeta_len, iarray_data_type_t *dtype) {
    INA_UNUSED(smeta_len);
    INA_VERIFY_NOT_NULL(smeta);
    INA_VERIFY_NOT_NULL(dtype);

    uint8_t *pmeta = smeta;

    //version
    uint8_t version = *pmeta;
    INA_USED_BY_ASSERT(version);
    pmeta +=1;

    // We only have an entry with the datatype (enumerated < 128)
    *dtype = *pmeta;
    pmeta += 1;

   // Transpose byte
    pmeta += 1;

    assert(pmeta - smeta == smeta_len);

    if (*dtype >= IARRAY_DATA_TYPE_MAX) {
        IARRAY_TRACE1(iarray.error, "The data type is invalid");
        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_container_dtshape_equal(iarray_dtshape_t *a, iarray_dtshape_t *b)
{
    if (a->dtype != b->dtype) {
        IARRAY_TRACE1(iarray.error, "The data types are not equal");
        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }
    if (a->ndim != b->ndim) {
        IARRAY_TRACE1(iarray.error, "The dimensions are not equal");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }
    for (int i = 0; i < CATERVA_MAX_DIM; ++i) {
        if (a->shape[i] != b->shape[i]) {
            IARRAY_TRACE1(iarray.error, "The shapes are not equal\n");
            return INA_ERROR(IARRAY_ERR_INVALID_SHAPE);
        }
    }
    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_container_new(iarray_context_t *ctx,
                                       iarray_dtshape_t *dtshape,
                                       iarray_storage_t *storage,
                                       int flags,
                                       iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    return _iarray_container_new(ctx, dtshape, storage, flags, container);
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
        blosc2_frame *frame = blosc2_frame_new(filename);
        if (frame == NULL) {
            IARRAY_TRACE1(iarray.error, "Error creating blosc2 frame");
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }
        int64_t err = blosc2_frame_from_schunk(container->catarr->sc, frame);

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

    if (access( filename, 0) == -1) {
        IARRAY_TRACE1(iarray.error, "File not exists");
        return INA_ERROR(INA_ERR_FILE_OPEN);
    }
    caterva_config_t cfg;
    IARRAY_RETURN_IF_FAILED(iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg));
    caterva_context_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));
    if (cat_ctx == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the caterva context");
        return INA_ERROR(IARRAY_ERR_CATERVA_FAILED);
    }

    caterva_array_t *catarr;
    IARRAY_ERR_CATERVA(caterva_array_from_file(cat_ctx, filename, enforce_frame, &catarr));

    if (catarr == NULL) {
        IARRAY_TRACE1(iarray.error, "Error creating the caterva array from a file");
        return INA_ERROR(IARRAY_ERR_CATERVA_FAILED);
    }

    uint8_t *smeta;
    uint32_t smeta_len;
    if (blosc2_get_metalayer(catarr->sc, "iarray", &smeta, &smeta_len) < 0) {
        IARRAY_TRACE1(iarray.error, "Error getting a blosc metalayer");
        return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
    }
    iarray_data_type_t dtype;

    if (deserialize_meta(smeta, smeta_len, &dtype) != 0) {
        IARRAY_TRACE1(iarray.error, "Error deserializing a sframe");
        return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
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
    }

    // Build the auxshape
    (*container)->auxshape = (iarray_auxshape_t*)ina_mem_alloc(sizeof(iarray_auxshape_t));
    iarray_auxshape_t* auxshape = (*container)->auxshape;
    for (int8_t i = 0; i < catarr->ndim; ++i) {
        auxshape->index[i] = i;
        auxshape->offset[i] = 0;
        auxshape->shape_wos[i] = catarr->shape[i];
        auxshape->chunkshape_wos[i] = catarr->chunkshape[i];
        auxshape->blockshape_wos[i] = catarr->blockshape[i];
    }

    (*container)->storage = ina_mem_alloc(sizeof(iarray_storage_t));
    if ((*container)->storage == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the store parameter");
        return INA_ERROR(INA_ERR_FAILED);
    }
    (*container)->storage->filename = filename;
    (*container)->storage->backend = IARRAY_STORAGE_BLOSC;
    (*container)->storage->enforce_frame = enforce_frame;
    for (int i = 0; i < catarr->ndim; ++i) {
        (*container)->storage->chunkshape[i] = catarr->chunkshape[i];
        (*container)->storage->blockshape[i] = catarr->blockshape[i];
    }

    (*container)->view = false;
    (*container)->transposed = false;

    free(smeta);

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_get_slice(iarray_context_t *ctx,
                                   iarray_container_t *src,
                                   int64_t *start,
                                   int64_t *stop,
                                   bool view,
                                   iarray_storage_t *storage,
                                   int flags,
                                   iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(src);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(container);

    if (src->view) {
        IARRAY_TRACE1(iarray.error, "Slicing a view into another is not supported");
        return INA_ERROR(INA_ERR_NOT_SUPPORTED);
    }

    int8_t ndim = src->dtshape->ndim;
    int64_t *offset = src->auxshape->offset;
    int8_t *index = src->auxshape->index;

    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t stop_[IARRAY_DIMENSION_MAX];

    for (int i = 0; i < src->catarr->ndim; ++i) {
        start_[i] = 0 + offset[i];
        stop_[i] = 1 + offset[i];
    }

    for (int i = 0; i < ndim; ++i) {
        if (start[i] < 0) {
            start_[index[i]] += start[i] + src->dtshape->shape[i];
        } else{
            start_[index[i]] += (int64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[index[i]] += stop[i] + src->dtshape->shape[i] - 1;
        } else {
            stop_[index[i]] += (int64_t) stop[i] - 1;
        }
    }

    if (src->transposed) {
        int64_t start_trans[IARRAY_DIMENSION_MAX];
        int64_t stop_trans[IARRAY_DIMENSION_MAX];
        int64_t chunkshape_trans[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < ndim; ++i) {
            start_trans[i] = start_[ndim - 1 - i];
            stop_trans[i] = stop_[ndim - 1 - i];
        }
        for (int i = 0; i < ndim; ++i) {
            start_[i] = start_trans[i];
            stop_[i] = stop_trans[i];
        }
    }

    for (int i = 0; i < src->dtshape->ndim; ++i) {
        if (start_[i] > stop_[i]) {
            IARRAY_TRACE1(iarray.error, "Start is bigger than stop");
            return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
        }
        if (!view) {
            if (storage->backend == IARRAY_STORAGE_BLOSC && storage->chunkshape[i] > stop_[i] - start_[i]) {
                IARRAY_TRACE1(iarray.error, "The chunkshape is bigger than shape");
                return INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE);
            }
        }
    }

    if (view) {

        iarray_dtshape_t dtshape;
        dtshape.ndim = src->dtshape->ndim;
        dtshape.dtype = src->dtshape->dtype;

        for (int i = 0; i < dtshape.ndim; ++i) {
            dtshape.shape[i] = stop_[i] - start_[i];
        }

        IARRAY_RETURN_IF_FAILED(_iarray_view_new(ctx, src, &dtshape, start_, container));

        (*container)->view = true;

    } else {
        iarray_dtshape_t dtshape;

        dtshape.ndim = src->dtshape->ndim;
        dtshape.dtype = src->dtshape->dtype;

        for (int i = 0; i < dtshape.ndim; ++i) {
            dtshape.shape[i] = stop_[i] - start_[i];
        }
        IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, &dtshape, storage, flags, container));

        caterva_config_t cfg = {0};
        IARRAY_RETURN_IF_FAILED(
                iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg));
        caterva_context_t *cat_ctx;
        IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

        caterva_storage_t cat_storage = {0};
        IARRAY_RETURN_IF_FAILED(iarray_create_caterva_storage(&dtshape, storage, &cat_storage));
        if ((*container)->catarr->storage == CATERVA_STORAGE_BLOSC) {
            cat_storage.properties.blosc.nmetalayers = 1;
            cat_storage.properties.blosc.metalayers[0].name = "iarray";
            uint8_t *smeta;
            int32_t smeta_len = serialize_meta((*container)->dtshape->dtype, &smeta);
            cat_storage.properties.blosc.metalayers[0].sdata = smeta;
            cat_storage.properties.blosc.metalayers[0].size = smeta_len;
        }
        IARRAY_ERR_CATERVA(caterva_array_free(cat_ctx, &(*container)->catarr));

        IARRAY_ERR_CATERVA(caterva_array_get_slice(cat_ctx, src->catarr, start_, stop_, &cat_storage, &(*container)->catarr));

        IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));
    }

    return INA_SUCCESS;
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

    uint8_t *buffer = NULL;

    if (container->dtshape->dtype != slice->dtshape->dtype) {
        IARRAY_TRACE1(iarray.error, "The data types are different");
        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }
    if (container->dtshape->ndim != slice->dtshape->ndim) {
        IARRAY_TRACE1(iarray.error, "The dimensions are different");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }

    int typesize = slice->catarr->itemsize;
    int64_t buflen = slice->catarr->nitems;

    if (slice->catarr->storage == CATERVA_STORAGE_BLOSC) {
        buffer = ina_mem_alloc(buflen * typesize);
        IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, slice, buffer, buflen * typesize));
    } else {
        buffer = slice->catarr->buf;
    }

    IARRAY_RETURN_IF_FAILED(iarray_set_slice_buffer(ctx, container, start, stop, buffer, buflen * typesize));

    if (slice->catarr->storage == CATERVA_STORAGE_BLOSC) {
        INA_MEM_FREE_SAFE(buffer);
    }

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_get_slice_buffer(iarray_context_t *ctx,
                                          iarray_container_t *container,
                                          const int64_t *start,
                                          const int64_t *stop,
                                          void *buffer,
                                          int64_t buflen)
{
    int64_t size = 1;
    for (int i = 0; i < container->dtshape->ndim; ++i) {
        size *= container->dtshape->shape[i];
    }
    if (size == 0) {
        return INA_SUCCESS;
    }

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(buffer);

    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t stop_[IARRAY_DIMENSION_MAX];
    int64_t shape_[IARRAY_DIMENSION_MAX];

    for (int i = 0; i < container->dtshape->ndim; ++i) {
        if (start[i] < 0) {
            start_[i] = start[i] + container->dtshape->shape[i];
        } else{
            start_[i] = (int64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[i] = stop[i] + container->dtshape->shape[i];
        } else {
            stop_[i] = (int64_t) stop[i];
        }
        shape_[i] = stop_[i] - start_[i];
    }

    IARRAY_RETURN_IF_FAILED(_iarray_get_slice_buffer(ctx, container, start, stop, shape_, buffer,
                                                     buflen));

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_set_slice_buffer(iarray_context_t *ctx,
                                          iarray_container_t *container,
                                          const int64_t *start,
                                          const int64_t *stop,
                                          void *buffer,
                                          int64_t buflen)
{
    // TODO: make use of buflen so as to avoid exceeding the buffer boundaries
    INA_UNUSED(ctx);
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(buffer);

    if (container->view) {
        IARRAY_TRACE1(iarray.error, "Can not set data in a view");
        return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
    }

    if (container->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
        IARRAY_TRACE1(iarray.error, "The container is not backed by a plainbuffer");
        return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
    }

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

    for (int i = 0; i < container->catarr->ndim; ++i) {
        if (start_[i] < 0) {
            IARRAY_TRACE1(iarray.error, "Start is negative");
            return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
        }
        if (stop_[i] < start_[i]) {
            IARRAY_TRACE1(iarray.error, "Start is larger than stop");
            return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
        }
        if (container->catarr->shape[i] < stop_[i]) {
            IARRAY_TRACE1(iarray.error, "Stop is larger than the container shape");
            return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
        }
    }

    int64_t chunksize = 1;
    for (int i = 0; i < container->catarr->ndim; ++i) {
        chunksize *= stop_[i] - start_[i];
    }

    switch (container->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if (chunksize * (int64_t)sizeof(double) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                return INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER);
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if (chunksize * (int64_t)sizeof(float) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                return INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER);
            }
            break;
        default:
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }

    caterva_config_t cfg = {0};
    IARRAY_RETURN_IF_FAILED(iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg));
    caterva_context_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

    IARRAY_ERR_CATERVA(caterva_array_set_slice_buffer(cat_ctx, buffer, buflen, start_, stop_, container->catarr));

    IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));
    return INA_SUCCESS;
}


int _caterva_get_slice_buffer_no_copy(void **dest, caterva_array_t *src, int64_t *start,
                                      int64_t *stop, int64_t *chunkshape) {
    CATERVA_UNUSED_PARAM(chunkshape);
    CATERVA_UNUSED_PARAM(stop);
    int64_t start_[CATERVA_MAX_DIM];
    // int64_t stop_[CATERVA_MAX_DIM];
    int8_t s_ndim = src->ndim;

    int64_t *shape = src->shape;
    int64_t s_shape[CATERVA_MAX_DIM];
    for (int i = 0; i < CATERVA_MAX_DIM; ++i) {
        start_[(CATERVA_MAX_DIM - s_ndim + i) % CATERVA_MAX_DIM] = i < s_ndim ? start[i] : 1;
        // stop_[(CATERVA_MAX_DIM - s_ndim + i) % CATERVA_MAX_DIM] = i < s_ndim ? stop[i] : 1;
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
                                                   iarray_container_t *container,
                                                   int64_t *start,
                                                   int64_t *stop,
                                                   void **buffer,
                                                   int64_t buflen)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(buffer);

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

    int64_t chunkshape[IARRAY_DIMENSION_MAX];
    int64_t chunksize = 1;
    for (int i = 0; i < container->catarr->ndim; ++i) {
        chunkshape[i] = stop_[i] - start_[i];
        chunksize *= chunkshape[i];
    }

    switch (container->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if (chunksize * (int64_t)sizeof(double) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                return INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER);
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if (chunksize * (int64_t)sizeof(float) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                return INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER);
            }
            break;
        default:
            IARRAY_TRACE1(iarray.error, "the data type is invalid");
            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }



    IARRAY_ERR_CATERVA(_caterva_get_slice_buffer_no_copy(buffer, container->catarr, start_, stop_, chunkshape));

    return INA_SUCCESS;
}

ina_rc_t _iarray_get_slice_buffer(iarray_context_t *ctx,
                                  iarray_container_t *container,
                                  int64_t *start,
                                  int64_t *stop,
                                  int64_t *chunkshape,
                                  void *buffer,
                                  int64_t buflen)
{
    int8_t ndim = container->dtshape->ndim;
    int64_t *offset = container->auxshape->offset;
    int8_t *index = container->auxshape->index;

    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t stop_[IARRAY_DIMENSION_MAX];
    int64_t chunkshape_[IARRAY_DIMENSION_MAX];

    for (int i = 0; i < container->catarr->ndim; ++i) {
        start_[i] = 0 + offset[i];
        stop_[i] = 1 + offset[i];
        chunkshape_[i] = 1;
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
        chunkshape_[index[i]] += chunkshape[i] - 1;
    }

    if (container->transposed) {
        int64_t start_trans[IARRAY_DIMENSION_MAX];
        int64_t stop_trans[IARRAY_DIMENSION_MAX];
        int64_t chunkshape_trans[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < ndim; ++i) {
            start_trans[i] = start_[ndim - 1 - i];
            stop_trans[i] = stop_[ndim - 1 - i];
            chunkshape_trans[i] = chunkshape_[ndim - 1 - i];
        }
        for (int i = 0; i < ndim; ++i) {
            start_[i] = start_trans[i];
            stop_[i] = stop_trans[i];
            chunkshape_[i] = chunkshape_trans[i];
        }
    }

    for (int i = 0; i < container->dtshape->ndim; ++i) {
        if (start_[i] > stop_[i]) {
            IARRAY_TRACE1(iarray.error, "Start is bigger than stop");
            return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
        }
    }

    int64_t chunksize = 1;
    for (int i = 0; i < container->catarr->ndim; ++i) {
        chunksize *= chunkshape_[i];
    }

    if (container->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        if (chunksize * (int64_t)sizeof(double) > buflen) {
            IARRAY_TRACE1(iarray.error, "The buffer size is not enough\n");
            return INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER);
        }
    } else {
        if (chunksize * (int64_t)sizeof(float) > buflen) {
            IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
            return INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER);
        }
    }

    caterva_config_t cfg = {0};
    IARRAY_RETURN_IF_FAILED(iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg));
    caterva_context_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

    if(container->transposed) {
        uint8_t *buffer_aux = malloc(chunksize * container->catarr->itemsize);
        IARRAY_ERR_CATERVA(caterva_array_get_slice_buffer(cat_ctx, container->catarr, start_,
                                                          stop_, chunkshape_, buffer_aux,
                                                          chunksize * container->catarr->itemsize));
        char ordering = 'R';
        char trans = 'T';
        int rows = chunkshape_[0];
        int cols = chunkshape_[1];
        uint8_t *src = buffer_aux;
        int src_ld = cols;
        uint8_t *dst = buffer;
        int dst_ld = rows;

        switch (container->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_domatcopy(ordering, trans, rows, cols, 1., (double *) src, src_ld,
                              (double *) dst, dst_ld);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_somatcopy(ordering, trans, rows, cols, 1.f, (float *) src, src_ld,
                              (float *) dst, dst_ld);
                break;
            default:
                return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
        }
        free(buffer_aux);
    } else {
        IARRAY_ERR_CATERVA(caterva_array_get_slice_buffer(cat_ctx, container->catarr, start_, stop_,
                                                          chunkshape_, buffer, buflen));
    }
    IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_squeeze_index(iarray_context_t *ctx,
                                       iarray_container_t *container,
                                       bool *index)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);


    if (!container->view) {
        caterva_config_t cfg = {0};
        iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
        caterva_context_t *cat_ctx;
        IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

        IARRAY_ERR_CATERVA(caterva_array_squeeze_index(cat_ctx, container->catarr, index));
        IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));

        uint8_t inc = 0;
        if (container->dtshape->ndim != container->catarr->ndim) {
            container->dtshape->ndim = (uint8_t) container->catarr->ndim;
            for (int i = 0; i < container->catarr->ndim; ++i) {
                if (index[i]) {
                    inc ++;
                }
                container->dtshape->shape[i] = container->catarr->shape[i];
                container->storage->chunkshape[i] = container->catarr->chunkshape[i];
                container->storage->blockshape[i] = container->catarr->blockshape[i];
                container->auxshape->shape_wos[i] = container->catarr->shape[i];
                container->auxshape->chunkshape_wos[i] = container->catarr->chunkshape[i];
                container->auxshape->blockshape_wos[i] = container->catarr->blockshape[i];
                container->auxshape->offset[i] = container->auxshape->offset[i + inc];
            }
        }
    } else {
        uint8_t inc = 0;
        for (int i = 0; i < container->dtshape->ndim; ++i) {
            if (index[i]) {
                inc ++;
            } else {
                container->dtshape->shape[i - inc] = container->dtshape->shape[i];
                container->storage->chunkshape[i - inc] = container->storage->chunkshape[i];
                container->storage->blockshape[i - inc] = container->storage->blockshape[i];
                container->auxshape->index[i - inc] = (uint8_t) i;
            }
        }
        container->dtshape->ndim -= inc;
    }

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_squeeze(iarray_context_t *ctx,
                                 iarray_container_t *container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);

    bool index[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < container->dtshape->ndim; ++i) {
        if (container->dtshape->shape[i] == 1) {
            index[i] = true;
        }
    }

    IARRAY_RETURN_IF_FAILED(iarray_squeeze_index(ctx, container, index));

    return INA_SUCCESS;
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
    }
    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_get_storage(iarray_context_t *ctx,
                                     iarray_container_t *c,
                                     iarray_storage_t *storage)
{
    INA_UNUSED(ctx);
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(storage);
    ina_mem_cpy(storage, c->storage, sizeof(iarray_storage_t));
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_is_view(iarray_context_t *ctx,
                                 iarray_container_t *c,
                                 bool *view)
{
    INA_UNUSED(ctx);
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(view);

    if (c->view) {
        *view = true;
    } else {
        *view = false;
    }
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_container_info(iarray_container_t *container, int64_t *nbytes, int64_t *cbytes)
{
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(nbytes);
    INA_VERIFY_NOT_NULL(cbytes);

    if (container->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        *nbytes = container->catarr->nitems * container->catarr->itemsize;
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
    if (a->dtshape->dtype != b->dtshape->dtype){
        IARRAY_TRACE1(iarray.error, "The data types are not equals");
        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }
    if (a->dtshape->ndim != b->dtshape->ndim) {
        IARRAY_TRACE1(iarray.error, "The dimensions are not equals");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        if (a->dtshape->shape[i] != b->dtshape->shape[i]) {
            IARRAY_TRACE1(iarray.error, "The shapes are not equals");
            return INA_ERROR(IARRAY_ERR_INVALID_SHAPE);
        }
    }

    int dtype = a->dtshape->dtype;
    int ndim = a->dtshape->ndim;

    // For the blocksize, choose the maximum of the partition shapes
    int64_t blocksize[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ndim; ++i) {
        blocksize[i] = INA_MAX(a->storage->chunkshape[i], b->storage->chunkshape[i]);
    }

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_t *ctx = NULL;
    IARRAY_RETURN_IF_FAILED(iarray_context_new(&cfg, &ctx));
    iarray_iter_read_block_t *iter_a;
    iarray_iter_read_block_value_t val_a;
    IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_new(ctx, &iter_a, a, blocksize, &val_a, false));
    iarray_iter_read_block_t *iter_b;
    iarray_iter_read_block_value_t val_b;
    IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_new(ctx, &iter_b, b, blocksize, &val_b, false));

    while (INA_SUCCEED(iarray_iter_read_block_has_next(iter_a))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_next(iter_a, NULL, 0));
        IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_next(iter_b, NULL, 0));

        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (int64_t i = 0; i < val_a.block_size; ++i) {
                double adiff = fabs(((double *)val_a.block_pointer)[i] - ((double *)val_b.block_pointer)[i]);
                double rdiff = fabs(((double *)val_a.block_pointer)[i] - ((double *)val_b.block_pointer)[i]) /
                    ((double *)val_a.block_pointer)[i];
                if (rdiff > tol) {
                    printf("%f, %f (adiff: %f, rdiff: %f)\n", ((double *)val_a.block_pointer)[i],
                        ((double *)val_b.block_pointer)[i], adiff, rdiff);
                    IARRAY_TRACE1(iarray.error, "Values are different");
                    return INA_ERROR(IARRAY_ERR_ASSERTION_FAILED);
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
                    return INA_ERROR(IARRAY_ERR_ASSERTION_FAILED);
                }
            }
        }
    }

    IARRAY_ITER_FINISH();
    iarray_iter_read_block_free(&iter_a);
    iarray_iter_read_block_free(&iter_b);

    iarray_context_free(&ctx);

    return INA_SUCCESS;
}


INA_API(void) iarray_container_free(iarray_context_t *ctx, iarray_container_t **container)
{
    INA_UNUSED(ctx);
    INA_VERIFY_FREE(container);

    if (!(*container)->view) {
        if ((*container)->catarr != NULL) {
            // It can happen in some automatic garbage collection environments (e.g. Python)
            // that context objects are collected prior to containers.  These situations
            // typically happen during exceptions, so even if we are leaving leaks, there
            // is little we can do.
            if (ctx == NULL) return;
            caterva_config_t cfg = {0};
            iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
            caterva_context_t *cat_ctx;
            caterva_context_new(&cfg, &cat_ctx);
            caterva_array_free(cat_ctx, &(*container)->catarr);
            caterva_context_free(&cat_ctx);
        }
        INA_MEM_FREE_SAFE((*container)->storage);
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
