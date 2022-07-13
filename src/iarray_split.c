/*
 * Copyright ironArray SL 2022.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include "libiarray/iarray.h"
# include "iarray_private.h"

ina_rc_t iarray_compute_chunk_shape(iarray_container_t *a,
                                    int64_t *chunk_in_shape_strides,
                                    int64_t nchunk,
                                    iarray_dtshape_t *dtshape) {
    int64_t n_chunk_n[IARRAY_DIMENSION_MAX];
    iarray_index_unidim_to_multidim(a->catarr->ndim, chunk_in_shape_strides, nchunk, n_chunk_n);

    for (int i = 0; i < a->catarr->ndim; ++i) {
        if ((n_chunk_n[i] + 1) * a->catarr->chunkshape[i] > a->catarr->shape[i]) {
            dtshape->shape[i] = a->catarr->shape[i] - n_chunk_n[i] * a->catarr->chunkshape[i];
        } else {
            dtshape->shape[i] = a->catarr->chunkshape[i];
        }
    }

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_split_new(iarray_context_t *ctx,
                                   iarray_container_t *a,
                                   iarray_split_container_t **b) {
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(b);

    *b = ina_mem_alloc(sizeof(iarray_split_container_t));
    (*b)->dtshape = ina_mem_alloc(sizeof(iarray_dtshape_t));

    memcpy((*b)->dtshape, a->dtshape, sizeof(iarray_dtshape_t));

    for (int i = 0; i < a->catarr->ndim; ++i) {
        (*b)->n_splits_n[i] = a->catarr->extchunkshape[i] / a->catarr->blockshape[i];
    }
    (*b)->n_splits = a->catarr->nchunks;
    (*b)->splits = ina_mem_alloc((*b)->n_splits * sizeof(iarray_container_t *));

    iarray_dtshape_t split_dtshape;
    split_dtshape.dtype = a->dtshape->dtype;
    split_dtshape.dtype_size = a->dtshape->dtype_size;
    split_dtshape.ndim = a->dtshape->ndim;
    for (int i = 0; i < split_dtshape.ndim; ++i) {
        split_dtshape.shape[i] = a->storage->chunkshape[i];
    }

    iarray_storage_t split_storage;
    split_storage.urlpath = NULL;
    split_storage.contiguous = true;
    for (int i = 0; i < split_dtshape.ndim; ++i) {
        split_storage.chunkshape[i] = a->storage->chunkshape[i];
        split_storage.blockshape[i] = a->storage->blockshape[i];
    }

    int64_t chunk_in_shape_strides[IARRAY_DIMENSION_MAX] = {0};
    chunk_in_shape_strides[a->catarr->ndim - 1] = 1;
    for (int i = a->catarr->ndim - 2; i >= 0; --i) {
        chunk_in_shape_strides[i] = chunk_in_shape_strides[i + 1] * a->catarr->extshape[i + 1] / a->catarr->chunkshape[i + 1];
    }

    for (int i = 0; i < (*b)->n_splits; ++i) {
        // Create an empty split (array with one chunk)
        IARRAY_RETURN_IF_FAILED(
                iarray_compute_chunk_shape(a, chunk_in_shape_strides, i, &split_dtshape));
        IARRAY_RETURN_IF_FAILED(
                iarray_empty(ctx, &split_dtshape, &split_storage, &((*b)->splits[i])));

        // Get the desired chunk from the original array
        uint8_t *chunk;
        bool needs_free;
        int csize = blosc2_schunk_get_chunk(a->catarr->sc, i, &chunk, &needs_free);
        if (csize < 0) {
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }

        // Set the chunk to the splited array
        int64_t nchunk = blosc2_schunk_update_chunk(((*b)->splits[i])->catarr->sc, 0, chunk, true);
        if (nchunk != 1) {
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }
        if (needs_free) {
            free(chunk);
        }
    }

    return INA_SUCCESS;
}

INA_API(void) iarray_split_free(iarray_context_t *ctx,
                                iarray_split_container_t **a) {
    INA_VERIFY_FREE(a);
    for (int i = 0; i < (*a)->n_splits; ++i) {
        iarray_container_free(ctx, &((*a)->splits[i]));
    }
    INA_MEM_FREE_SAFE((*a)->dtshape);
    INA_MEM_FREE_SAFE(*a);
}

INA_API(ina_rc_t) iarray_concatenate(iarray_context_t *ctx,
                                     iarray_split_container_t *b,
                                     iarray_storage_t *storage,
                                     iarray_container_t **a) {
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(b);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(a);

    // Check chunkshape and blockshape
    for (int i = 0; i < b->dtshape->ndim; ++i) {
        if (b->n_splits != 0) {
            iarray_container_t *split = b->splits[0];
            for (int j = 0; j < b->dtshape->ndim; ++j) {
                if (split->storage->chunkshape[i] != storage->chunkshape[i]) {
                    return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
                }
                if (split->storage->blockshape[i] != storage->blockshape[i]) {
                    return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
                }
            }
        }
    }

    IARRAY_RETURN_IF_FAILED(
            iarray_empty(ctx, b->dtshape, storage, a)
            );

    for (int i = 0; i < b->n_splits; ++i) {
        iarray_container_t *split = b->splits[i];

        // Get the desired chunk from the split
        uint8_t *chunk;
        bool needs_free;
        int csize = blosc2_schunk_get_chunk(split->catarr->sc, 0, &chunk, &needs_free);
        if (csize < 0) {
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }

        // Set the chunk to the concatenated array
        int64_t nchunk = blosc2_schunk_update_chunk((*a)->catarr->sc, i, chunk, true);
        if (nchunk < 0) {
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }
        if (needs_free) {
            free(chunk);
        }

    }
    return INA_SUCCESS;
}
