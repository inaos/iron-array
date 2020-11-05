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


static void index_unidim_to_multidim(int8_t ndim, int64_t *shape, int64_t i, int64_t *index) {
    int64_t strides[CATERVA_MAX_DIM];
    strides[ndim - 1] = 1;
    for (int j = ndim - 2; j >= 0; --j) {
        strides[j] = shape[j + 1] * strides[j + 1];
    }

    index[0] = i / strides[0];
    for (int j = 1; j < ndim; ++j) {
        index[j] = (i % strides[j - 1]) / strides[j];
    }
}


typedef struct iarray_reduce_params_s {
    void (*ufunc)(uint8_t*, int64_t, uint8_t*);
    iarray_container_t *input;
    iarray_container_t *result;
    uint8_t *data;
    int64_t *data_shape;
    int8_t axis;
    int64_t *chunk_shape;
} iarray_reduce_params_t;


static int _reduce_prefilter(blosc2_prefilter_params *pparams) {
    iarray_reduce_params_t *rparams = (iarray_reduce_params_t *) pparams->user_data;

    // Compute offset
    int64_t offset_u = pparams->out_offset / pparams->out_typesize;
    int64_t offset_n[IARRAY_DIMENSION_MAX] = {0};

    int64_t shape_of_blocks[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
        shape_of_blocks[i] = rparams->result->catarr->extchunkshape[i] / rparams->result->catarr
                ->blockshape[i];
    }
    int64_t offset_u_blocks = offset_u / (pparams->out_size / pparams->out_typesize);
    index_unidim_to_multidim(rparams->result->catarr->ndim,
                             shape_of_blocks,
                             offset_u_blocks,
                             offset_n);
    for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
        offset_n[i] *= rparams->result->catarr->blockshape[i];
    }
    // Compute the strides
    int64_t strides[IARRAY_DIMENSION_MAX] = {0};
    strides[rparams->input->dtshape->ndim - 1] = 1;
    for (int i = rparams->input->dtshape->ndim - 2; i >= 0 ; --i) {
        strides[i] = rparams->data_shape[i + 1] * strides[i + 1];
    }

    // Alloc dest
    uint8_t *vector = malloc(rparams->data_shape[rparams->axis] * pparams->out_typesize);

    for (int64_t ind = 0; ind < pparams->out_size / pparams->out_typesize; ++ind) {

        // Compute index in dest
        int64_t elem_index_n[IARRAY_DIMENSION_MAX] = {0};
        index_unidim_to_multidim(rparams->result->catarr->ndim,
                                 rparams->result->storage->blockshape,
                                 ind,
                                 elem_index_n);

        bool empty = false;
        int64_t elem_index_n2[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
            elem_index_n2[i] = elem_index_n[i] + offset_n[i];
        }
        for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
            if (rparams->chunk_shape[i] <= elem_index_n[i] + offset_n[i]) {
                empty = true;
                break;
            }
        }
        if (empty) {
            ((double *) pparams->out)[ind] = 0;
            continue;
        }

        // Compute index in slice
        for (int i = rparams->input->dtshape->ndim - 1; i >= 0; --i) {
            if (i > rparams->axis) {
                elem_index_n[i] = elem_index_n[i - 1] + offset_n[i - 1];
            } else if (i == rparams->axis) {
                elem_index_n[i] = 0;
            } else {
                elem_index_n[i] = elem_index_n[i] + offset_n[i];
            }
        }

        int64_t elem_index_u = 0;
        for (int i = 0; i < rparams->input->dtshape->ndim; ++i) {
            elem_index_u += elem_index_n[i] * strides[i];
        }

        vdPackI(rparams->data_shape[rparams->axis],
         &((double *) rparams->data)[elem_index_u],
         strides[rparams->axis], (double *) vector);

        double red;
        rparams->ufunc(vector, rparams->data_shape[rparams->axis], &red);
        ((double *) pparams->out)[ind] = red;
    }

    free(vector);

    return 0;
}


INA_API(ina_rc_t) iarray_reduce_double(iarray_context_t *ctx,
                                       iarray_container_t *a,
                                       void (*ufunc)(uint8_t*, int64_t, uint8_t*),
                                       int8_t axis,
                                       iarray_storage_t *storage,
                                       iarray_container_t **b) {

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(ufunc);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(b);

    if (a->dtshape->ndim < 2) {
        IARRAY_TRACE1(iarray.error, "The container dimensions must be greater than 1");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }

    if (a->dtshape->dtype != IARRAY_DATA_TYPE_DOUBLE) {
        IARRAY_TRACE1(iarray.error, "Reductions are only supported for double data");
        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }
    iarray_dtshape_t dtshape;
    dtshape.dtype = a->dtshape->dtype;
    dtshape.ndim = a->dtshape->ndim - 1;
    for (int i = 0; i < dtshape.ndim; ++i) {
        dtshape.shape[i] = i < axis ? a->dtshape->shape[i] : a->dtshape->shape[i + 1];
    }

    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, &dtshape, storage, 0, b));

    iarray_container_t *c = *b;

    // Set up prefilter
    iarray_context_t *prefilter_ctx = ina_mem_alloc(sizeof(iarray_context_t));
    memcpy(prefilter_ctx, ctx, sizeof(iarray_context_t));
    prefilter_ctx->prefilter_fn = (blosc2_prefilter_fn) _reduce_prefilter;
    iarray_reduce_params_t reduce_params = {0};
    blosc2_prefilter_params pparams = {0};
    pparams.user_data = &reduce_params;
    prefilter_ctx->prefilter_params = &pparams;

    // Alloc temporal
    int64_t shape[IARRAY_DIMENSION_MAX];
    int64_t cache_size = a->catarr->itemsize;
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        if (i < axis) {
            shape[i] = c->catarr->chunkshape[i];
        } else if (i == axis) {
            shape[i] = a->dtshape->shape[i];
        } else {
            shape[i] = c->catarr->chunkshape[i - 1];
        }
        cache_size *= shape[i];
    }
    uint8_t *cache = malloc(cache_size);

    // Fill prefilter params
    reduce_params.input = a;
    reduce_params.result = c;
    reduce_params.data = cache;
    reduce_params.data_shape = shape;
    reduce_params.axis = axis;
    reduce_params.ufunc = ufunc;

    // Compute the amount of chunks that there are in each dimension
    int64_t shape_of_chunks[IARRAY_DIMENSION_MAX]={0};
    for (int i = 0; i < c->dtshape->ndim; ++i) {
        shape_of_chunks[i] = c->catarr->extshape[i] / c->catarr->chunkshape[i];
    }

    // Iterate over chunks
    int64_t chunk_index[IARRAY_DIMENSION_MAX] = {0};
    int64_t nchunk = 0;
    while (nchunk < c->catarr->extnitems / c->catarr->chunknitems) {

        // Conmpute first chunk element and the chunk shape
        int64_t elem_index[IARRAY_DIMENSION_MAX] = {0};
        for (int i = 0; i < c->dtshape->ndim; ++i) {
            elem_index[i] = chunk_index[i] * c->catarr->chunkshape[i];
        }
        int64_t chunk_shape[IARRAY_DIMENSION_MAX] = {0};
        for (int i = 0; i < c->dtshape->ndim; ++i) {
            if (elem_index[i] + c->catarr->chunkshape[i] <= c->catarr->shape[i]) {
                chunk_shape[i] = c->catarr->chunkshape[i];
            } else {
                chunk_shape[i] = c->catarr->shape[i] - elem_index[i];
            }
        }

        reduce_params.chunk_shape = chunk_shape;

        // Compute the start and the stop of the slice
        int64_t start[IARRAY_DIMENSION_MAX];
        int64_t stop[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < a->dtshape->ndim; ++i) {
            if (i < axis) {
                start[i] = elem_index[i];
                stop[i] = start[i] + chunk_shape[i];
            } else if (i == axis) {
                start[i] = 0;
                stop[i] = start[i] + a->dtshape->shape[i];
            } else {
                start[i] = elem_index[i - 1];
                stop[i] = start[i] + chunk_shape[i - 1];
            }
        }

        // Get the slice into cache
        _iarray_get_slice_buffer(ctx, a, start, stop, shape, cache, cache_size);

        // Compress data
        blosc2_cparams cparams = {0};
        IARRAY_RETURN_IF_FAILED(iarray_create_blosc_cparams(&cparams, prefilter_ctx, c->catarr->itemsize,
                                                            c->catarr->blocknitems * c->catarr->itemsize));
        blosc2_context *cctx = blosc2_create_cctx(cparams);
        uint8_t *chunk = malloc(c->catarr->extchunknitems * c->catarr->itemsize +
                                BLOSC_MAX_OVERHEAD);
        int csize = blosc2_compress_ctx(cctx, NULL, c->catarr->extchunknitems * c->catarr->itemsize,
                                        chunk,
                                        c->catarr->extchunknitems * c->catarr->itemsize +
                                        BLOSC_MAX_OVERHEAD);
        if (csize <= 0) {
            IARRAY_TRACE1(iarray.error, "Error compressing a blosc chunk");
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }
        blosc2_free_ctx(cctx);

        // Append to schunk
        blosc2_schunk_append_chunk(c->catarr->sc, chunk, false);

        nchunk++;
        index_unidim_to_multidim(c->dtshape->ndim, shape_of_chunks, nchunk, chunk_index);
    }
    c->catarr->empty = false;
    c->catarr->filled = true;

    return INA_SUCCESS;
}
