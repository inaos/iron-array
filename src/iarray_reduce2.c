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
#include "iarray_reduce_operations.h"
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
    void (*ufunc)(void*, int64_t, void*);
    iarray_container_t *input;
    iarray_container_t *result;
    int8_t axis;
    int64_t *chunk_shape;
    uint8_t *chunk;
    int64_t chunk_index;
    uint8_t *chunk_out;
} iarray_reduce_params_t;


static int _reduce_prefilter(blosc2_prefilter_params *pparams) {
    memset(pparams->out, 0, pparams->out_size);
    iarray_reduce_params_t *rparams = (iarray_reduce_params_t *) pparams->user_data;

    // Compute result block offset
    int64_t block_offset_u = pparams->out_offset / pparams->out_size;
    int64_t block_offset_n[IARRAY_DIMENSION_MAX] = {0};

    int64_t shape_of_blocks[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
        shape_of_blocks[i] = rparams->result->catarr->extchunkshape[i] /
                rparams->result->catarr->blockshape[i];
    }
    index_unidim_to_multidim(rparams->result->catarr->ndim,
                             shape_of_blocks,
                             block_offset_u,
                             block_offset_n);


    // Compute the input strides
    int64_t strides[IARRAY_DIMENSION_MAX] = {0};
    strides[rparams->input->dtshape->ndim - 1] = 1;
    for (int i = rparams->input->dtshape->ndim - 2; i >= 0 ; --i) {
        strides[i] = rparams->input->storage->blockshape[i + 1] * strides[i + 1];
    }

    int64_t nblocks = rparams->input->catarr->extchunkshape[rparams->axis] /
                      rparams->input->catarr->blockshape[rparams->axis];

    int64_t block_strides[IARRAY_DIMENSION_MAX];
    block_strides[rparams->input->dtshape->ndim - 1] = 1;
    for (int i = rparams->input->dtshape->ndim - 2; i >= 0 ; --i) {
        int64_t nblocks_ = rparams->input->catarr->extchunkshape[i + 1] /
                           rparams->input->catarr->blockshape[i + 1];
        block_strides[i] = nblocks_ * block_strides[i + 1];
    }

    // Allocate destination
    uint8_t *block = malloc(rparams->input->catarr->blocknitems * rparams->input->catarr->itemsize);



        for (int block_ind = 0; block_ind < nblocks; ++block_ind) {
            int64_t nblock = block_ind * block_strides[rparams->axis];
            for (int j = 0; j < rparams->result->catarr->ndim; ++j) {
                if (j < rparams->axis)
                    nblock += block_offset_n[j] * block_strides[j];
                else
                    nblock += block_offset_n[j] * block_strides[j + 1];
            }
            int64_t start = nblock * rparams->input->catarr->blocknitems;

            blosc_getitem(rparams->chunk, start, rparams->input->catarr->blocknitems, block);

            // Check if there are padding in reduction axis
            int64_t aux = (block_ind + 1) * rparams->result->catarr->blockshape[rparams->axis];
            aux += rparams->chunk_index * rparams->input->catarr->chunkshape[rparams->axis];

            int64_t vector_nelems;
            if (aux > rparams->input->catarr->shape[rparams->axis]) {
                vector_nelems = rparams->input->catarr->shape[rparams->axis] - aux;
            } else {
                vector_nelems = rparams->input->catarr->blockshape[rparams->axis];
            }

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
                    elem_index_n2[i] = elem_index_n[i] + block_offset_n[i] *
                            rparams->result->catarr->blockshape[i];
                }
                for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
                    if (rparams->chunk_shape[i] <= elem_index_n2[i]) {
                        empty = true;
                        break;
                    }
                }
                if (empty) {
                    switch (rparams->result->dtshape->dtype) {
                        case IARRAY_DATA_TYPE_DOUBLE:
                            ((double *) pparams->out)[ind] = 0;
                            break;
                        case IARRAY_DATA_TYPE_FLOAT:
                            ((float *) pparams->out)[ind] = 0;
                            break;
                        default:
                            IARRAY_TRACE1(iarray.error, "Invalid dtype");
                            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
                    }
                    continue;
                }

                // Compute index in slice
                for (int i = rparams->input->dtshape->ndim - 1; i >= 0; --i) {
                    if (i > rparams->axis) {
                        elem_index_n[i] = elem_index_n[i - 1];
                    } else if (i == rparams->axis) {
                        elem_index_n[i] = 0;
                    } else {
                        elem_index_n[i] = elem_index_n[i];
                    }
                }

                int64_t elem_index_u = 0;
                for (int i = 0; i < rparams->input->dtshape->ndim; ++i) {
                    elem_index_u += elem_index_n[i] * strides[i];
                }

                for (int i = 0; i < vector_nelems; ++i) {
                    switch (rparams->result->dtshape->dtype) {
                        case IARRAY_DATA_TYPE_DOUBLE:
                            ((double *) pparams->out)[ind] += ((double *) block)[elem_index_u +
                                                                                 strides[rparams->axis] * i];
                            break;
                        case IARRAY_DATA_TYPE_FLOAT:
                            ((float *) pparams->out)[ind] += ((float *) block)[elem_index_u +
                                                                                 strides[rparams->axis] * i];
                            break;
                        default:
                            IARRAY_TRACE1(iarray.error, "Invalid dtype");
                            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
                    }
                }
            }
        }

    free(block);

    return 0;
}


static ina_rc_t _iarray_reduce_udf(iarray_context_t *ctx,
                                   iarray_container_t *a,
                                   void (*ufunc)(void*, int64_t, void*),
                                   int8_t axis,
                                   iarray_container_t **b) {

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(ufunc);
    INA_VERIFY_NOT_NULL(b);

    if (a->storage->backend == IARRAY_STORAGE_PLAINBUFFER) {
        IARRAY_TRACE1(iarray.error, "Reduction can not be performed over a plainbuffer "
                                    "container");
        return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
    }
    if (a->dtshape->ndim < 2) {
        IARRAY_TRACE1(iarray.error, "The container dimensions must be greater than 1");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }

    iarray_dtshape_t dtshape;
    dtshape.dtype = a->dtshape->dtype;
    dtshape.ndim = a->dtshape->ndim - 1;
    for (int i = 0; i < dtshape.ndim; ++i) {
        dtshape.shape[i] = i < axis ? a->dtshape->shape[i] : a->dtshape->shape[i + 1];
    }

    iarray_storage_t storage_red;
    storage_red.backend = IARRAY_STORAGE_BLOSC;
    storage_red.enforce_frame = false;
    storage_red.filename = NULL;
    for (int i = 0; i < dtshape.ndim; ++i) {
        if (i < axis) {
            storage_red.chunkshape[i] = a->storage->chunkshape[i];
            storage_red.blockshape[i] = a->storage->blockshape[i];
        } else {
            storage_red.chunkshape[i] = a->storage->chunkshape[i + 1];
            storage_red.blockshape[i] = a->storage->blockshape[i + 1];
        }
    }
    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, &dtshape, &storage_red, 0, b));

    iarray_container_t *c = *b;

    // Set up prefilter
    iarray_context_t *prefilter_ctx = ina_mem_alloc(sizeof(iarray_context_t));
    memcpy(prefilter_ctx, ctx, sizeof(iarray_context_t));
    prefilter_ctx->prefilter_fn = (blosc2_prefilter_fn) _reduce_prefilter;
    iarray_reduce_params_t reduce_params = {0};
    blosc2_prefilter_params pparams = {0};
    pparams.user_data = &reduce_params;
    prefilter_ctx->prefilter_params = &pparams;



    // Fill prefilter params
    reduce_params.input = a;
    reduce_params.result = c;
    reduce_params.axis = axis;
    reduce_params.ufunc = ufunc;

    // Compute the amount of chunks in each dimension
    int64_t shape_of_chunks[IARRAY_DIMENSION_MAX]={0};
    for (int i = 0; i < c->dtshape->ndim; ++i) {
        shape_of_chunks[i] = c->catarr->extshape[i] / c->catarr->chunkshape[i];
    }

    // Iterate over chunks
    int64_t chunk_index[IARRAY_DIMENSION_MAX] = {0};
    int64_t nchunk = 0;

    uint8_t *chunk_out = malloc(c->catarr->extchunknitems * c->catarr->itemsize +
                                BLOSC_MAX_OVERHEAD);
    uint8_t *chunk_part = malloc(c->catarr->extchunknitems * c->catarr->itemsize +
                                 BLOSC_MAX_OVERHEAD);
    uint8_t *chunk_tmp = malloc(c->catarr->extchunknitems * c->catarr->itemsize +
                                BLOSC_MAX_OVERHEAD);
    while (nchunk < c->catarr->extnitems / c->catarr->chunknitems) {
        memset(chunk_out, 0, c->catarr->extchunknitems * c->catarr->itemsize +
                             BLOSC_MAX_OVERHEAD);
        // Compute result chunk offset
        int64_t chunk_offset_u = c->catarr->sc->nchunks;
        int64_t chunk_offset_n[IARRAY_DIMENSION_MAX] = {0};

        index_unidim_to_multidim(c->catarr->ndim,
                                 shape_of_chunks,
                                 chunk_offset_u,
                                 chunk_offset_n);

        // Compute first chunk element and the chunk shape
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

        int64_t nchunks = a->catarr->extshape[axis] / a->catarr->chunkshape[axis];

        int64_t chunk_strides[IARRAY_DIMENSION_MAX];
        chunk_strides[a->dtshape->ndim - 1] = 1;
        for (int i = a->dtshape->ndim - 2; i >= 0 ; --i) {
            int64_t nchunks_ = a->catarr->extshape[i + 1] /
                               a->catarr->chunkshape[i + 1];
            chunk_strides[i] = nchunks_ * chunk_strides[i + 1];
        }

        for (int chunk_ind = 0; chunk_ind < nchunks; ++chunk_ind) {
            int64_t nchunk_axis = chunk_ind * chunk_strides[axis];
            for (int j = 0; j < a->catarr->ndim; ++j) {
                if (j < axis)
                    nchunk_axis += chunk_offset_n[j] * chunk_strides[j];
                else
                    nchunk_axis += chunk_offset_n[j] * chunk_strides[j + 1];
            }
            uint8_t *chunk_axis;
            bool needs_free;
            blosc2_schunk_get_chunk(a->catarr->sc, nchunk_axis, &chunk_axis, &needs_free);

            reduce_params.chunk = chunk_axis;
            reduce_params.chunk_index = chunk_ind;

            // Compress data
            blosc2_cparams cparams = {0};
            IARRAY_RETURN_IF_FAILED(
                    iarray_create_blosc_cparams(&cparams, prefilter_ctx, c->catarr->itemsize,
                                                c->catarr->blocknitems * c->catarr->itemsize));
            blosc2_context *cctx = blosc2_create_cctx(cparams);

            int csize = blosc2_compress_ctx(cctx, NULL,
                                            c->catarr->extchunknitems * c->catarr->itemsize,
                                            chunk_tmp,
                                            c->catarr->extchunknitems * c->catarr->itemsize +
                                            BLOSC_MAX_OVERHEAD);
            if (csize <= 0) {
                IARRAY_TRACE1(iarray.error, "Error compressing a blosc chunk");
                return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
            }

            blosc2_dparams dparams = {.nthreads = ctx->cfg->max_num_threads};
            blosc2_context *dctx = blosc2_create_dctx(dparams);

            blosc2_decompress_ctx(dctx, chunk_tmp,
                                  c->catarr->extchunknitems * c->catarr->itemsize +
                                  BLOSC_MAX_OVERHEAD,
                                  chunk_part,
                                  c->catarr->extchunknitems * c->catarr->itemsize +
                                  BLOSC_MAX_OVERHEAD);
            for (int i = 0; i < c->catarr->extchunknitems; ++i) {
                ((double *) chunk_out)[i] += ((double *) chunk_part)[i];
            }
            blosc2_free_ctx(cctx);
        }

        blosc2_schunk_append_buffer(c->catarr->sc, chunk_out, c->catarr->extchunknitems * c->catarr->itemsize);

        nchunk++;
        index_unidim_to_multidim(c->dtshape->ndim, shape_of_chunks, nchunk, chunk_index);
    }
    free(chunk_part);
    free(chunk_out);
    free(chunk_tmp);
    c->catarr->empty = false;
    c->catarr->filled = true;

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_reduce2(iarray_context_t *ctx,
                                iarray_container_t *a,
                                iarray_reduce_func_t func,
                                int8_t axis,
                                iarray_container_t **b) {
    void *reduce_funtion = NULL;

    switch (func) {
        case IARRAY_REDUCE_MAX:
            reduce_funtion = a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ?
                             (void (*)(void *, int64_t, void *)) dmax :
                             (void (*)(void *, int64_t, void *)) smax;
            break;
        case IARRAY_REDUCE_MIN:
            reduce_funtion = a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ?
                             (void (*)(void *, int64_t, void *)) dmin :
                             (void (*)(void *, int64_t, void *)) smin;
            break;
        case IARRAY_REDUCE_SUM:
            reduce_funtion = a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ?
                             (void (*)(void *, int64_t, void *)) dsum :
                             (void (*)(void *, int64_t, void *)) ssum;
            break;
        case IARRAY_REDUCE_PROD:
            reduce_funtion = a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ?
                             (void (*)(void *, int64_t, void *)) dprod :
                             (void (*)(void *, int64_t, void *)) sprod;
            break;
        case IARRAY_REDUCE_MEAN:
            reduce_funtion = a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ?
                             (void (*)(void *, int64_t, void *)) dmean :
                             (void (*)(void *, int64_t, void *)) smean;
            break;
        case IARRAY_REDUCE_STD:
            reduce_funtion = a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ?
                             (void (*)(void *, int64_t, void *)) dstd :
                             (void (*)(void *, int64_t, void *)) sstd;
            break;
    }
    IARRAY_RETURN_IF_FAILED(_iarray_reduce_udf(ctx, a, reduce_funtion, axis, b));

    return INA_SUCCESS;
}
