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
    if (ndim == 0) {
        return;
    }
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
    iarray_reduce_function_t *ufunc;
    iarray_container_t *input;
    iarray_container_t *result;
    int8_t axis;
    int64_t *chunk_shape;
    int64_t nchunk;
} iarray_reduce_params_t;


static bool check_padding(int64_t *block_offset_n,
                          int64_t *elem_index_n,
                          iarray_reduce_params_t *rparams) {
    int64_t elem_index_n2[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
        elem_index_n2[i] = elem_index_n[i] + block_offset_n[i] *
                                             rparams->result->catarr->blockshape[i];
    }
    for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
        if (rparams->chunk_shape[i] <= elem_index_n2[i]) {
            return true;
        }
    }
    return false;
}


static int _reduce_prefilter(blosc2_prefilter_params *pparams) {
    iarray_reduce_params_t *rparams = (iarray_reduce_params_t *) pparams->user_data;
    user_data_t user_data = {0};
    user_data.inv_nelem = 1. / rparams->input->dtshape->shape[rparams->axis];

    blosc2_dparams dparams = {.nthreads = 1, .schunk = rparams->input->catarr->sc};
    blosc2_context *dctx = blosc2_create_dctx(dparams);

    // Compute result chunk offset
    int64_t chunk_offset_u = rparams->nchunk;
    int64_t chunk_offset_n[IARRAY_DIMENSION_MAX] = {0};

    int64_t shape_of_chunks[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
        shape_of_chunks[i] = rparams->result->catarr->extshape[i] /
                rparams->result->catarr->chunkshape[i];
    }
    index_unidim_to_multidim(rparams->result->catarr->ndim,
                             shape_of_chunks,
                             chunk_offset_u,
                             chunk_offset_n);

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

    int64_t nchunks = rparams->input->catarr->extshape[rparams->axis] /
                      rparams->input->catarr->chunkshape[rparams->axis];

    int64_t chunk_strides[IARRAY_DIMENSION_MAX];
    chunk_strides[rparams->input->dtshape->ndim - 1] = 1;
    for (int i = rparams->input->dtshape->ndim - 2; i >= 0 ; --i) {
        int64_t nchunks_ = rparams->input->catarr->extshape[i + 1] /
                           rparams->input->catarr->chunkshape[i + 1];
        chunk_strides[i] = nchunks_ * chunk_strides[i + 1];
    }

    // Init reduction
    double *dout = (double *) pparams->out;
    float *fout = (float *) pparams->out;
    for (int64_t ind = 0; ind < pparams->out_size / pparams->out_typesize; ++ind) {
        // Compute index in dest
        int64_t elem_index_n[IARRAY_DIMENSION_MAX] = {0};
        index_unidim_to_multidim(rparams->result->catarr->ndim,
                                 rparams->result->storage->blockshape,
                                 ind,
                                 elem_index_n);

        bool empty = check_padding(block_offset_n, elem_index_n, rparams);

        switch (rparams->result->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                if (empty)
                    break;
                else
                    rparams->ufunc->init(dout, &user_data);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                if (empty)
                    break;
                else
                    rparams->ufunc->init(fout, &user_data);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "Invalid dtype");
                return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
        }
        dout++;
        fout++;
    }

    // Allocate destination
    uint8_t *block = malloc(rparams->input->catarr->blocknitems * rparams->input->catarr->itemsize);

    for (int chunk_ind = 0; chunk_ind < nchunks; ++chunk_ind) {
        int64_t nchunk = chunk_ind * chunk_strides[rparams->axis];
        for (int j = 0; j < rparams->result->catarr->ndim; ++j) {
            if (j < rparams->axis)
                nchunk += chunk_offset_n[j] * chunk_strides[j];
            else
                nchunk += chunk_offset_n[j] * chunk_strides[j + 1];
        }
        uint8_t *chunk;
        bool needs_free;
        int csize = blosc2_schunk_get_lazychunk(rparams->input->catarr->sc, nchunk, &chunk,
                                                 &needs_free);
        if (csize < 0) {
            IARRAY_TRACE1(iarray.tracing, "Error getting lazy chunk");
            return -1;
        }

        for (int block_ind = 0; block_ind < nblocks; ++block_ind) {
            int64_t nblock = block_ind * block_strides[rparams->axis];
            for (int j = 0; j < rparams->result->catarr->ndim; ++j) {
                if (j < rparams->axis)
                    nblock += (block_offset_n[j] % shape_of_blocks[j]) * block_strides[j];
                else
                    nblock += (block_offset_n[j] % shape_of_blocks[j]) * block_strides[j + 1];
            }
            int64_t start = nblock * rparams->input->catarr->blocknitems;

            // Compress data
            int bsize = blosc2_getitem_ctx(dctx, chunk, csize, start,
                                           rparams->input->catarr->blocknitems,
                                           block, rparams->input->catarr->blocknitems * rparams->input->catarr->itemsize);
            if (bsize < 0) {
                IARRAY_TRACE1(iarray.tracing, "Error getting block");
                return -1;
            }

            // Check if there are padding in reduction axis
            int64_t aux = block_ind * rparams->input->catarr->blockshape[rparams->axis];
            aux += chunk_ind * rparams->input->catarr->chunkshape[rparams->axis];

            int64_t vector_nelems;
            if (aux + rparams->input->catarr->blockshape[rparams->axis] >
            rparams->input->catarr->shape[rparams->axis]) {
                vector_nelems = rparams->input->catarr->shape[rparams->axis] - aux;
            } else {
                vector_nelems = rparams->input->catarr->blockshape[rparams->axis];
            }

            // Check if there are padding in reduction axis
            aux = block_ind * rparams->input->catarr->blockshape[rparams->axis];

            int64_t vector_nelems2;
            if (aux + rparams->input->catarr->blockshape[rparams->axis] >
                rparams->input->catarr->chunkshape[rparams->axis]) {
                vector_nelems2 = rparams->input->catarr->chunkshape[rparams->axis] - aux;
            } else {
                vector_nelems2 = rparams->input->catarr->blockshape[rparams->axis];
            }

            if (vector_nelems2 < vector_nelems) {
                vector_nelems = vector_nelems2;
            }

            dout = (double *) pparams->out;
            fout = (float *) pparams->out;

            for (int64_t ind = 0; ind < pparams->out_size / pparams->out_typesize; ++ind) {
                // Compute index in dest
                int64_t elem_index_n[IARRAY_DIMENSION_MAX] = {0};
                index_unidim_to_multidim(rparams->result->catarr->ndim,
                                         rparams->result->storage->blockshape,
                                         ind,
                                         elem_index_n);

                if (check_padding(block_offset_n, elem_index_n, rparams)) {
                    dout++;
                    fout++;
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

                double *dblock = ((double *) block) + elem_index_u;
                float *fblock = ((float *) block) + elem_index_u;

                switch (rparams->result->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        rparams->ufunc->reduction(dout, 0, dblock, strides[rparams->axis],
                                                  vector_nelems, &user_data);
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        rparams->ufunc->reduction(fout, 0, fblock, strides[rparams->axis],
                                                  vector_nelems, &user_data);
                        break;
                    default:
                        IARRAY_TRACE1(iarray.error, "Invalid dtype");
                        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);

                }
                dout++;
                fout++;
            }
        }
        if (needs_free) {
            free(chunk);
        }
    }

    // Finish reduction
    dout = (double *) pparams->out;
    fout = (float *) pparams->out;
    for (int64_t ind = 0; ind < pparams->out_size / pparams->out_typesize; ++ind) {
        // Compute index in dest
        int64_t elem_index_n[IARRAY_DIMENSION_MAX] = {0};
        index_unidim_to_multidim(rparams->result->catarr->ndim,
                                 rparams->result->storage->blockshape,
                                 ind,
                                 elem_index_n);

        bool padding = check_padding(block_offset_n, elem_index_n, rparams);

        switch (rparams->result->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                if (padding)
                    *dout = 0;
                else
                    rparams->ufunc->finish(dout, &user_data);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                if (padding)
                    *fout = 0;
                else
                    rparams->ufunc->finish(fout, &user_data);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "Invalid dtype");
                return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
        }
        dout++;
        fout++;
    }

    blosc2_free_ctx(dctx);

    free(block);

    return 0;
}


INA_API(ina_rc_t) _iarray_reduce_udf(iarray_context_t *ctx,
                                     iarray_container_t *a,
                                     iarray_reduce_function_t *ufunc,
                                     int8_t axis,
                                     iarray_storage_t *storage,
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
    if (a->dtshape->ndim < 1) {
        IARRAY_TRACE1(iarray.error, "The container dimensions must be greater than 1");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }

    iarray_dtshape_t dtshape;
    dtshape.dtype = a->dtshape->dtype;
    dtshape.ndim = a->dtshape->ndim - 1;
    for (int i = 0; i < dtshape.ndim; ++i) {
        dtshape.shape[i] = i < axis ? a->dtshape->shape[i] : a->dtshape->shape[i + 1];
    }

    IARRAY_RETURN_IF_FAILED(iarray_empty(ctx, &dtshape, storage, 0, b));

    iarray_container_t *c = *b;

    // Set up prefilter
    iarray_context_t *prefilter_ctx;
    iarray_context_new(ctx->cfg, &prefilter_ctx);
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
    while (nchunk < c->catarr->extnitems / c->catarr->chunknitems) {

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
        reduce_params.nchunk = nchunk;
        // Compress data
        blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
        IARRAY_RETURN_IF_FAILED(iarray_create_blosc_cparams(&cparams, prefilter_ctx, c->catarr->itemsize,
                                                            c->catarr->blocknitems * c->catarr->itemsize));
        cparams.schunk = a->catarr->sc;
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

        blosc2_schunk_update_chunk(c->catarr->sc, nchunk, chunk, false);

        nchunk++;
        index_unidim_to_multidim(c->dtshape->ndim, shape_of_chunks, nchunk, chunk_index);
    }

    iarray_context_free(&prefilter_ctx);

    return INA_SUCCESS;
}

ina_rc_t _iarray_reduce(iarray_context_t *ctx,
                        iarray_container_t *a,
                        iarray_reduce_func_t func,
                        int8_t axis,
                        iarray_storage_t *storage,
                        iarray_container_t **b) {
    void *reduce_funtion = NULL;

    switch (func) {
        case IARRAY_REDUCE_SUM:
            reduce_funtion = a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ?
                    &DSUM :
                    &FSUM;
            break;
        case IARRAY_REDUCE_PROD:
            reduce_funtion = a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ?
                             &DPROD :
                             &FPROD;
            break;
        case IARRAY_REDUCE_MAX:
            reduce_funtion = a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ?
                             &DMAX :
                             &FMAX;
            break;
        case IARRAY_REDUCE_MIN:
            reduce_funtion = a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ?
                             &DMIN :
                             &FMIN;
            break;
        case IARRAY_REDUCE_MEAN:
            reduce_funtion = a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ?
                             &DMEAN :
                             &FMEAN;
            break;
    }

    IARRAY_RETURN_IF_FAILED(_iarray_reduce_udf(ctx, a, reduce_funtion, axis, storage, b));

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_reduce_multi(iarray_context_t *ctx,
                                      iarray_container_t *a,
                                      iarray_reduce_func_t func,
                                      int8_t naxis,
                                      int8_t *axis,
                                      iarray_storage_t *storage,
                                      iarray_container_t **b) {

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(axis);
    INA_VERIFY_NOT_NULL(b);

    int err_io;

    iarray_container_t *aa = a;
    if (naxis > aa->dtshape->ndim) {
        return INA_ERROR(IARRAY_ERR_INVALID_AXIS);
    }

    int8_t axis_new[IARRAY_DIMENSION_MAX] = {0};
    bool axis_used[IARRAY_DIMENSION_MAX] = {0};

    // Check if an axis is higher than array dimensions and if an axis is repeated
    int ii = 0;
    for (int i = 0; i < naxis; ++i) {
        if (axis[i] > aa->dtshape->ndim) {
            return INA_ERROR(IARRAY_ERR_INVALID_AXIS);
        } else if (axis_used[axis[i]]) {
            continue;
        }
        axis_new[ii] = axis[i];
        axis_used[axis[i]] = true;
        ii++;
    }


    // Start reductions
    iarray_container_t *c = NULL;
    for (int i = 0; i < ii; ++i) {
        if (i > 0) {
            if (storage->urlpath != NULL) {
                if (i > 1) {
                    err_io = remove("iarray_red_temp2.iarray");
                    if (err_io != 0) {
                        IARRAY_TRACE1(iarray.tracing, "Invalid io");
                        return INA_ERROR(INA_ERR_OPERATION_INVALID);
                    }
                }
                err_io = rename("iarray_red_temp.iarray", "iarray_red_temp2.iarray");
                if (err_io != 0) {
                    IARRAY_TRACE1(iarray.tracing, "Invalid io");
                    return INA_ERROR(INA_ERR_OPERATION_INVALID);
                }
                IARRAY_RETURN_IF_FAILED(iarray_container_open(ctx, "iarray_red_temp2.iarray", &aa));
            } else {
                aa = c;
            }
        }

        iarray_storage_t storage_red;
        storage_red.backend = IARRAY_STORAGE_BLOSC;
        storage_red.enforce_frame = storage->enforce_frame;
        storage_red.urlpath = storage->urlpath != NULL ? "iarray_red_temp.iarray" : NULL;
        for (int j = 0; j < aa->dtshape->ndim; ++j) {
            if (j < axis_new[i]) {
                storage_red.chunkshape[j] = aa->storage->chunkshape[j];
                storage_red.blockshape[j] = aa->storage->blockshape[j];
            } else {
                storage_red.chunkshape[j] = aa->storage->chunkshape[j + 1];
                storage_red.blockshape[j] = aa->storage->blockshape[j + 1];
            }
        }

        IARRAY_RETURN_IF_FAILED(_iarray_reduce(ctx, aa, func, axis_new[i], &storage_red, &c));
        if (i != 0) {
            iarray_container_free(ctx, &aa);
        }

        for (int j = i + 1; j < ii; ++j) {
            if (axis_new[j] > axis_new[i]) {
                axis_new[j]--;
            }
        }
    }

    // Check if a copy is needed
    bool copy = false;
    for (int i = 0; i < c->dtshape->ndim; ++i) {
        if (storage->chunkshape[i] != c->storage->chunkshape[i]) {
            copy = true;
            break;
        }
        if (storage->blockshape[i] != c->storage->blockshape[i]) {
            copy = true;
            break;
        }
    }

    if (copy) {
        IARRAY_RETURN_IF_FAILED(iarray_copy(ctx, c, false, storage, 0, b));
        iarray_container_free(ctx, &c);
        if (storage->urlpath != NULL) {
            err_io = remove("iarray_red_temp.iarray");
            if (err_io != 0) {
                IARRAY_TRACE1(iarray.tracing, "Invalid io");
                return INA_ERROR(INA_ERR_OPERATION_INVALID);
            }
            if (ii > 1) {
                err_io = remove("iarray_red_temp2.iarray");
                if (err_io != 0) {
                    IARRAY_TRACE1(iarray.tracing, "Invalid io");
                    return INA_ERROR(INA_ERR_OPERATION_INVALID);
                }
            }
        }
    } else {
        if (storage->urlpath != NULL) {
            iarray_container_free(ctx, &c);
            err_io = rename("iarray_red_temp.iarray", storage->urlpath);
            if (err_io != 0) {
                IARRAY_TRACE1(iarray.tracing, "Invalid io");
                return INA_ERROR(INA_ERR_OPERATION_INVALID);
            }
            if (ii > 1) {
                err_io = remove("iarray_red_temp2.iarray");
                if (err_io != 0) {
                    IARRAY_TRACE1(iarray.tracing, "Invalid io");
                    return INA_ERROR(INA_ERR_OPERATION_INVALID);
                }
            }
            IARRAY_RETURN_IF_FAILED(iarray_container_open(ctx, storage->urlpath, b));
        } else {
            *b = c;
        }
    }


    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_reduce(iarray_context_t *ctx,
                                iarray_container_t *a,
                                iarray_reduce_func_t func,
                                int8_t axis,
                                iarray_storage_t *storage,
                                iarray_container_t **b) {

    int8_t axis_[] = {0};
    axis_[0] = axis;

    IARRAY_RETURN_IF_FAILED(iarray_reduce_multi(ctx, a, func, 1, axis_, storage, b));
    return INA_SUCCESS;
}