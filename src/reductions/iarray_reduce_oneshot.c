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


/**
 * Description:
 *
 * The algorithm implemented in iarray_reduce.c reduces the array dimension-by-dimension,
 * creating n-1 (if naxis=n) temporary arrays. Moreover, the std and the var reductions
 * can not be done using that algorithm.
 *
 * However, the algorithm implemented in this file reduce all the dimensions at once,
 * avoiding the temporary files and allowing us to use the var and the std reductions.
 * A consequence of this implementation is that if all axis are reduced, the reduction
 * will be done in single-thread mode.
 *
 * In a future version of ironArray, this algorithm could be used to allow reductions
 * inside the expression machinery.
 *
 */


#include "iarray_private.h"
#include "operations/iarray_reduce_operations.h"
#include "operations/iarray_reduce_private.h"
#include <libiarray/iarray.h>


int64_t iarray_reduce_init(blosc2_prefilter_params *pparams,
                           iarray_reduce_os_params_t *rparams,
                           user_data_os_t *user_data) {
    uint8_t *out = pparams->out;
    for (int i = 0; i < pparams->out_size / pparams->out_typesize; ++i) {
        user_data->median_nelems[i] = 0;
        user_data->i = i;
        rparams->ufunc->init(out, user_data);
        out+= pparams->out_typesize;
    }
    return INA_SUCCESS;
}


int64_t iarray_reduce_finish(blosc2_prefilter_params *pparams,
                           iarray_reduce_os_params_t *rparams,
                           user_data_os_t *user_data) {
    uint8_t *out = pparams->out;
    for (int i = 0; i < pparams->out_size / pparams->out_typesize; ++i) {
        user_data->i = i;
        rparams->ufunc->finish(out, user_data);
        out += pparams->out_typesize;
    }
    return INA_SUCCESS;
}


static bool _iarray_check_output_padding(const int64_t *block_index,
                                         const int64_t *item_index,
                                         iarray_reduce_os_params_t *rparams) {
    int64_t elem_index_n2[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
        elem_index_n2[i] = item_index[i] + block_index[i] *
                                           rparams->result->catarr->blockshape[i];
    }
    for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
        if (rparams->out_chunkshape[i] <= elem_index_n2[i]) {
            return true;
        }
    }
    return false;
}


static bool _iarray_check_input_padding(const int64_t *block_index,
                                         const int64_t *item_index,
                                         const int64_t *input_chunkshape,
                                         iarray_reduce_os_params_t *rparams) {
    int64_t elem_index_n2[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < rparams->input->catarr->ndim; ++i) {
        elem_index_n2[i] = item_index[i] + block_index[i] *
                                           rparams->input->catarr->blockshape[i];
    }
    for (int i = 0; i < rparams->input->catarr->ndim; ++i) {
        if (input_chunkshape[i] <= elem_index_n2[i]) {
            return true;
        }
    }
    return false;
}


int64_t iarray_reduce_item_iter(blosc2_prefilter_params *pparams,
                                iarray_reduce_os_params_t *rparams,
                                user_data_os_t *user_data, int8_t ndim, uint8_t *chunk, uint8_t *block, uint8_t *aux_block, uint8_t *out,
                                int64_t *input_chunkshape,
                                int64_t *block_index,
                                int64_t *item_index, int64_t *item_start, int64_t *item_stop, int64_t *item_strides) {
    item_index[ndim] = item_start[ndim];
    while (item_index[ndim] < item_stop[ndim]) {
        if (ndim < rparams->input->dtshape->ndim - 1) {
            IARRAY_RETURN_IF_FAILED(
                    iarray_reduce_item_iter(pparams, rparams, user_data, ndim + 1, chunk, block, aux_block, out,
                                             input_chunkshape,
                                             block_index, item_index, item_start, item_stop, item_strides));
        } else {
            if (_iarray_check_input_padding(block_index, item_index, input_chunkshape, rparams)) {
                item_index[ndim]++;
                continue;
            }
            int64_t nitem = 0;
            for (int i = 0; i < rparams->input->dtshape->ndim; ++i) {
                nitem += item_index[i] * item_strides[i];

            }
            uint8_t *data1 = &block[nitem * rparams->input->catarr->itemsize];
            uint8_t *data0 = out;
            user_data->mean = aux_block;
            rparams->ufunc->reduction(data0, 0, data1, 0, 0, user_data);
        }
        item_index[ndim]++;
    }
    return INA_SUCCESS;
}


int64_t iarray_reduce_block_iter(blosc2_prefilter_params *pparams,
                                 iarray_reduce_os_params_t *rparams, user_data_os_t *user_data, int8_t ndim,
                                 bool *reduced_axis, uint8_t *chunk, int32_t csize, uint8_t *block, uint8_t *aux_block,
                                 int64_t *input_chunkshape,
                                 int64_t *block_index, int64_t *block_start, int64_t *block_stop, int64_t *block_strides, bool *is_padding,
                                 int64_t **item_start, int64_t **item_stop, int64_t *item_strides) {
    block_index[ndim] = block_start[ndim];
    while (block_index[ndim] < block_stop[ndim]) {
        if (ndim < rparams->input->dtshape->ndim - 1) {
            IARRAY_RETURN_IF_FAILED(
                    iarray_reduce_block_iter(pparams, rparams, user_data, ndim + 1,
                                             reduced_axis, chunk, csize, block, aux_block,
                                             input_chunkshape,
                                             block_index, block_start, block_stop, block_strides, is_padding,
                                             item_start, item_stop, item_strides));
        } else {
            int64_t nblock = 0;
            for (int i = 0; i < rparams->input->dtshape->ndim; ++i) {
                nblock += block_index[i] * block_strides[i];
            }

            int64_t start = nblock * rparams->input->catarr->blocknitems;
            int32_t blocksize = rparams->input->catarr->blocknitems * rparams->input->catarr->itemsize;

            blosc2_dparams dparams = {.nthreads = 1, .schunk = rparams->input->catarr->sc, .postfilter = NULL};
            blosc2_context *dctx = blosc2_create_dctx(dparams);
            int bsize = blosc2_getitem_ctx(dctx, chunk, csize, (int) start,
                                           rparams->input->catarr->blocknitems,
                                           block, blocksize);
            if (bsize < 0) {
                IARRAY_TRACE1(iarray.tracing, "Error getting block");
                return -1;
            }
            blosc2_free_ctx(dctx);

            for (int out_item_offset_u = 0; out_item_offset_u < pparams->out_size / pparams->out_typesize; ++out_item_offset_u) {

                user_data->i = out_item_offset_u;
                user_data->median = &user_data->medians[out_item_offset_u][user_data->median_nelems[out_item_offset_u] * rparams->input->catarr->itemsize];

                if (is_padding[out_item_offset_u]) {
                    continue;
                }
                int64_t item_index[IARRAY_DIMENSION_MAX];
                IARRAY_RETURN_IF_FAILED(
                        iarray_reduce_item_iter(pparams, rparams, user_data, 0, chunk, block, &aux_block[out_item_offset_u * pparams->out_typesize], &pparams->out[out_item_offset_u * pparams->out_typesize],
                                                input_chunkshape,
                                                block_index,
                                                item_index, item_start[out_item_offset_u], item_stop[out_item_offset_u], item_strides));
            }
        }
        block_index[ndim]++;
    }
    return INA_SUCCESS;
}


int64_t iarray_reduce_chunk_iter(blosc2_prefilter_params *pparams,
                                 iarray_reduce_os_params_t *rparams, user_data_os_t *user_data, int8_t ndim,
                                 bool *reduced_axis, uint8_t *block, uint8_t *aux_block,
                                 int64_t *chunk_index, int64_t *chunk_start, int64_t *chunk_stop, int64_t *chunk_strides,
                                 int64_t *block_start, int64_t *block_stop, int64_t *block_strides, bool *is_padding,
                                 int64_t **item_start, int64_t **item_stop, int64_t *item_strides) {
    chunk_index[ndim] = chunk_start[ndim];
    while (chunk_index[ndim] < chunk_stop[ndim]) {
        if (ndim < rparams->input->dtshape->ndim - 1) {
            IARRAY_RETURN_IF_FAILED(
                    iarray_reduce_chunk_iter(pparams, rparams, user_data, ndim + 1,
                                             reduced_axis, block, aux_block,
                                             chunk_index, chunk_start, chunk_stop, chunk_strides,
                                             block_start, block_stop, block_strides, is_padding,
                                             item_start, item_stop, item_strides));
        } else {
            int64_t nchunk = 0;
            for (int i = 0; i < rparams->input->dtshape->ndim; ++i) {
                nchunk += chunk_index[i] * chunk_strides[i];
            }

            int64_t elem_index[IARRAY_DIMENSION_MAX] = {0};
            for (int i = 0; i < rparams->input->dtshape->ndim; ++i) {
                elem_index[i] = chunk_index[i] * rparams->input->catarr->chunkshape[i];
            }
            int64_t input_chunkshape[IARRAY_DIMENSION_MAX] = {0};
            for (int i = 0; i < rparams->input->dtshape->ndim; ++i) {
                if (elem_index[i] + rparams->input->catarr->chunkshape[i] <= rparams->input->catarr->shape[i]) {
                    input_chunkshape[i] = rparams->input->catarr->chunkshape[i];
                } else {
                    input_chunkshape[i] = rparams->input->catarr->shape[i] - elem_index[i];
                }
            }
            uint8_t *chunk;
            bool needs_free;
            int csize = blosc2_schunk_get_lazychunk(rparams->input->catarr->sc, (int) nchunk, &chunk,
                                                    &needs_free);
            if (csize < 0) {
                IARRAY_TRACE1(iarray.tracing, "Error getting lazy chunk");
                return -1;
            }

            int64_t block_index[IARRAY_DIMENSION_MAX];
            IARRAY_RETURN_IF_FAILED(
                    iarray_reduce_block_iter(pparams, rparams, user_data, 0,
                                             reduced_axis, chunk, csize, block, aux_block,
                                             input_chunkshape,
                                             block_index, block_start, block_stop, block_strides, is_padding,
                                             item_start, item_stop, item_strides));

            if(needs_free) {
                free(chunk);
            }

        }
        chunk_index[ndim]++;
    }
    return INA_SUCCESS;
}


static int _reduce_general_prefilter(blosc2_prefilter_params *pparams) {
    iarray_reduce_os_params_t *rparams = (iarray_reduce_os_params_t *) pparams->user_data;
    user_data_os_t user_data = {0};
    user_data.inv_nelem = 1.;
    for (int i = 0; i < rparams->naxis; ++i) {
        user_data.inv_nelem /= (double) rparams->input->dtshape->shape[rparams->axis[i]];
    }
    user_data.input_itemsize = rparams->input->dtshape->dtype_size;
    user_data.not_nan_nelems = malloc(
            (pparams->out_size / pparams->out_typesize) * sizeof(int64_t));
    user_data.nan_nelems = malloc(
            (pparams->out_size / pparams->out_typesize) * sizeof(int64_t));
    user_data.rparams = rparams;
    user_data.pparams = pparams;
    user_data.medians = malloc((pparams->out_size / pparams->out_typesize) * sizeof(uint8_t *)); \
    user_data.median_nelems = malloc(
            (pparams->out_size / pparams->out_typesize) * sizeof(int64_t));

    int8_t in_ndim = rparams->input->dtshape->ndim;

    bool reduced_axis[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < rparams->naxis; ++i) {
        reduced_axis[rparams->axis[i]] = true;
    }
    user_data.reduced_items = 1; \
    for (int i = 0; i < rparams->input->catarr->ndim; ++i) {
        if (reduced_axis[i]) {
            user_data.reduced_items *= rparams->input->catarr->shape[i];
        }
    }

    uint8_t *aux_block;
    if (rparams->func == IARRAY_REDUCE_VAR || rparams->func == IARRAY_REDUCE_NAN_VAR ||
        rparams->func == IARRAY_REDUCE_STD || rparams->func == IARRAY_REDUCE_NAN_STD) {
        int64_t aux_start = pparams->nblock * rparams->aux->catarr->blocknitems;
        int32_t aux_blocksize = rparams->aux->catarr->blocknitems * rparams->aux->catarr->itemsize;
        aux_block = malloc(aux_blocksize);
        blosc2_dparams dparams = {.nthreads = 1, .schunk = rparams->aux->catarr->sc, .postfilter = NULL};
        blosc2_context *dctx = blosc2_create_dctx(dparams);
        int bsize = blosc2_getitem_ctx(dctx, rparams->aux_chunk, rparams->aux_csize, (int) aux_start,
                                       rparams->aux->catarr->blocknitems,
                                       aux_block, aux_blocksize);
        if (bsize < 0) {
            IARRAY_TRACE1(iarray.tracing, "Error getting aux block");
            return -1;
        }
        blosc2_free_ctx(dctx);
    }

    // Compute chunk-related variables
    int64_t out_chunk_offset_u = rparams->nchunk;
    int64_t out_chunk_offset_n[IARRAY_DIMENSION_MAX] = {0};

    // The number of chunks that output has in each dimension
    int64_t out_chunks_shape[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
        out_chunks_shape[i] = rparams->result->catarr->extshape[i] /
                              rparams->result->catarr->chunkshape[i];
    }

    iarray_index_unidim_to_multidim_shape(rparams->result->catarr->ndim,
                                          out_chunks_shape,
                                          out_chunk_offset_u,
                                          out_chunk_offset_n);

    // The number of chunks that input has in each dimension
    int64_t in_chunks_shape[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < rparams->input->catarr->ndim; ++i) {
        in_chunks_shape[i] = rparams->input->catarr->extshape[i] /
                             rparams->input->catarr->chunkshape[i];
    }

    // The strides of the input chunks
    int64_t in_chunks_strides[IARRAY_DIMENSION_MAX];
    in_chunks_strides[rparams->input->dtshape->ndim - 1] = 1;
    for (int i = rparams->input->dtshape->ndim - 2; i >= 0 ; --i) {
        in_chunks_strides[i] = in_chunks_shape[i + 1] * in_chunks_strides[i + 1];
    }

    // The stop index of the input chunks
    int64_t in_chunks_stop[IARRAY_DIMENSION_MAX] = {0};
    int8_t j = 0;
    for (int i = 0; i < rparams->input->catarr->ndim; ++i) {
        if (reduced_axis[i]) {
            in_chunks_stop[i] = rparams->input->catarr->extshape[i] /
                                rparams->input->catarr->chunkshape[i];
        } else {
            in_chunks_stop[i] = out_chunk_offset_n[j++] + 1;
        }
    }

    // The start index of the input chunks
    int64_t in_chunks_start[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < in_ndim; ++i) {
        if (reduced_axis[i]) {
            in_chunks_start[i] = 0;
        } else {
            in_chunks_start[i] = in_chunks_stop[i] - 1;
        }
    }

    int64_t chunk_index[IARRAY_DIMENSION_MAX] = {0};

    // Compute block-related variables
    int64_t out_block_offset_u = pparams->nblock;
    int64_t out_block_offset_n[IARRAY_DIMENSION_MAX] = {0};

    // The number of blocks that output has in each dimension
    int64_t out_blocks_shape[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
        out_blocks_shape[i] = rparams->result->catarr->extchunkshape[i] /
                              rparams->result->catarr->blockshape[i];
    }

    iarray_index_unidim_to_multidim_shape(rparams->result->catarr->ndim,
                                          out_blocks_shape,
                                          out_block_offset_u,
                                          out_block_offset_n);

    // The number of blocks that input has in each dimension
    int64_t in_blocks_shape[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < rparams->input->catarr->ndim; ++i) {
        in_blocks_shape[i] = rparams->input->catarr->extchunkshape[i] /
                             rparams->input->catarr->blockshape[i];
    }

    // The strides of the input blocks
    int64_t in_blocks_strides[IARRAY_DIMENSION_MAX];
    in_blocks_strides[rparams->input->dtshape->ndim - 1] = 1;
    for (int i = rparams->input->dtshape->ndim - 2; i >= 0 ; --i) {
        in_blocks_strides[i] = in_blocks_shape[i + 1] * in_blocks_strides[i + 1];
    }


    // The stop index of the input blocks
    int64_t in_blocks_stop[IARRAY_DIMENSION_MAX] = {0};
    j = 0;
    for (int i = 0; i < rparams->input->catarr->ndim; ++i) {
        if (reduced_axis[i]) {
            in_blocks_stop[i] = rparams->input->catarr->extchunkshape[i] /
                                rparams->input->catarr->blockshape[i];
        } else {
            in_blocks_stop[i] = out_block_offset_n[j++] + 1;
        }
    }

    // The start index of the input blocks
    int64_t in_blocks_start[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < in_ndim; ++i) {
        if (reduced_axis[i]) {
            in_blocks_start[i] = 0;
        } else {
            in_blocks_start[i] = in_blocks_stop[i] - 1;
        }
    }

    // The number of items that output has in each dimension
    int64_t out_items_shape[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < rparams->result->catarr->ndim; ++i) {
        out_items_shape[i] = rparams->result->catarr->blockshape[i];
    }

    // The number of items that input has in each dimension
    int64_t in_items_shape[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < rparams->input->catarr->ndim; ++i) {
        in_items_shape[i] = rparams->input->catarr->blockshape[i];
    }

    // The strides of the input items
    int64_t in_items_strides[IARRAY_DIMENSION_MAX];
    in_items_strides[rparams->input->dtshape->ndim - 1] = 1;
    for (int i = rparams->input->dtshape->ndim - 2; i >= 0 ; --i) {
        in_items_strides[i] = in_items_shape[i + 1] * in_items_strides[i + 1];
    }

    bool *is_padding = malloc((pparams->out_size / pparams->out_typesize) * sizeof(bool));
    int64_t **in_item_start = (int64_t **) malloc((pparams->out_size / pparams->out_typesize) * sizeof(int64_t *));
    int64_t **in_item_stop = (int64_t **) malloc((pparams->out_size / pparams->out_typesize) * sizeof(int64_t *));
    for (int out_item_offset_u = 0; out_item_offset_u < pparams->out_size / pparams->out_typesize; ++out_item_offset_u) {

        int64_t out_item_offset_n[IARRAY_DIMENSION_MAX] = {0};

        iarray_index_unidim_to_multidim_shape(rparams->result->catarr->ndim,
                                              out_items_shape,
                                              out_item_offset_u,
                                              out_item_offset_n);

        in_item_stop[out_item_offset_u] = (int64_t *) malloc(sizeof(int64_t) * IARRAY_DIMENSION_MAX);
        j = 0;
        for (int i = 0; i < rparams->input->catarr->ndim; ++i) {
            if (reduced_axis[i]) {
                in_item_stop[out_item_offset_u][i] = rparams->input->catarr->blockshape[i];
            } else {
                in_item_stop[out_item_offset_u][i] = out_item_offset_n[j++] + 1;
            }
        }

        in_item_start[out_item_offset_u] =  (int64_t *) malloc(sizeof(int64_t) * IARRAY_DIMENSION_MAX);
        for (int i = 0; i < rparams->input->catarr->ndim; ++i) {
            if (reduced_axis[i]) {
                in_item_start[out_item_offset_u][i] = 0;
            } else {
                in_item_start[out_item_offset_u][i] = in_item_stop[out_item_offset_u][i] - 1;
            }
        }

        is_padding[out_item_offset_u] =
                _iarray_check_output_padding(out_block_offset_n, out_item_offset_n, rparams);

    }

    IARRAY_RETURN_IF_FAILED(
            iarray_reduce_init(pparams, rparams, &user_data));

    uint8_t *block = malloc(rparams->input->catarr->blocknitems * rparams->input->catarr->itemsize);
    IARRAY_RETURN_IF_FAILED(
            iarray_reduce_chunk_iter(pparams, rparams, &user_data, 0,
                                     reduced_axis, block, aux_block,
                                     chunk_index, in_chunks_start, in_chunks_stop, in_chunks_strides,
                                     in_blocks_start, in_blocks_stop, in_blocks_strides, is_padding,
                                     in_item_start, in_item_stop, in_items_strides));
    IARRAY_RETURN_IF_FAILED(
            iarray_reduce_finish(pparams, rparams, &user_data));

    free(block);

    for (int out_item_offset_u = 0; out_item_offset_u < pparams->out_size / pparams->out_typesize; ++out_item_offset_u) {
        free(in_item_stop[out_item_offset_u]);
        free(in_item_start[out_item_offset_u]);
    }
    free(in_item_start);
    free(in_item_stop);
    free(is_padding);
    if (rparams->func == IARRAY_REDUCE_VAR || rparams->func == IARRAY_REDUCE_NAN_VAR ||
        rparams->func == IARRAY_REDUCE_STD || rparams->func == IARRAY_REDUCE_NAN_STD) {
        free(aux_block);
    }
    free(user_data.not_nan_nelems);
    free(user_data.nan_nelems);
    free(user_data.median_nelems);
    free(user_data.medians);

    return 0;
}


ina_rc_t
_iarray_reduce2_udf(iarray_context_t *ctx, iarray_container_t *a, iarray_reduce_function_t *ufunc,
                    iarray_reduce_func_t func,
                    int8_t naxis, const int8_t *axis, iarray_storage_t *storage,
                    iarray_container_t **b, iarray_data_type_t res_dtype, iarray_container_t *aux) {

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(ufunc);
    INA_VERIFY_NOT_NULL(b);

    if (a->dtshape->ndim < 1) {
        IARRAY_TRACE1(iarray.error, "The container dimensions must be greater than 1");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }

    iarray_dtshape_t dtshape;
    dtshape.dtype = res_dtype;
    dtshape.ndim = (int8_t) (a->dtshape->ndim - naxis);

    for (int i = 0; i < naxis; ++i) {
        dtshape.shape[axis[i]] = -1;
    }
    int inc = 0;
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        if (dtshape.shape[i] == -1) {
            inc++;
        } else {
            dtshape.shape[i - inc] = a->dtshape->shape[i];
        }
    }

    IARRAY_RETURN_IF_FAILED(iarray_empty(ctx, &dtshape, storage, b));

    iarray_container_t *c = *b;

    // Set up prefilter
    iarray_context_t *prefilter_ctx;
    iarray_context_new(ctx->cfg, &prefilter_ctx);
    prefilter_ctx->prefilter_fn = (blosc2_prefilter_fn) _reduce_general_prefilter;
    iarray_reduce_os_params_t reduce_params = {0};
    blosc2_prefilter_params pparams = {0};
    pparams.user_data = &reduce_params;
    prefilter_ctx->prefilter_params = &pparams;

    // Fill prefilter params
    reduce_params.input = a;
    reduce_params.result = c;
    reduce_params.naxis = naxis;
    reduce_params.axis = axis;
    reduce_params.ufunc = ufunc;
    reduce_params.func = func;
    reduce_params.aux = aux;
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
        reduce_params.out_chunkshape = chunk_shape;
        reduce_params.nchunk = nchunk;
        uint8_t *aux_chunk = NULL;
        bool aux_needs_free;
        if (func == IARRAY_REDUCE_VAR ||
            func == IARRAY_REDUCE_NAN_VAR ||
            func == IARRAY_REDUCE_STD ||
            func == IARRAY_REDUCE_NAN_STD ) {
            reduce_params.aux_csize = blosc2_schunk_get_lazychunk(aux->catarr->sc, nchunk, &aux_chunk,
                                                                  &aux_needs_free);
            if (reduce_params.aux_csize < 0) {
                IARRAY_TRACE1(iarray.tracing, "Error getting lazy chunk");
                return -1;
            }
            reduce_params.aux_chunk = aux_chunk;
        }

        // Compress data
        blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
        IARRAY_RETURN_IF_FAILED(iarray_create_blosc_cparams(&cparams, prefilter_ctx, c->catarr->itemsize,
                                                            c->catarr->blocknitems * c->catarr->itemsize));
        cparams.schunk = a->catarr->sc;
        blosc2_context *cctx = blosc2_create_cctx(cparams);
        uint8_t *chunk = malloc(c->catarr->extchunknitems * c->catarr->itemsize +
                                BLOSC_MAX_OVERHEAD);
        int csize = blosc2_compress_ctx(cctx, NULL, (int32_t) (c->catarr->extchunknitems * c->catarr->itemsize),
                                        chunk,
                                        (int32_t) (c->catarr->extchunknitems * c->catarr->itemsize +
                                        BLOSC_MAX_OVERHEAD));
        if (csize <= 0) {
            IARRAY_TRACE1(iarray.error, "Error compressing a blosc chunk");
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }
        blosc2_free_ctx(cctx);

        blosc2_schunk_update_chunk(c->catarr->sc, (int) nchunk, chunk, false);

        nchunk++;
        iarray_index_unidim_to_multidim_shape(c->dtshape->ndim, shape_of_chunks, nchunk,
                                              chunk_index);
        if (func == IARRAY_REDUCE_VAR ||
            func == IARRAY_REDUCE_NAN_VAR ||
            func == IARRAY_REDUCE_STD ||
            func == IARRAY_REDUCE_NAN_STD ) {
            if (aux_needs_free) {
                free(aux_chunk);
            }
        }
    }

    iarray_context_free(&prefilter_ctx);

    return INA_SUCCESS;
}

ina_rc_t _iarray_reduce2(iarray_context_t *ctx,
                        iarray_container_t *a,
                        iarray_reduce_func_t func,
                         int8_t naxis,
                         const int8_t *axis,
                        iarray_storage_t *storage,
                        iarray_container_t **b) {
    void *reduce_function = NULL;
    // res data type
    iarray_data_type_t dtype;
    switch (func) {
        case IARRAY_REDUCE_VAR:
            // If the input is of type integer or unsigned int the result will be of type double
            switch (a->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    reduce_function = &DVAR;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    reduce_function = &FVAR;
                    dtype = IARRAY_DATA_TYPE_FLOAT;
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    reduce_function = &I64VAR;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    reduce_function = &I32VAR;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_INT16:
                    reduce_function = &I16VAR;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_INT8:
                    reduce_function = &I8VAR;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    reduce_function = &UI64VAR;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    reduce_function = &UI32VAR;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT16:
                    reduce_function = &UI16VAR;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT8:
                    reduce_function = &UI8VAR;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_BOOL:
                    reduce_function = &BOOLVAR;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Invalid dtype");
                    return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
            }
            break;
        case IARRAY_REDUCE_NAN_VAR:
            // If the input is of type integer or unsigned int the result will be of type double
            switch (a->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    reduce_function = &DNANVAR;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    reduce_function = &FNANVAR;
                    dtype = IARRAY_DATA_TYPE_FLOAT;
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Invalid dtype");
                    return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
            }
            break;
        case IARRAY_REDUCE_NAN_STD:
            // If the input is of type integer or unsigned int the result will be of type double
            switch (a->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    reduce_function = &DNANSTD;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    reduce_function = &FNANSTD;
                    dtype = IARRAY_DATA_TYPE_FLOAT;
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Invalid dtype");
                    return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
            }
            break;
        case IARRAY_REDUCE_STD:
            // If the input is of type integer or unsigned int the result will be of type double
            switch (a->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    reduce_function = &DSTD;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    reduce_function = &FSTD;
                    dtype = IARRAY_DATA_TYPE_FLOAT;
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    reduce_function = &I64STD;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    reduce_function = &I32STD;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_INT16:
                    reduce_function = &I16STD;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_INT8:
                    reduce_function = &I8STD;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    reduce_function = &UI64STD;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    reduce_function = &UI32STD;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT16:
                    reduce_function = &UI16STD;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT8:
                    reduce_function = &UI8STD;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_BOOL:
                    reduce_function = &BOOLSTD;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Invalid dtype");
                    return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
            }
            break;
        case IARRAY_REDUCE_MEDIAN:
            // If the input is of type integer or unsigned int the result will be of type double
            switch (a->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    reduce_function = &REDUCTION(MEDIAN, double);
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    reduce_function = &REDUCTION(MEDIAN, float);
                    dtype = IARRAY_DATA_TYPE_FLOAT;
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    reduce_function = &REDUCTION(MEDIAN, int64_t);
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    reduce_function = &REDUCTION(MEDIAN, int32_t);
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_INT16:
                    reduce_function = &REDUCTION(MEDIAN, int16_t);
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_INT8:
                    reduce_function = &REDUCTION(MEDIAN, int8_t);
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    reduce_function = &REDUCTION(MEDIAN, uint64_t);
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    reduce_function = &REDUCTION(MEDIAN, uint32_t);
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT16:
                    reduce_function = &REDUCTION(MEDIAN, uint16_t);
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT8:
                    reduce_function = &REDUCTION(MEDIAN, uint8_t);
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_BOOL:
                    reduce_function = &REDUCTION(MEDIAN, bool);
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Invalid dtype");
                    return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
            }
            break;
        case IARRAY_REDUCE_NAN_MEDIAN:
            switch (a->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    reduce_function = &NANREDUCTION(MEDIAN, double);
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    reduce_function = &NANREDUCTION(MEDIAN, float);
                    dtype = IARRAY_DATA_TYPE_FLOAT;
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Invalid dtype");
                    return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
            }
            break;
        case IARRAY_REDUCE_NAN_MEAN:
            switch (a->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    reduce_function = &REDUCTION(NANMEAN, double);
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    reduce_function = &REDUCTION(NANMEAN, float);
                    dtype = IARRAY_DATA_TYPE_FLOAT;
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Invalid dtype");
                    return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
            }
            break;
        default:
            IARRAY_TRACE1(iarray.error, "Invalid function");
            return INA_ERROR(IARRAY_ERR_INVALID_EVAL_METHOD);
    }

    iarray_container_t *mean = NULL;
    iarray_storage_t mean_storage;
    memcpy(&mean_storage, storage, sizeof(iarray_storage_t));
    mean_storage.urlpath = storage->urlpath ? "_iarray_mean_reduce.iarray" : NULL;

    switch (func) {
        case IARRAY_REDUCE_STD:
        case IARRAY_REDUCE_VAR:
            IARRAY_RETURN_IF_FAILED(
                    iarray_reduce_multi(ctx, a, IARRAY_REDUCE_MEAN, naxis, axis, &mean_storage, &mean));
            break;
        case IARRAY_REDUCE_NAN_STD:
        case IARRAY_REDUCE_NAN_VAR:
            IARRAY_RETURN_IF_FAILED(
                    iarray_reduce_multi(ctx, a, IARRAY_REDUCE_NAN_MEAN, naxis, axis, &mean_storage, &mean));
            break;
        default:
            ;
    }

    IARRAY_RETURN_IF_FAILED(
            _iarray_reduce2_udf(ctx, a, reduce_function, func, naxis, axis, storage, b, dtype,
                                mean));
    switch (func) {
        case IARRAY_REDUCE_STD:
        case IARRAY_REDUCE_VAR:
        case IARRAY_REDUCE_NAN_STD:
        case IARRAY_REDUCE_NAN_VAR:
            iarray_container_free(ctx, &mean);
            iarray_container_remove(mean_storage.urlpath);
            break;
        default:
            ;
    }

    return INA_SUCCESS;
}

ina_rc_t _iarray_reduce_oneshot(iarray_context_t *ctx,
                                iarray_container_t *a,
                                iarray_reduce_func_t func,
                                int8_t naxis,
                                const int8_t *axis,
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

    bool axis_used[IARRAY_DIMENSION_MAX] = {0};
    // Check if an axis is higher than array dimensions and if an axis is repeated
    int ii = 0;
    for (int i = 0; i < naxis; ++i) {
        if (axis[i] > aa->dtshape->ndim || axis_used[axis[i]]) {
            return INA_ERROR(IARRAY_ERR_INVALID_AXIS);
        }
        axis_used[axis[i]] = true;
        ii++;
    }

    if (a->container_viewed != NULL) {
        iarray_storage_t view_storage = {0};
        memcpy(&view_storage, a->storage, sizeof(iarray_storage_t));
        if (a->storage->urlpath) {
            view_storage.urlpath = "_iarray_view.iarr";
            if (access(view_storage.urlpath, 0) == 0) {
                IARRAY_TRACE1(iarray.tracing, "The temporary file %s already exists, delete it first",
                              view_storage.urlpath);
                return INA_ERROR(INA_ERR_INVALID);
            }
        }
        iarray_copy(ctx, a, false, &view_storage, &aa);
    }

    // Start reductions
    iarray_container_t *c = NULL;
    iarray_storage_t storage_red;
    storage_red.contiguous = storage->contiguous;
    storage_red.urlpath = storage->urlpath != NULL ? "_iarray_red.iarr" : NULL;
    for (int j = 0; j < aa->dtshape->ndim; ++j) {
        storage_red.chunkshape[j] = aa->storage->chunkshape[j];
        storage_red.blockshape[j] = aa->storage->blockshape[j];
    }
    if (storage_red.urlpath != NULL && access(storage_red.urlpath, 0) == 0) {
        IARRAY_TRACE1(iarray.tracing, "The temporary file %s already exists, delete it first", storage_red.urlpath);
        return INA_ERROR(INA_ERR_INVALID);
    }

    for (int i = 0; i < naxis; ++i) {
        storage_red.chunkshape[axis[i]] = -1;
        storage_red.blockshape[axis[i]] = -1;
    }

    int inc = 0;
    for (int i = 0; i < aa->dtshape->ndim; ++i) {
        if (storage_red.chunkshape[i] == -1) {
            inc += 1;
        } else {
            storage_red.chunkshape[i - inc] = storage_red.chunkshape[i];
            storage_red.blockshape[i - inc] = storage_red.blockshape[i];
        }
    }

    IARRAY_RETURN_IF_FAILED(_iarray_reduce2(ctx, aa, func, naxis, axis, &storage_red, &c));


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
        IARRAY_RETURN_IF_FAILED(iarray_copy(ctx, c, false, storage, b));
        iarray_container_free(ctx, &c);
        if (storage->urlpath != NULL) {
            err_io = blosc2_remove_urlpath("_iarray_red.iarr");
            if (err_io != 0) {
                IARRAY_TRACE1(iarray.tracing, "Invalid io");
                return INA_ERROR(INA_ERR_OPERATION_INVALID);
            }
        }
    } else {
        if (storage->urlpath != NULL) {
            iarray_container_free(ctx, &c);
            err_io = blosc2_rename_urlpath("_iarray_red.iarr", storage->urlpath);
            if (err_io != 0) {
                IARRAY_TRACE1(iarray.tracing, "Invalid io");
                return INA_ERROR(INA_ERR_OPERATION_INVALID);
            }
            IARRAY_RETURN_IF_FAILED(iarray_container_open(ctx, storage->urlpath, b));
        } else {
            *b = c;
        }
    }

    if (storage->urlpath != NULL) {
        blosc2_remove_urlpath("_iarray_red.iarr");
    }
    if (a->container_viewed != NULL && a->storage->urlpath != NULL) {
        blosc2_remove_urlpath("_iarray_view.iarr");
    }

    return INA_SUCCESS;
}
