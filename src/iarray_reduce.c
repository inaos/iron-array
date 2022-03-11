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
    int64_t *i64out = (int64_t *) pparams->out;
    int32_t *i32out = (int32_t *) pparams->out;
    int16_t *i16out = (int16_t *) pparams->out;
    int8_t *i8out = (int8_t *) pparams->out;
    uint64_t *ui64out = (uint64_t *) pparams->out;
    uint32_t *ui32out = (uint32_t *) pparams->out;
    uint16_t *ui16out = (uint16_t *) pparams->out;
    uint8_t *ui8out = (uint8_t *) pparams->out;
    bool *boolout = (bool *) pparams->out;

    for (int64_t ind = 0; ind < pparams->out_size / pparams->out_typesize; ++ind) {
        // Compute index in dest
        int64_t elem_index_n[IARRAY_DIMENSION_MAX] = {0};
        index_unidim_to_multidim(rparams->result->catarr->ndim,
                                 rparams->result->storage->blockshape,
                                 ind,
                                 elem_index_n);

        bool empty = check_padding(block_offset_n, elem_index_n, rparams);

        if (!empty) {
            switch (rparams->result->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    rparams->ufunc->init(dout, &user_data);
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    rparams->ufunc->init(fout, &user_data);
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    rparams->ufunc->init(i64out, &user_data);
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    rparams->ufunc->init(i32out, &user_data);
                    break;
                case IARRAY_DATA_TYPE_INT16:
                    rparams->ufunc->init(i16out, &user_data);
                    break;
                case IARRAY_DATA_TYPE_INT8:
                    rparams->ufunc->init(i8out, &user_data);
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    rparams->ufunc->init(ui64out, &user_data);
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    rparams->ufunc->init(ui32out, &user_data);
                    break;
                case IARRAY_DATA_TYPE_UINT16:
                    rparams->ufunc->init(ui16out, &user_data);
                    break;
                case IARRAY_DATA_TYPE_UINT8:
                    rparams->ufunc->init(ui8out, &user_data);
                    break;
                case IARRAY_DATA_TYPE_BOOL:
                    rparams->ufunc->init(boolout, &user_data);
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Invalid dtype");
                    return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
            }
        }
        switch (rparams->result->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                dout++;
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                fout++;
                break;
            case IARRAY_DATA_TYPE_INT64:
                i64out++;
                break;
            case IARRAY_DATA_TYPE_INT32:
                i32out++;
                break;
            case IARRAY_DATA_TYPE_INT16:
                i16out++;
                break;
            case IARRAY_DATA_TYPE_INT8:
                i8out++;
                break;
            case IARRAY_DATA_TYPE_UINT64:
                ui64out++;
                break;
            case IARRAY_DATA_TYPE_UINT32:
                ui32out++;
                break;
            case IARRAY_DATA_TYPE_UINT16:
                ui16out++;
                break;
            case IARRAY_DATA_TYPE_UINT8:
                ui8out++;
                break;
            case IARRAY_DATA_TYPE_BOOL:
                boolout++;
                break;
            default:
                IARRAY_TRACE1(iarray.error, "Invalid dtype");
                return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
        }
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
            i64out = (int64_t *) pparams->out;
            i32out = (int32_t *) pparams->out;
            i16out = (int16_t *) pparams->out;
            i8out = (int8_t *) pparams->out;
            ui64out = (uint64_t *) pparams->out;
            ui32out = (uint32_t *) pparams->out;
            ui16out = (uint16_t *) pparams->out;
            ui8out = (uint8_t *) pparams->out;
            boolout = (bool *) pparams->out;

            for (int64_t ind = 0; ind < pparams->out_size / pparams->out_typesize; ++ind) {
                // Compute index in dest
                int64_t elem_index_n[IARRAY_DIMENSION_MAX] = {0};
                index_unidim_to_multidim(rparams->result->catarr->ndim,
                                         rparams->result->storage->blockshape,
                                         ind,
                                         elem_index_n);

                if (check_padding(block_offset_n, elem_index_n, rparams)) {
                    switch (rparams->result->dtshape->dtype) {
                        case IARRAY_DATA_TYPE_DOUBLE:
                            dout++;
                            break;
                        case IARRAY_DATA_TYPE_FLOAT:
                            fout++;
                            break;
                        case IARRAY_DATA_TYPE_INT64:
                            i64out++;
                            break;
                        case IARRAY_DATA_TYPE_INT32:
                            i32out++;
                            break;
                        case IARRAY_DATA_TYPE_INT16:
                            i16out++;
                            break;
                        case IARRAY_DATA_TYPE_INT8:
                            i8out++;
                            break;
                        case IARRAY_DATA_TYPE_UINT64:
                            ui64out++;
                            break;
                        case IARRAY_DATA_TYPE_UINT32:
                            ui32out++;
                            break;
                        case IARRAY_DATA_TYPE_UINT16:
                            ui16out++;
                            break;
                        case IARRAY_DATA_TYPE_UINT8:
                            ui8out++;
                            break;
                        case IARRAY_DATA_TYPE_BOOL:
                            boolout++;
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

                double *dblock = ((double *) block) + elem_index_u;
                float *fblock = ((float *) block) + elem_index_u;
                int64_t *i64block = ((int64_t *) block) + elem_index_u;
                int32_t *i32block = ((int32_t *) block) + elem_index_u;
                int16_t *i16block = ((int16_t *) block) + elem_index_u;
                int8_t *i8block = ((int8_t *) block) + elem_index_u;
                uint64_t *ui64block = ((uint64_t *) block) + elem_index_u;
                uint32_t *ui32block = ((uint32_t *) block) + elem_index_u;
                uint16_t *ui16block = ((uint16_t *) block) + elem_index_u;
                uint8_t *ui8block = ((uint8_t *) block) + elem_index_u;
                bool *boolblock = ((bool *) block) + elem_index_u;

                switch (rparams->result->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        switch (rparams->input->dtshape->dtype) {
                            case IARRAY_DATA_TYPE_DOUBLE:
                                rparams->ufunc->reduction(dout, 0, dblock, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_FLOAT:
                                rparams->ufunc->reduction(dout, 0, fblock, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_INT64:
                                rparams->ufunc->reduction(dout, 0, i64block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_INT32:
                                rparams->ufunc->reduction(dout, 0, i32block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_INT16:
                                rparams->ufunc->reduction(dout, 0, i16block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_INT8:
                                rparams->ufunc->reduction(dout, 0, i8block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_UINT64:
                                rparams->ufunc->reduction(dout, 0, ui64block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_UINT32:
                                rparams->ufunc->reduction(dout, 0, ui32block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_UINT16:
                                rparams->ufunc->reduction(dout, 0, ui16block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_UINT8:
                                rparams->ufunc->reduction(dout, 0, ui8block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_BOOL:
                                rparams->ufunc->reduction(dout, 0, boolblock, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                        }
                        dout++;
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        rparams->ufunc->reduction(fout, 0, fblock, strides[rparams->axis],
                                                  vector_nelems, &user_data);
                        fout++;
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        // If the reduction is a sum or a product the result is of type int64_t but the source may not
                        switch (rparams->input->dtshape->dtype) {
                            case IARRAY_DATA_TYPE_INT64:
                                rparams->ufunc->reduction(i64out, 0, i64block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_INT32:
                                rparams->ufunc->reduction(i64out, 0, i32block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_INT16:
                                rparams->ufunc->reduction(i64out, 0, i16block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_INT8:
                                rparams->ufunc->reduction(i64out, 0, i8block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_BOOL:
                                rparams->ufunc->reduction(i64out, 0, boolblock, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                        }
                        i64out++;
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        rparams->ufunc->reduction(i32out, 0, i32block, strides[rparams->axis],
                                                  vector_nelems, &user_data);
                        i32out++;
                        break;
                    case IARRAY_DATA_TYPE_INT16:
                        rparams->ufunc->reduction(i16out, 0, i16block, strides[rparams->axis],
                                                  vector_nelems, &user_data);
                        i16out++;
                        break;
                    case IARRAY_DATA_TYPE_INT8:
                        rparams->ufunc->reduction(i8out, 0, i8block, strides[rparams->axis],
                                                  vector_nelems, &user_data);
                        i8out++;
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        // If the reduction is a sum or a product the result is of type uint64_t but the source may not
                        switch (rparams->input->dtshape->dtype) {
                            case IARRAY_DATA_TYPE_UINT64:
                                rparams->ufunc->reduction(ui64out, 0, ui64block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_UINT32:
                                rparams->ufunc->reduction(ui64out, 0, ui32block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_UINT16:
                                rparams->ufunc->reduction(ui64out, 0, ui16block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                            case IARRAY_DATA_TYPE_UINT8:
                                rparams->ufunc->reduction(ui64out, 0, ui8block, strides[rparams->axis],
                                                          vector_nelems, &user_data);
                                break;
                        }
                        ui64out++;
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        rparams->ufunc->reduction(ui32out, 0, ui32block, strides[rparams->axis],
                                                  vector_nelems, &user_data);
                        ui32out++;
                        break;
                    case IARRAY_DATA_TYPE_UINT16:
                        rparams->ufunc->reduction(ui16out, 0, ui16block, strides[rparams->axis],
                                                  vector_nelems, &user_data);
                        ui16out++;
                        break;
                    case IARRAY_DATA_TYPE_UINT8:
                        rparams->ufunc->reduction(ui8out, 0, ui8block, strides[rparams->axis],
                                                  vector_nelems, &user_data);
                        ui8out++;
                        break;
                    case IARRAY_DATA_TYPE_BOOL:
                        rparams->ufunc->reduction(boolout, 0, boolblock, strides[rparams->axis],
                                                  vector_nelems, &user_data);
                        boolout++;
                        break;
                    default:
                        IARRAY_TRACE1(iarray.error, "Invalid dtype");
                        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
                }
            }
        }
        if (needs_free) {
            free(chunk);
        }
    }

    // Finish reduction
    dout = (double *) pparams->out;
    fout = (float *) pparams->out;
    i64out = (int64_t *) pparams->out;
    i32out = (int32_t *) pparams->out;
    i16out = (int16_t *) pparams->out;
    i8out = (int8_t *) pparams->out;
    ui64out = (uint64_t *) pparams->out;
    ui32out = (uint32_t *) pparams->out;
    ui16out = (uint16_t *) pparams->out;
    ui8out = (uint8_t *) pparams->out;
    boolout = (bool *) pparams->out;
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
                if (padding) {
                    *dout = 0;
                } else {
                    rparams->ufunc->finish(dout, &user_data);
                }
                dout++;
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                if (padding) {
                    *fout = 0;
                } else {
                    rparams->ufunc->finish(fout, &user_data);
                }
                fout++;
                break;
            case IARRAY_DATA_TYPE_INT64:
                if (padding) {
                    *i64out = 0;
                } else {
                    rparams->ufunc->finish(i64out, &user_data);
                }
                i64out++;
                break;
            case IARRAY_DATA_TYPE_INT32:
                if (padding) {
                    *i32out = 0;
                } else {
                    rparams->ufunc->finish(i32out, &user_data);
                }
                i32out++;
                break;
            case IARRAY_DATA_TYPE_INT16:
                if (padding) {
                    *i16out = 0;
                } else {
                    rparams->ufunc->finish(i16out, &user_data);
                }
                i16out++;
                break;
            case IARRAY_DATA_TYPE_INT8:
                if (padding) {
                    *i8out = 0;
                } else {
                    rparams->ufunc->finish(i8out, &user_data);
                }
                i8out++;
                break;
            case IARRAY_DATA_TYPE_UINT64:
                if (padding) {
                    *ui64out = 0;
                } else {
                    rparams->ufunc->finish(ui64out, &user_data);
                }
                ui64out++;
                break;
            case IARRAY_DATA_TYPE_UINT32:
                if (padding) {
                    *ui32out = 0;
                } else {
                    rparams->ufunc->finish(ui32out, &user_data);
                }
                ui32out++;
                break;
            case IARRAY_DATA_TYPE_UINT16:
                if (padding) {
                    *ui16out = 0;
                } else {
                    rparams->ufunc->finish(ui16out, &user_data);
                }
                ui16out++;
                break;
            case IARRAY_DATA_TYPE_UINT8:
                if (padding) {
                    *ui8out = 0;
                } else {
                    rparams->ufunc->finish(ui8out, &user_data);
                }
                ui8out++;
                break;
            case IARRAY_DATA_TYPE_BOOL:
                if (padding) {
                    *boolout = 0;
                } else {
                    rparams->ufunc->finish(boolout, &user_data);
                }
                boolout++;
                break;
            default:
                IARRAY_TRACE1(iarray.error, "Invalid dtype");
                return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
        }
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
                                     iarray_container_t **b, iarray_data_type_t res_dtype) {

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
    void *reduce_function = NULL;
    // res data type
    iarray_data_type_t dtype;
    switch (func) {
        case IARRAY_REDUCE_SUM:
            // If the input is of type integer or unsigned int the result will be of type int64_t or uint64_t respectively
            switch (a->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    reduce_function = &DSUM;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    reduce_function = &FSUM;
                    dtype = IARRAY_DATA_TYPE_FLOAT;
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    reduce_function = &I64SUM;
                    dtype = IARRAY_DATA_TYPE_INT64;
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    reduce_function = &I32SUM;
                    dtype = IARRAY_DATA_TYPE_INT64;
                    break;
                case IARRAY_DATA_TYPE_INT16:
                    reduce_function = &I16SUM;
                    dtype = IARRAY_DATA_TYPE_INT64;
                    break;
                case IARRAY_DATA_TYPE_INT8:
                    reduce_function = &I8SUM;
                    dtype = IARRAY_DATA_TYPE_INT64;
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    reduce_function = &UI64SUM;
                    dtype = IARRAY_DATA_TYPE_UINT64;
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    reduce_function = &UI32SUM;
                    dtype = IARRAY_DATA_TYPE_UINT64;
                    break;
                case IARRAY_DATA_TYPE_UINT16:
                    reduce_function = &UI16SUM;
                    dtype = IARRAY_DATA_TYPE_UINT64;
                    break;
                case IARRAY_DATA_TYPE_UINT8:
                    reduce_function = &UI8SUM;
                    dtype = IARRAY_DATA_TYPE_UINT64;
                    break;
                case IARRAY_DATA_TYPE_BOOL:
                    reduce_function = &BOOLSUM;
                    dtype = IARRAY_DATA_TYPE_INT64;
                    break;
            }
            break;
        case IARRAY_REDUCE_PROD:
            // If the input is of type integer or unsigned int the result will be of type int64_t or uint64_t respectively
            switch (a->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    reduce_function = &DPROD;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    reduce_function = &FPROD;
                    dtype = IARRAY_DATA_TYPE_FLOAT;
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    reduce_function = &I64PROD;
                    dtype = IARRAY_DATA_TYPE_INT64;
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    reduce_function = &I32PROD;
                    dtype = IARRAY_DATA_TYPE_INT64;
                    break;
                case IARRAY_DATA_TYPE_INT16:
                    reduce_function = &I16PROD;
                    dtype = IARRAY_DATA_TYPE_INT64;
                    break;
                case IARRAY_DATA_TYPE_INT8:
                    reduce_function = &I8PROD;
                    dtype = IARRAY_DATA_TYPE_INT64;
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    reduce_function = &UI64PROD;
                    dtype = IARRAY_DATA_TYPE_UINT64;
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    reduce_function = &UI32PROD;
                    dtype = IARRAY_DATA_TYPE_UINT64;
                    break;
                case IARRAY_DATA_TYPE_UINT16:
                    reduce_function = &UI16PROD;
                    dtype = IARRAY_DATA_TYPE_UINT64;
                    break;
                case IARRAY_DATA_TYPE_UINT8:
                    reduce_function = &UI8PROD;
                    dtype = IARRAY_DATA_TYPE_UINT64;
                    break;
                case IARRAY_DATA_TYPE_BOOL:
                    reduce_function = &BOOLPROD;
                    dtype = IARRAY_DATA_TYPE_INT64;
                    break;
            }
            break;
        case IARRAY_REDUCE_MAX:
            switch (a->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    reduce_function = &DMAX;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    reduce_function = &FMAX;
                    dtype = IARRAY_DATA_TYPE_FLOAT;
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    reduce_function = &I64MAX;
                    dtype = IARRAY_DATA_TYPE_INT64;
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    reduce_function = &I32MAX;
                    dtype = IARRAY_DATA_TYPE_INT32;
                    break;
                case IARRAY_DATA_TYPE_INT16:
                    reduce_function = &I16MAX;
                    dtype = IARRAY_DATA_TYPE_INT16;
                    break;
                case IARRAY_DATA_TYPE_INT8:
                    reduce_function = &I8MAX;
                    dtype = IARRAY_DATA_TYPE_INT8;
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    reduce_function = &UI64MAX;
                    dtype = IARRAY_DATA_TYPE_UINT64;
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    reduce_function = &UI32MAX;
                    dtype = IARRAY_DATA_TYPE_UINT32;
                    break;
                case IARRAY_DATA_TYPE_UINT16:
                    reduce_function = &UI16MAX;
                    dtype = IARRAY_DATA_TYPE_UINT16;
                    break;
                case IARRAY_DATA_TYPE_UINT8:
                    reduce_function = &UI8MAX;
                    dtype = IARRAY_DATA_TYPE_UINT8;
                    break;
                case IARRAY_DATA_TYPE_BOOL:
                    reduce_function = &BOOLMAX;
                    dtype = IARRAY_DATA_TYPE_BOOL;
                    break;
            }
            break;
        case IARRAY_REDUCE_MIN:
            switch (a->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    reduce_function = &DMIN;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    reduce_function = &FMIN;
                    dtype = IARRAY_DATA_TYPE_FLOAT;
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    reduce_function = &I64MIN;
                    dtype = IARRAY_DATA_TYPE_INT64;
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    reduce_function = &I32MIN;
                    dtype = IARRAY_DATA_TYPE_INT32;
                    break;
                case IARRAY_DATA_TYPE_INT16:
                    reduce_function = &I16MIN;
                    dtype = IARRAY_DATA_TYPE_INT16;
                    break;
                case IARRAY_DATA_TYPE_INT8:
                    reduce_function = &I8MIN;
                    dtype = IARRAY_DATA_TYPE_INT8;
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    reduce_function = &UI64MIN;
                    dtype = IARRAY_DATA_TYPE_UINT64;
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    reduce_function = &UI32MIN;
                    dtype = IARRAY_DATA_TYPE_UINT32;
                    break;
                case IARRAY_DATA_TYPE_UINT16:
                    reduce_function = &UI16MIN;
                    dtype = IARRAY_DATA_TYPE_UINT16;
                    break;
                case IARRAY_DATA_TYPE_UINT8:
                    reduce_function = &UI8MIN;
                    dtype = IARRAY_DATA_TYPE_UINT8;
                    break;
                case IARRAY_DATA_TYPE_BOOL:
                    reduce_function = &BOOLMIN;
                    dtype = IARRAY_DATA_TYPE_BOOL;
                    break;
            }
            break;
        case IARRAY_REDUCE_MEAN:
            // If the input is of type integer or unsigned int the result will be of type double
            switch (a->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    reduce_function = &DMEAN;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    reduce_function = &FMEAN;
                    dtype = IARRAY_DATA_TYPE_FLOAT;
                    break;
                case IARRAY_DATA_TYPE_INT64:
                    reduce_function = &I64MEAN;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_INT32:
                    reduce_function = &I32MEAN;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_INT16:
                    reduce_function = &I16MEAN;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_INT8:
                    reduce_function = &I8MEAN;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT64:
                    reduce_function = &UI64MEAN;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT32:
                    reduce_function = &UI32MEAN;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT16:
                    reduce_function = &UI16MEAN;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_UINT8:
                    reduce_function = &UI8MEAN;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
                case IARRAY_DATA_TYPE_BOOL:
                    reduce_function = &BOOLMEAN;
                    dtype = IARRAY_DATA_TYPE_DOUBLE;
                    break;
            }
            break;
    }

    IARRAY_RETURN_IF_FAILED(_iarray_reduce_udf(ctx, a, reduce_function, axis, storage, b, dtype));

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

    if (a->view) {
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
        iarray_copy(ctx, a, false, &view_storage, 0, &aa);
    }

    // Start reductions
    iarray_container_t *c = NULL;
    iarray_storage_t storage_red;
    storage_red.contiguous = storage->contiguous;
    storage_red.urlpath = storage->urlpath != NULL ? "_iarray_red.iarr" : NULL;
    if (storage_red.urlpath != NULL && access(storage_red.urlpath, 0) == 0) {
        IARRAY_TRACE1(iarray.tracing, "The temporary file %s already exists, delete it first", storage_red.urlpath);
        return INA_ERROR(INA_ERR_INVALID);
    }
    for (int i = 0; i < ii; ++i) {
        if (i > 0) {
            if (storage->urlpath != NULL) {
                if (i > 1) {
                    err_io = blosc2_remove_urlpath("_iarray_red_2.iarr");
                    if (err_io != 0) {
                        IARRAY_TRACE1(iarray.tracing, "Invalid io");
                        return INA_ERROR(INA_ERR_OPERATION_INVALID);
                    }
                }
                err_io = blosc2_rename_urlpath("_iarray_red.iarr", "_iarray_red_2.iarr");
                if (err_io != 0) {
                    IARRAY_TRACE1(iarray.tracing, "Invalid io");
                    return INA_ERROR(INA_ERR_OPERATION_INVALID);
                }
                IARRAY_RETURN_IF_FAILED(iarray_container_open(ctx, "_iarray_red_2.iarr", &aa));
            } else {
                aa = c;
            }
        }

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
            err_io = blosc2_remove_urlpath("_iarray_red.iarr");
            if (err_io != 0) {
                IARRAY_TRACE1(iarray.tracing, "Invalid io");
                return INA_ERROR(INA_ERR_OPERATION_INVALID);
            }
            if (ii > 1) {
                err_io = blosc2_remove_urlpath("_iarray_red_2.iarr");
                if (err_io != 0) {
                    IARRAY_TRACE1(iarray.tracing, "Invalid io");
                    return INA_ERROR(INA_ERR_OPERATION_INVALID);
                }
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
            if (ii > 1) {
                err_io = blosc2_remove_urlpath("_iarray_red_2.iarr");
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
    if (storage->urlpath != NULL) {
        blosc2_remove_urlpath("_iarray_red.iarr");
        blosc2_remove_urlpath("_iarray_red_2.iarr");
    }
    if (a->view && a->storage->urlpath != NULL) {
        blosc2_remove_urlpath("_iarray_view.iarr");
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