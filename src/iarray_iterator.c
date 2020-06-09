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

/*
 * Matmul iterator
 */


void _iarray_iter_matmul_init(iarray_iter_matmul_t *itr)
{
    itr->cont = 0;
    itr->npart1 = 0;
    itr->npart2 = 0;
}

void _iarray_iter_matmul_next(iarray_iter_matmul_t *itr)
{
    int64_t B0 = itr->B0;
    int64_t B1 = itr->B1;
    int64_t B2 = itr->B2;
    int64_t M = itr->M;
    int64_t N = itr->N;
    int64_t K = itr->K;

    itr->cont++;

    int64_t n, k, m;

    if (itr->container2->catarr->ndim == 1) {
        m = itr->cont / ((K/B1)) % (M/B0);
        k = itr->cont % (K/B1);

        itr->npart1 = (m * (K/B1) + k);
        itr->npart2 = k;

    } else {
        m = itr->cont / ((K/B1) * (N/B2)) % (M/B0);
        k = itr->cont % (K/B1);
        n = itr->cont / ((K/B1)) % (N/B2);

        itr->npart1 = (m * (K/B1) + k);
        itr->npart2 = (k * (N/B2) + n);
    }
}

int _iarray_iter_matmul_finished(iarray_iter_matmul_t *itr)
{
    int64_t B0 = itr->B0;
    int64_t B1 = itr->B1;
    int64_t B2 = itr->B2;
    int64_t M = itr->M;
    int64_t N = itr->N;
    int64_t K = itr->K;

    if (itr->container2->dtshape->ndim == 1) {
        return itr->cont >= (M/B0) * (K/B1);
    }

    return itr->cont >= (M/B0) * (N/B2) * (K/B1);
}

ina_rc_t _iarray_iter_matmul_new(iarray_context_t *ctx, iarray_container_t *c1, iarray_container_t *c2,
                                 int64_t *bshape_a, int64_t *bshape_b, iarray_iter_matmul_t **itr)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(c1);
    INA_VERIFY_NOT_NULL(c2);
    INA_VERIFY_NOT_NULL(bshape_a);
    INA_VERIFY_NOT_NULL(bshape_b);
    INA_VERIFY_NOT_NULL(itr);

    ina_rc_t rc;

    // Verify that block shape is < than container shapes
    for (int i = 0; i < c1->dtshape->ndim; ++i) {
        if (c1->dtshape->shape[i] < bshape_a[i]) {
            IARRAY_TRACE1(iarray.error, "The blockshape is larger than the container shape");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_BSHAPE));
        }
    }
    for (int i = 0; i < c2->dtshape->ndim; ++i) {
        if (c2->dtshape->shape[i] < bshape_b[i]) {
            IARRAY_TRACE1(iarray.error, "The blockshape is larger than the container shape");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_BSHAPE));
        }
    }

    *itr = (iarray_iter_matmul_t*)ina_mem_alloc(sizeof(iarray_iter_matmul_t));
    if (*itr == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the matmul iterator");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }

    (*itr)->ctx = ctx;
    (*itr)->container1 = c1;
    (*itr)->container2 = c2;
    (*itr)->B0 = bshape_a[0];
    (*itr)->B1 = bshape_a[1];
    (*itr)->B2 = bshape_b[1];

    // Calculate the ext shape from the block shape
    if (c1->dtshape->shape[0] % bshape_a[0] == 0) {
        (*itr)->M = c1->dtshape->shape[0];
    } else {
        (*itr)->M = (c1->dtshape->shape[0] / bshape_a[0] + 1) * bshape_a[0];
    }

    if (c1->dtshape->shape[1] % bshape_a[1] == 0) {
        (*itr)->K = c1->dtshape->shape[1];
    } else {
        (*itr)->K = (c1->dtshape->shape[1] / bshape_a[1] + 1) * bshape_a[1];
    }

    if (c2->dtshape->ndim == 2) {
        if (c2->dtshape->shape[1] % bshape_b[1] == 0) {
            (*itr)->N = c2->dtshape->shape[1];
        } else {
            (*itr)->N = (c2->dtshape->shape[1] / bshape_b[1] + 1) * bshape_b[1];
        }
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    _iarray_iter_matmul_free(itr);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}


void _iarray_iter_matmul_free(iarray_iter_matmul_t **itr)
{
    INA_VERIFY_FREE(itr);

    INA_MEM_FREE_SAFE(*itr);
}



/*
 * Block-wise read iterator
 */


INA_API(ina_rc_t) iarray_iter_read_block_next(iarray_iter_read_block_t *itr, void *buffer, int32_t bufsize)
{
    int64_t typesize = itr->cont->catarr->itemsize;

    // Check if a external buffer is passed
    if (itr->external_buffer) {
        if (bufsize < itr->block_shape_size * typesize + BLOSC_MAX_OVERHEAD) {
            IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
        }
        itr->block = buffer;
        itr->block_pointer = (void **) &itr->block;
    }

    int8_t ndim = itr->cont->dtshape->ndim;

    // Calculate the start of the desired block
    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t inc = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        start_[i] = itr->nblock % (itr->aux[i] * inc) / inc;
        itr->cur_block_index[i] = start_[i];
        start_[i] *= itr->block_shape[i];
        itr->cur_elem_index[i] = start_[i];
        inc *= itr->aux[i];
    }

    // Calculate the stop of the desired block
    int64_t stop_[IARRAY_DIMENSION_MAX];
    int64_t actual_block_size = 1;
    itr->cur_block_size = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if(start_[i] + itr->block_shape[i] <= itr->cont->dtshape->shape[i]) {
            stop_[i] = start_[i] + itr->block_shape[i];
        } else {
            stop_[i] = itr->cont->dtshape->shape[i];
        }
        itr->cur_block_shape[i] = stop_[i] - start_[i];
        itr->cur_block_size *= itr->cur_block_shape[i];
        actual_block_size *= itr->block_shape[i];
    }

    // Get the desired block
    if (itr->contiguous && (itr->cont->view == false)) {
        IARRAY_FAIL_IF_ERROR(_iarray_get_slice_buffer_no_copy(itr->ctx, itr->cont, (int64_t *) start_,
                                                              (int64_t *) stop_, (void **) &itr->block,
                                                               actual_block_size * typesize));
    } else {
        IARRAY_FAIL_IF_ERROR(iarray_get_slice_buffer(itr->ctx, itr->cont, (int64_t *) start_,
                                                     (int64_t *) stop_, itr->block,
                                                     actual_block_size * typesize));
    }

    // Update the structure that user can see
    itr->val->block_pointer = *itr->block_pointer;
    itr->val->block_index = itr->cur_block_index;
    itr->val->elem_index = itr->cur_elem_index;
    itr->val->nblock = itr->nblock;
    itr->val->block_shape = itr->cur_block_shape;
    itr->val->block_size = itr->cur_block_size;

    // Increment the block counter
    itr->nblock += 1;

    return INA_SUCCESS;

    fail:
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_iter_read_block_has_next(iarray_iter_read_block_t *itr)
{
    if (itr->nblock < itr->total_blocks) {
        return INA_SUCCESS;
    }
    return INA_ERROR(IARRAY_ERR_END_ITER);
}


INA_API(ina_rc_t) iarray_iter_read_block_new(iarray_context_t *ctx,
                                             iarray_iter_read_block_t **itr,
                                             iarray_container_t *cont,
                                             const int64_t *blockshape,
                                             iarray_iter_read_block_value_t *value,
                                             bool external_buffer)
{

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(cont);
    INA_VERIFY_NOT_NULL(value);

    ina_rc_t rc;

    if (!cont->catarr->filled) {
        IARRAY_TRACE1(iarray.error, "The container is filled");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
    }

    if (blockshape == NULL) {
        IARRAY_TRACE1(iarray.error, "The blockshape can not be NULL");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
    }

    INA_VERIFY_NOT_NULL(itr);
    *itr = (iarray_iter_read_block_t *) ina_mem_alloc(sizeof(iarray_iter_read_block_t));
    if (*itr == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating iterator");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }
    memcpy(*itr, &IARRAY_ITER_READ_BLOCK_EMPTY, sizeof(iarray_iter_read_block_t));

    (*itr)->ctx = ctx;

    (*itr)->cont = cont;
    int64_t typesize = (*itr)->cont->catarr->itemsize;

    (*itr)->val = value;
    (*itr)->aux = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));
    (*itr)->block_shape = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));
    (*itr)->cur_block_shape = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));
    (*itr)->cur_block_index = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));
    (*itr)->cur_elem_index = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));

    // Create a buffer where data is stored to pass it to the user
    (*itr)->block_shape_size = 1;
    for (int i = 0; i < cont->dtshape->ndim; ++i) {
        (*itr)->block_shape[i] = blockshape[i];
        (*itr)->block_shape_size *= (*itr)->block_shape[i];
    }
    int64_t block_size = typesize * (*itr)->block_shape_size;


    // Check if is blocks are contigous in memory
    (*itr)->contiguous = (cont->catarr->storage == CATERVA_STORAGE_BLOSC) ? false: true;
    (*itr)->contiguous = !(cont->view) && (*itr)->contiguous;

    if ((*itr)->contiguous) {
        bool before_is_one = true;
        for (int i = 0; i < cont->dtshape->ndim; ++i) {
            if (blockshape[i] != cont->dtshape->shape[i] && !before_is_one) {
                (*itr)->contiguous = false;
                break;
            }
            before_is_one = (blockshape[i] == 1)? true: false;
        }
    }

    // Check if to alloc a block is needed
    if (!(*itr)->contiguous) {
        if (!external_buffer) {
            (*itr)->external_buffer = false;
            (*itr)->block = (uint8_t *) ina_mem_alloc((size_t) block_size + BLOSC_MAX_OVERHEAD);
            (*itr)->block_pointer = (void **) &(*itr)->block;
        } else {
            (*itr)->external_buffer = true;
            (*itr)->block = NULL;
        }
    } else {
        (*itr)->external_buffer = false;
        (*itr)->block = cont->catarr->buf;
        (*itr)->block_pointer = (void **) &(*itr)->block;
    }

    // Calculate the total number of blocks
    (*itr)->total_blocks = 1;
    for (int i = 0; i < cont->dtshape->ndim; ++i) {
        if(cont->dtshape->shape[i] % (*itr)->block_shape[i] == 0) {
            (*itr)->total_blocks *= cont->dtshape->shape[i] / (*itr)->block_shape[i];
        } else {
            (*itr)->total_blocks *= cont->dtshape->shape[i] / (*itr)->block_shape[i] + 1;
        }
    }

    // Calculate aux param
    for (int i = cont->dtshape->ndim - 1; i >= 0; --i) {
        if (cont->dtshape->shape[i] % (*itr)->block_shape[i] == 0) {
            (*itr)->aux[i] = cont->dtshape->shape[i] / (*itr)->block_shape[i];
        } else {
            (*itr)->aux[i] = cont->dtshape->shape[i] / (*itr)->block_shape[i] + 1;
        }
    }
    // Set params to 0
    for (int i = 0; i <IARRAY_DIMENSION_MAX; ++i) {
        (*itr)->cur_elem_index[i] = 0;
        (*itr)->cur_block_index[i] = 0;
    }

    if (cont->catarr->storage == CATERVA_STORAGE_BLOSC) {
        switch (cont->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                cont->catarr->part_cache.data =
                    ina_mempool_dalloc(ctx->mp_part_cache, (size_t) cont->catarr->chunksize * sizeof(double));
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                cont->catarr->part_cache.data =
                    ina_mempool_dalloc(ctx->mp_part_cache, (size_t) cont->catarr->chunksize * sizeof(float));
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }
    }
    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_iter_read_block_free(itr);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}


INA_API(void) iarray_iter_read_block_free(iarray_iter_read_block_t **itr)
{
    INA_VERIFY_FREE(itr);

    if (!(*itr)->contiguous && !(*itr)->external_buffer) {
        INA_MEM_FREE_SAFE((*itr)->block);
    }

    // Invalidate caches and get rid of memory pool
    (*itr)->cont->catarr->part_cache.data = NULL;
    (*itr)->cont->catarr->part_cache.nchunk = -1;
    ina_mempool_reset((*itr)->ctx->mp_part_cache);

    INA_MEM_FREE_SAFE((*itr)->aux);
    INA_MEM_FREE_SAFE((*itr)->block_shape);
    INA_MEM_FREE_SAFE((*itr)->cur_block_shape);
    INA_MEM_FREE_SAFE((*itr)->cur_block_index);
    INA_MEM_FREE_SAFE((*itr)->cur_elem_index);

    INA_MEM_FREE_SAFE((*itr));
}


/*
 * Block-wise write iterator
 */
INA_API(ina_rc_t) iarray_iter_write_block_next(iarray_iter_write_block_t *itr,
                                               void *buffer,
                                               int32_t bufsize) {

    caterva_array_t *catarr = itr->cont->catarr;
    int8_t ndim = catarr->ndim;
    int64_t typesize = itr->cont->catarr->itemsize;

    // Check if block is the first
    if (itr->nblock != 0) {
        if (itr->cont->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
            if (itr->contiguous) {
                int64_t dir = itr->nblock * itr->cur_block_size * typesize;
                itr->block = &itr->cont->catarr->buf[dir];
            } else {
                int64_t *start =itr->cur_elem_index;
                int64_t stop[IARRAY_DIMENSION_MAX];
                for (int i = 0; i < ndim; ++i) {
                    stop[i] = start[i] + itr->cur_block_shape[i];
                }
                int64_t blocksize = typesize;
                for (int i = 0; i < catarr->ndim; ++i) {
                    blocksize *= itr->block_shape[i];
                }

                caterva_config_t cfg = {0};
                iarray_create_caterva_cfg(itr->ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
                caterva_context_t *cat_ctx;
                IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

                IARRAY_ERR_CATERVA(caterva_array_set_slice_buffer(cat_ctx, itr->block, blocksize, start, stop, catarr));

                IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));
                if (itr->external_buffer) {
                    free(itr->block);
                }
            }
        } else {
            if (itr->compressed_chunk_buffer) {
                int err = blosc2_schunk_append_chunk(catarr->sc, itr->block, false);
                if (err < 0) {
                    IARRAY_TRACE1(iarray.error, "Error appending a chunk in a blosc schunk");
                    IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
                }
            } else {
                caterva_array_append(itr->cat_ctx, catarr, itr->block, itr->cur_block_size * typesize);
                if (itr->external_buffer) {
                    free(itr->block);
                }
            }

        }
    }

    // Check if a external buffer is needed
    if (itr->external_buffer) {
        if (bufsize < itr->block_shape_size * typesize + BLOSC_MAX_OVERHEAD) {
            IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
        }
        itr->block = buffer;
        itr->block_pointer = (void **) &itr->block;
    }

    // Update index
    itr->cur_block_index[ndim - 1] = itr->nblock % (itr->cont_eshape[ndim - 1] / itr->block_shape[ndim - 1]);
    itr->cur_elem_index[ndim - 1] = itr->cur_block_index[ndim - 1] * itr->block_shape[ndim - 1];

    int64_t inc = itr->cont_eshape[ndim - 1] / itr->block_shape[ndim - 1];

    for (int i = ndim - 2; i >= 0; --i) {
        itr->cur_block_index[i] = itr->nblock % (inc * itr->cont_eshape[i] / itr->block_shape[i]) / (inc);
        itr->cur_elem_index[i] = itr->cur_block_index[i] * itr->block_shape[i];
        inc *= itr->cont_eshape[i] / itr->block_shape[i];
    }

    // calculate the buffer size
    itr->cur_block_size = 1;
    for (int i = 0; i < ndim; ++i) {
        if ((itr->cur_block_index[i] + 1) * itr->block_shape[i] > catarr->shape[i]) {
            itr->cur_block_shape[i] = catarr->shape[i] - itr->cont_eshape[i] + itr->block_shape[i];
        } else {
            itr->cur_block_shape[i] = itr->block_shape[i];
        }
        itr->cur_block_size *= itr->cur_block_shape[i];
    }

    itr->val->block_pointer = *itr->block_pointer;
    itr->val->block_index = itr->cur_block_index;
    itr->val->elem_index = itr->cur_elem_index;
    itr->val->nblock = itr->nblock;
    itr->val->block_shape = itr->cur_block_shape;
    itr->val->block_size = itr->cur_block_size;

    itr->nblock += 1;

    return INA_SUCCESS;

    fail:
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_iter_write_block_has_next(iarray_iter_write_block_t *itr)
{
    if ( itr->nblock == (itr->cont_esize / itr->block_shape_size)) {  // TODO: cannot it be itr->total_blocks ?
        caterva_array_t *catarr = itr->cont->catarr;
        int8_t ndim = catarr->ndim;
        int64_t typesize = itr->cont->catarr->itemsize;
        if (itr->cont->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
            if (!itr->contiguous) {
                int64_t *start = itr->cur_elem_index;

                int64_t stop[IARRAY_DIMENSION_MAX];
                for (int i = 0; i < ndim; ++i) {
                    stop[i] = start[i] + itr->cur_block_shape[i];
                }
                int64_t blocksize = typesize;
                for (int i = 0; i < catarr->ndim; ++i) {
                    blocksize *= itr->block_shape[i];
                }

                caterva_config_t cfg = {0};
                iarray_create_caterva_cfg(itr->ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
                caterva_context_t *cat_ctx;
                IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));
                IARRAY_ERR_CATERVA(caterva_array_set_slice_buffer(cat_ctx, itr->block, blocksize, start, stop, catarr));
                IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));

                if (itr->external_buffer) {
                    free(itr->block);
                }
            }
        } else {
            // check if the part should be padded with 0s
            if (itr->compressed_chunk_buffer) {
                int err = blosc2_schunk_append_chunk(catarr->sc, itr->block, false);
                if (err < 0) {
                    // TODO: if the next call is not zero, it can be interpreted as there are more elements
                    IARRAY_TRACE1(iarray.error, "Error appending a chunk to a blosc schunk");
                    IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
                }
            } else {
                caterva_array_append(itr->cat_ctx, catarr, itr->block, itr->cur_block_size * typesize);
                if (itr->external_buffer) {
                    free(itr->block);
                }
            }
        }
    }

    if (itr->nblock == itr->total_blocks) {
        itr->cont->catarr->filled = true;
    }
    if(itr->nblock < itr->total_blocks) {
        return INA_SUCCESS;
    }
    return INA_ERROR(IARRAY_ERR_END_ITER);
    fail:
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_iter_write_block_new(iarray_context_t *ctx,
                                              iarray_iter_write_block_t **itr,
                                              iarray_container_t *cont,
                                              const int64_t *blockshape,
                                              iarray_iter_write_block_value_t *value,
                                              bool external_buffer)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(cont);
    INA_VERIFY_NOT_NULL(value);

    ina_rc_t rc;

    if (!cont->catarr->empty && cont->catarr->storage == CATERVA_STORAGE_BLOSC) {
        IARRAY_TRACE1(iarray.error, "The container can not be full");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_FULL_CONTAINER)); //TODO: Should we allow a rewrite a non-empty iarray cont
    }

    if (blockshape == NULL) {
        IARRAY_TRACE1(iarray.error, "The blockshape can not be NULL");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
    }

    if (cont->catarr->storage == CATERVA_STORAGE_BLOSC) {
        for (int i = 0; i < cont->dtshape->ndim; ++i) {
            if (blockshape[i] != cont->storage->pshape[i]) {
                IARRAY_TRACE1(iarray.error, "The blockshape must be equal to the container pshape");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_BSHAPE));
            }
        }
    }

    cont->catarr->empty = false;

    INA_VERIFY_NOT_NULL(itr);
    *itr = (iarray_iter_write_block_t *)ina_mem_alloc(sizeof(iarray_iter_write_block_t));
    if (*itr == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iterator");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }

    memcpy(*itr, &IARRAY_ITER_WRITE_BLOCK_EMPTY, sizeof(iarray_iter_write_block_t));

    int64_t typesize = cont->catarr->itemsize;

    caterva_config_t cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
    cfg.prefilter = ctx->prefilter_fn;
    cfg.pparams = ctx->prefilter_params;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &(*itr)->cat_ctx));

    if (cont->catarr->storage == CATERVA_STORAGE_PLAINBUFFER && !cont->catarr->empty) {
        memset(cont->catarr->buf, 0, cont->catarr->size * typesize);
        if (cont->catarr->buf == NULL) {
            IARRAY_TRACE1(iarray.error, "Error allocating the caterva buffer where data is stored");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
        }
    }

    (*itr)->compressed_chunk_buffer = false;  // the default is to pass uncompressed buffers
    (*itr)->val = value;
    (*itr)->ctx = ctx;
    (*itr)->cont = cont;
    (*itr)->cur_block_index = (int64_t *) ina_mem_alloc(CATERVA_MAX_DIM * sizeof(int64_t));
    (*itr)->cur_elem_index = (int64_t *) ina_mem_alloc(CATERVA_MAX_DIM * sizeof(int64_t));
    (*itr)->cur_block_shape = (int64_t *) ina_mem_alloc(CATERVA_MAX_DIM * sizeof(int64_t));
    (*itr)->block_shape = (int64_t *) ina_mem_alloc(CATERVA_MAX_DIM * sizeof(int64_t));
    (*itr)->cont_eshape = (int64_t *) ina_mem_alloc(CATERVA_MAX_DIM * sizeof(int64_t));

    (*itr)->cont_esize = 1;
    (*itr)->block_shape_size = 1;
    int64_t size = typesize;
    for (int i = 0; i < (*itr)->cont->dtshape->ndim; ++i) {
        (*itr)->block_shape[i] = blockshape[i];
        size *= (*itr)->block_shape[i];
        if (cont->catarr->extshape[i] % blockshape[i] == 0) {
            (*itr)->cont_eshape[i] = (cont->catarr->extshape[i] / blockshape[i]) * blockshape[i];
        } else {
            (*itr)->cont_eshape[i] = (cont->catarr->extshape[i] / blockshape[i] + 1) * blockshape[i];

        }
        (*itr)->cont_esize *= (*itr)->cont_eshape[i];
        (*itr)->block_shape_size *= (*itr)->block_shape[i];
    }

    int64_t block_size = typesize;
    for (int i = 0; i < cont->dtshape->ndim; ++i) {
        (*itr)->block_shape[i] = blockshape[i];
        block_size *= (*itr)->block_shape[i];
    }

    (*itr)->contiguous = (cont->catarr->storage == CATERVA_STORAGE_BLOSC) ? false: true;

    if ((*itr)->contiguous) {
        bool before_is_one = true;
        for (int i = 0; i < cont->dtshape->ndim; ++i) {
            if (blockshape[i] != cont->dtshape->shape[i] && !before_is_one) {
                (*itr)->contiguous = false;
                break;
            }
            before_is_one = (blockshape[i] == 1)? true: false;
        }
    }

    if (!(*itr)->contiguous) {
        if (!external_buffer) {
            // We may want to use the output partition for hosting a compressed buffer, so we need space for the overhead.
            // TODO: the overhead is only useful for the prefilter approach, so think if there is a better option.
            (*itr)->external_buffer = false;
            (*itr)->block = (uint8_t *) ina_mem_alloc((size_t) block_size + BLOSC_MAX_OVERHEAD);
            (*itr)->block_pointer = (void **) &(*itr)->block;
        } else {
            (*itr)->external_buffer = true;
            (*itr)->block = NULL;
        }
    } else {
        (*itr)->external_buffer = false;
        (*itr)->block = cont->catarr->buf;
        (*itr)->block_pointer = (void **) &(*itr)->block;
    }

    int8_t ndim = (*itr)->cont->dtshape->ndim;
    caterva_array_t *catarr = (*itr)->cont->catarr;

    (*itr)->nblock = 0;
    for (int i = 0; i < CATERVA_MAX_DIM; ++i) {
        (*itr)->cur_block_index[i] = 0;
        (*itr)->cur_block_shape[i] = (*itr)->block_shape[i];
    }
    (*itr)->cur_block_size = (*itr)->block_shape_size;

    //update_index
    (*itr)->cur_block_index[ndim - 1] = (*itr)->nblock % ((*itr)->cont_eshape[ndim - 1] / (*itr)->block_shape[ndim - 1]);
    (*itr)->cur_elem_index[ndim - 1] = (*itr)->cur_block_index[ndim - 1] * (*itr)->block_shape[ndim - 1];

    int64_t inc = (*itr)->cont_eshape[ndim - 1] / (*itr)->block_shape[ndim - 1];

    for (int i = ndim - 2; i >= 0; --i) {
        (*itr)->cur_block_index[i] = (*itr)->nblock % (inc * (*itr)->cont_eshape[i] / (*itr)->block_shape[i]) / (inc);
        (*itr)->cur_elem_index[i] = (*itr)->cur_block_index[i] * (*itr)->block_shape[i];
        inc *= (*itr)->cont_eshape[i] / (*itr)->block_shape[i];
    }

    //calculate the buffer size
    (*itr)->cur_block_size = 1;
    for (int i = 0; i < ndim; ++i) {
        if (((*itr)->cur_block_index[i] + 1) * (*itr)->block_shape[i] > catarr->shape[i]) {
            (*itr)->cur_block_shape[i] = catarr->shape[i] - (*itr)->cont_eshape[i] + (*itr)->block_shape[i];
        } else {
            (*itr)->cur_block_shape[i] = (*itr)->block_shape[i];
        }
        (*itr)->cur_block_size *= (*itr)->cur_block_shape[i];
    }

    (*itr)->total_blocks = (*itr)->cont_esize / (*itr)->block_shape_size; // Total number of blocks

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_iter_write_block_free(itr);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}


INA_API(void) iarray_iter_write_block_free(iarray_iter_write_block_t **itr)
{
    INA_VERIFY_FREE(itr);

    if (!(*itr)->contiguous && !(*itr)->external_buffer) {
        INA_MEM_FREE_SAFE((*itr)->block);
    }
    INA_MEM_FREE_SAFE((*itr)->block_shape);
    INA_MEM_FREE_SAFE((*itr)->cur_block_shape);
    INA_MEM_FREE_SAFE((*itr)->cur_block_index);
    INA_MEM_FREE_SAFE((*itr)->cur_elem_index);
    INA_MEM_FREE_SAFE((*itr)->cont_eshape);

    INA_MEM_FREE_SAFE(*itr);
}


/*
 * Element-wise read iterator
 */

INA_API(ina_rc_t) iarray_iter_read_next(iarray_iter_read_t *itr)
{
    int ndim = itr->cont->dtshape->ndim;

    int64_t typesize = itr->cont->catarr->itemsize;

    // check if a block is readed totally and decompress next
    if ((itr->nelem_block == itr->cur_block_size - 1) || (itr->nelem == 0)){

        // Calculate aux variables
        int64_t aux[IARRAY_DIMENSION_MAX];
        for (int i = ndim - 1; i >= 0; --i) {
            if (itr->cont->dtshape->shape[i] % itr->block_shape[i] == 0) {
                aux[i] = itr->cont->dtshape->shape[i] / itr->block_shape[i];
            } else {
                aux[i] = itr->cont->dtshape->shape[i] / itr->block_shape[i] + 1;
            }
        }

        // Calculate the start of the next block
        int64_t start_[IARRAY_DIMENSION_MAX];

        int64_t inc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            start_[i] = itr->nblock % (aux[i] * inc) / inc;
            itr->cur_block_index[i] = start_[i];
            start_[i] *= itr->block_shape[i];
            inc *= aux[i];
        }

        // Calculate the stop of the next block
        int64_t stop_[IARRAY_DIMENSION_MAX];
        int64_t buflen = 1;
        itr->cur_block_size = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            if (start_[i] + itr->block_shape[i] <= itr->cont->dtshape->shape[i]) {
                stop_[i] = start_[i] + itr->block_shape[i];
            } else {
                stop_[i] = itr->cont->dtshape->shape[i];
            }
            itr->cur_block_shape[i] = stop_[i] - start_[i];
            itr->cur_block_size *= itr->cur_block_shape[i];
            buflen *= itr->block_shape[i];
        }

        // Decompress the next block
        if (itr->cont->catarr->storage == CATERVA_STORAGE_PLAINBUFFER && itr->cont->view == false) {
            IARRAY_FAIL_IF_ERROR(_iarray_get_slice_buffer_no_copy(itr->ctx,
                                                                  itr->cont,
                                                                  (int64_t *) start_,
                                                                  (int64_t *) stop_,
                                                                  (void **) &itr->part,
                                                                  buflen * typesize));
        } else {
            IARRAY_FAIL_IF_ERROR(iarray_get_slice_buffer(itr->ctx,
                                                         itr->cont,
                                                         (int64_t *) start_,
                                                         (int64_t *) stop_,
                                                         itr->part,
                                                         buflen * typesize));
        }

        itr->nelem_block = 0;

        // Update block counter
        itr->nblock += 1;
    } else {
        itr->nelem_block += 1;
    }

    int64_t *c_shape = itr->cont->dtshape->shape;
    int64_t ind_part_elem[IARRAY_DIMENSION_MAX];
    int64_t inc = 1;
    int64_t inc_s = 1;

    itr->elem_flat_index = 0;
    for (int i = ndim - 1; i >= 0; --i) {
        ind_part_elem[i] = itr->nelem_block % (inc * itr->cur_block_shape[i]) / inc;
        itr->elem_index[i] = ind_part_elem[i] + itr->cur_block_index[i] * itr->block_shape[i];
        itr->elem_flat_index += itr->elem_index[i] * inc_s;
        inc_s *= c_shape[i];
        inc *= itr->cur_block_shape[i];
    }
    itr->pointer = (void *)&(itr->part)[itr->nelem_block * typesize];

    itr->val->elem_pointer = itr->pointer;
    itr->val->elem_index = itr->elem_index;
    itr->val->elem_flat_index = itr->elem_flat_index;

    itr->nelem += 1;

    return INA_SUCCESS;

    fail:
    return ina_err_get_rc();
}

/*
 * Function: iarray_iter_read_finished
 */

INA_API(ina_rc_t) iarray_iter_read_has_next(iarray_iter_read_t *itr)
{
    if (itr->nelem < itr->cont_size) {
        return INA_SUCCESS;
    }
    return INA_ERROR(IARRAY_ERR_END_ITER);
}


INA_API(ina_rc_t) iarray_iter_read_new(iarray_context_t *ctx,
                                       iarray_iter_read_t **itr,
                                       iarray_container_t *cont,
                                       iarray_iter_read_value_t *val)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(cont);
    INA_VERIFY_NOT_NULL(itr);
    INA_VERIFY_NOT_NULL(val);

    ina_rc_t rc;

    if (cont->catarr->filled != true) {
        IARRAY_TRACE1(iarray.error, "The container must be filled");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_EMPTY_CONTAINER));
    }

    *itr = (iarray_iter_read_t*)ina_mem_alloc(sizeof(iarray_iter_read_t));
    if (*itr == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iterator");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }
    memcpy(*itr, &IARRAY_ITER_READ_EMPTY, sizeof(iarray_iter_read_t));

    (*itr)->ctx = ctx;
    (*itr)->cont = cont;

    (*itr)->elem_index = (int64_t *) ina_mem_alloc(CATERVA_MAX_DIM * sizeof(int64_t));
    (*itr)->block_shape = (int64_t *) ina_mem_alloc(CATERVA_MAX_DIM * sizeof(int64_t));
    (*itr)->cur_block_shape = (int64_t *) ina_mem_alloc(CATERVA_MAX_DIM * sizeof(int64_t));
    (*itr)->cur_block_index = (int64_t *) ina_mem_alloc(CATERVA_MAX_DIM * sizeof(int64_t));

    int64_t block_size = 1;
    for (int i = 0; i < cont->dtshape->ndim; ++i) {
        (*itr)->block_shape[i] = cont->storage->pshape[i];
        block_size *= (*itr)->block_shape[i];
    }

    if (cont->catarr->storage == CATERVA_STORAGE_BLOSC || cont->view) {
        (*itr)->part = (uint8_t *) ina_mem_alloc((size_t) block_size * cont->catarr->itemsize);
    }

    (*itr)->val = val;

    // Initialize element and block index
    for (int i = 0; i <IARRAY_DIMENSION_MAX; ++i) {
        (*itr)->cur_block_index[i] = 0;
    }

    // Initialize block_ params

    (*itr)->cont_size = 1;
    for (int i = 0; i < (*itr)->cont->dtshape->ndim; ++i) {
        (*itr)->cont_size *= (*itr)->cont->dtshape->shape[i];
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_iter_read_free(itr);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}

/*
 * Function: iarray_iter_read_free
 */

INA_API(void) iarray_iter_read_free(iarray_iter_read_t **itr)
{
    INA_VERIFY_FREE(itr);

    INA_MEM_FREE_SAFE((*itr)->elem_index);
    if ((*itr)->cont->catarr->storage != CATERVA_STORAGE_PLAINBUFFER || (*itr)->cont->view) {
        INA_MEM_FREE_SAFE((*itr)->part);
    }
    INA_MEM_FREE_SAFE((*itr)->block_shape);
    INA_MEM_FREE_SAFE((*itr)->cur_block_shape);
    INA_MEM_FREE_SAFE((*itr)->cur_block_index);

    INA_MEM_FREE_SAFE(*itr);
}


/*
 * Element by element write iterator
 */


INA_API(ina_rc_t) iarray_iter_write_next(iarray_iter_write_t *itr)
{
    caterva_array_t *catarr = itr->container->catarr;
    int ndim = catarr->ndim;
    int64_t typesize = itr->container->catarr->itemsize;

    // check if a part is filled totally and append it
    if (itr->nelem_block == itr->cur_block_size - 1) {
        if (itr->container->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
            int err = caterva_array_append(itr->cat_ctx, itr->container->catarr, itr->part, itr->cur_block_size * typesize);
            if (err < 0) {
                IARRAY_TRACE1(iarray.error, "Error appending a buffer to a blosc schunk");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
            }

            int64_t inc = 1;
            itr->cur_block_size = 1;

            itr->nblock += 1;

            for (int i = ndim - 1; i >= 0; --i) {
                itr->cur_block_index[i] = itr->nblock % (inc * (catarr->extshape[i] / catarr->chunkshape[i])) / inc;
                inc *= (catarr->extshape[i] / catarr->chunkshape[i]);
                if ((itr->cur_block_index[i] + 1) * catarr->chunkshape[i] > catarr->shape[i]) {
                    itr->cur_block_shape[i] = catarr->shape[i] - itr->cur_block_index[i] * catarr->chunkshape[i];
                } else {
                    itr->cur_block_shape[i] = catarr->chunkshape[i];
                }
                itr->cur_block_size *= itr->cur_block_shape[i];
            }
            itr->nelem_block = 0;
        }
    } else if (itr->nelem != 0) {
        itr->nelem_block += 1;
    }

    // jump to the next element
    int64_t ind_part_elem[IARRAY_DIMENSION_MAX];
    int64_t cont_pointer = 0;

    int64_t inc = 1;
    int64_t inc_s = 1;
    int64_t inc_p = 1;

    itr->elem_flat_index = 0;

    for (int i = ndim - 1; i >= 0; --i) {
        ind_part_elem[i] = itr->nelem_block % (inc * itr->cur_block_shape[i]) / inc;
        cont_pointer += ind_part_elem[i] * inc_p;
        itr->elem_index[i] = ind_part_elem[i] + itr->cur_block_index[i] * catarr->chunkshape[i];
        itr->elem_flat_index += itr->elem_index[i] * inc_s;
        inc *= itr->cur_block_shape[i];
        inc_p *= itr->cur_block_shape[i];
        inc_s *= catarr->shape[i];
    }
    itr->pointer = (void *)&(itr->part)[cont_pointer * typesize];

    itr->val->elem_pointer = itr->pointer;
    itr->val->elem_index = itr->elem_index;
    itr->val->elem_flat_index = itr->elem_flat_index;

    itr->nelem += 1;

    return INA_SUCCESS;
    fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_iter_write_has_next(iarray_iter_write_t *itr)
{
    int64_t typesize = itr->container->catarr->itemsize;
    if (itr->nelem == itr->container->catarr->size) {
        if (itr->container->catarr->storage == CATERVA_STORAGE_BLOSC) {
            caterva_array_append(itr->cat_ctx, itr->container->catarr, itr->part, itr->cur_block_size * typesize);
        } else {
            itr->container->catarr->filled = true;
        }
    }

    if (itr->nelem < itr->container->catarr->size) {
        return INA_SUCCESS;
    }
    return INA_ERROR(IARRAY_ERR_END_ITER);
}


INA_API(ina_rc_t) iarray_iter_write_new(iarray_context_t *ctx,
                                        iarray_iter_write_t **itr,
                                        iarray_container_t *cont,
                                        iarray_iter_write_value_t *val)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(cont);
    INA_VERIFY_NOT_NULL(itr);
    INA_VERIFY_NOT_NULL(val);

    ina_rc_t rc;

    *itr = (iarray_iter_write_t*) ina_mem_alloc(sizeof(iarray_iter_write_t));
    if (*itr == NULL) {
        IARRAY_TRACE1(iarray.error, "Error allocating the iterator");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }
    memcpy(*itr, &IARRAY_ITER_WRITE_EMPTY, sizeof(iarray_iter_write_t));

    (*itr)->ctx = ctx;
    (*itr)->container = cont;
    cont->catarr->empty = false;

    caterva_config_t cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
    caterva_context_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

    if (cont->catarr->storage == CATERVA_STORAGE_PLAINBUFFER && !cont->catarr->empty) {
        (*itr)->part = cont->catarr->buf;
    } else {
        (*itr)->part = (uint8_t *) ina_mem_alloc((size_t)cont->catarr->chunksize * cont->catarr->itemsize);
    }
    IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));

    (*itr)->elem_index = (int64_t *) ina_mem_alloc(CATERVA_MAX_DIM * sizeof(int64_t));
    (*itr)->cur_block_index = (int64_t *) ina_mem_alloc(CATERVA_MAX_DIM * sizeof(int64_t));
    (*itr)->cur_block_shape = (int64_t *) ina_mem_alloc(CATERVA_MAX_DIM * sizeof(int64_t));

    (*itr)->val = val;

    (*itr)->cur_block_size = 1;

    for (int i = 0; i < CATERVA_MAX_DIM; ++i) {
        (*itr)->elem_index[i] = 0;
        (*itr)->cur_block_index[i] = 0;
        if ((*itr)->container->catarr->chunkshape[i] > (*itr)->container->catarr->shape[i]) {
            (*itr)->cur_block_shape[i] = (*itr)->container->catarr->shape[i];
        } else {
            (*itr)->cur_block_shape[i] = (*itr)->container->catarr->chunkshape[i];
        }
        (*itr)->cur_block_size *= (*itr)->cur_block_shape[i];
    }
    memset((*itr)->part, 0, cont->catarr->chunksize * cont->catarr->itemsize);

    caterva_config_t cat_cfg;
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cat_cfg);
    caterva_context_new(&cat_cfg, &(*itr)->cat_ctx);

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_iter_write_free(itr);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}


INA_API(void) iarray_iter_write_free(iarray_iter_write_t **itr)
{
    INA_VERIFY_FREE(itr);

    INA_MEM_FREE_SAFE((*itr)->elem_index);
    if ((*itr)->container->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
        INA_MEM_FREE_SAFE((*itr)->part);
    }

    INA_MEM_FREE_SAFE((*itr)->cur_block_index);
    INA_MEM_FREE_SAFE((*itr)->cur_block_shape);

    INA_MEM_FREE_SAFE(*itr);
}
