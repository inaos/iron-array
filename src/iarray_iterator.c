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

    // Verify that block shape is < than container shapes
    for (int i = 0; i < c1->dtshape->ndim; ++i) {
        if (c1->dtshape->shape[i] < bshape_a[i]) {
            return INA_ERROR(INA_ERR_FAILED);
        }
    }
    for (int i = 0; i < c2->dtshape->ndim; ++i) {
        if (c2->dtshape->shape[i] < bshape_b[i]) {
            return INA_ERROR(INA_ERR_FAILED);
        }
    }

    *itr = (iarray_iter_matmul_t*)ina_mem_alloc(sizeof(iarray_iter_matmul_t));
    INA_RETURN_IF_NULL(itr);
    (*itr)->ctx = ctx;
    (*itr)->container1 = c1;
    (*itr)->container2 = c2;
    (*itr)->B0 = bshape_a[0];
    (*itr)->B1 = bshape_a[1];
    (*itr)->B2 = bshape_b[1];

    // Calculate the extended shape from the block shape
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

    return INA_SUCCESS;
}

void _iarray_iter_matmul_free(iarray_iter_matmul_t *itr)
{
    ina_mem_free(itr);
}



/*
 * Block-wise read iterator
 */


INA_API(ina_rc_t) iarray_iter_read_block_next(iarray_iter_read_block_t *itr, void *buffer, int32_t bufsize)
{
    int64_t typesize = itr->cont->catarr->ctx->cparams.typesize;

    // Check if a external buffer is passed
    if (itr->external_buffer) {
        if (bufsize < itr->block_shape_size * typesize + BLOSC_MAX_OVERHEAD) {
            return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
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
        INA_MUST_SUCCEED(_iarray_get_slice_buffer_no_copy(itr->ctx, itr->cont, (int64_t *) start_,
                                                          (int64_t *) stop_, (void **) &itr->block,
                                                          actual_block_size * typesize));
    } else {
        INA_MUST_SUCCEED(iarray_get_slice_buffer(itr->ctx, itr->cont, (int64_t *) start_,
                                                 (int64_t *) stop_, itr->block,
                                                 actual_block_size * typesize));
    }

    // Update the structure that user can see
    itr->val->block_pointer = *itr->block_pointer;
    itr->val->block_index = itr->cur_block_index;
    itr->val->elem_index = itr->cur_elem_index;
    itr->val->nblock = itr->nblock;
    itr->val->block_shape = itr->cur_block_shape;
    itr->val->block_size = actual_block_size;

    // Increment the block counter
    itr->nblock += 1;

    return INA_SUCCESS;
}


INA_API(int) iarray_iter_read_block_has_next(iarray_iter_read_block_t *itr)
{
    return itr->nblock < itr->total_blocks;
}


INA_API(ina_rc_t) iarray_iter_read_block_new(iarray_context_t *ctx,
                                             iarray_iter_read_block_t **itr,
                                             iarray_container_t *cont,
                                             const int64_t *blockshape,
                                             iarray_iter_read_block_value_t *value,
                                             bool external_buffer)
{
    INA_VERIFY_NOT_NULL(itr);
    *itr = (iarray_iter_read_block_t *) ina_mem_alloc(sizeof(iarray_iter_read_block_t));
    INA_RETURN_IF_NULL(itr);

    INA_VERIFY_NOT_NULL(ctx);
    (*itr)->ctx = ctx;

    INA_VERIFY_NOT_NULL(cont);
    (*itr)->cont = cont;
    int64_t typesize = (*itr)->cont->catarr->ctx->cparams.typesize;

    if (blockshape == NULL) {
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }

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
    (*itr)->nblock = 0;

    if (cont->catarr->storage == CATERVA_STORAGE_BLOSC) {
        switch (cont->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                cont->catarr->part_cache.data =
                    ina_mempool_dalloc(ctx->mp, (size_t) cont->catarr->psize * sizeof(double));
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                cont->catarr->part_cache.data =
                    ina_mempool_dalloc(ctx->mp, (size_t) cont->catarr->psize * sizeof(float));
                break;
            default:break;
        }
    }
    return INA_SUCCESS;
}

INA_API(void) iarray_iter_read_block_free(iarray_iter_read_block_t *itr)
{
    if (!itr->contiguous && !itr->external_buffer) {
        ina_mem_free(itr->block);
    }

    itr->cont->catarr->part_cache.data = NULL;  // reset to NULL here (the memory pool will be reset later)
    itr->cont->catarr->part_cache.nchunk = -1;  // means no valid cache yet

    ina_mem_free(itr->aux);
    ina_mem_free(itr->block_shape);
    ina_mem_free(itr->cur_block_shape);
    ina_mem_free(itr->cur_block_index);
    ina_mem_free(itr->cur_elem_index);

    ina_mem_free(itr);
}


/*
 * Block-wise write iterator
 */
INA_API(ina_rc_t) iarray_iter_write_block_next(iarray_iter_write_block_t *itr,
                                               void *buffer,
                                               int32_t bufsize) {

    caterva_array_t *catarr = itr->cont->catarr;
    int8_t ndim = catarr->ndim;
    int64_t typesize = itr->cont->catarr->ctx->cparams.typesize;
    int64_t psizeb = itr->cur_block_size * typesize;

    // Check if block is the first
    if (itr->nblock != 0) {
        if (itr->cont->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
            if (itr->contiguous) {
                int64_t dir = itr->nblock * itr->cur_block_size * typesize;
                itr->block = &itr->cont->catarr->buf[dir];
            } else {
                caterva_dims_t start = caterva_new_dims(itr->cur_elem_index, ndim);

                int64_t stop_[IARRAY_DIMENSION_MAX];
                for (int i = 0; i < ndim; ++i) {
                    stop_[i] = start.dims[i] + itr->cur_block_shape[i];
                }
                caterva_dims_t stop = caterva_new_dims(stop_, ndim);

                caterva_set_slice_buffer(catarr, itr->block, &start, &stop);
            }
        } else {
            // check if the part should be padded with 0s
            if (itr->cur_block_size == catarr->psize) {
                if (itr->compressed_chunk_buffer) {
                    int err = blosc2_schunk_append_chunk(catarr->sc, itr->block, false);
                    if (err < 0) {
                        return INA_ERROR(INA_ERR_FAILED);
                    }
                } else {
                    int err = blosc2_schunk_append_buffer(catarr->sc, itr->block, (size_t) psizeb);
                    if (err < 0) {
                        return INA_ERROR(INA_ERR_FAILED);
                    }
                }
            } else {
                uint8_t *part_aux = malloc((size_t) catarr->psize * typesize);
                memset(part_aux, 0, catarr->psize * typesize);

                //reverse part_shape
                int64_t shaper[CATERVA_MAXDIM];
                for (int i = 0; i < CATERVA_MAXDIM; ++i) {
                    if (i >= CATERVA_MAXDIM - ndim) {
                        shaper[i] = itr->cur_block_shape[i - CATERVA_MAXDIM + ndim];
                    } else {
                        shaper[i] = 1;
                    }
                }

                //copy buffer data to an aux buffer padded with 0's
                int64_t ii[CATERVA_MAXDIM];
                for (ii[0] = 0; ii[0] < shaper[0]; ++ii[0]) {
                    for (ii[1] = 0; ii[1] < shaper[1]; ++ii[1]) {
                        for (ii[2] = 0; ii[2] < shaper[2]; ++ii[2]) {
                            for (ii[3] = 0; ii[3] < shaper[3]; ++ii[3]) {
                                for (ii[4] = 0; ii[4] < shaper[4]; ++ii[4]) {
                                    for (ii[5] = 0; ii[5] < shaper[5]; ++ii[5]) {
                                        for (ii[6] = 0; ii[6] < shaper[6]; ++ii[6]) {

                                            int64_t aux_p = 0;
                                            int64_t aux_i = catarr->pshape[ndim - 1];

                                            for (int i = ndim - 2; i >= 0; --i) {
                                                aux_p += ii[CATERVA_MAXDIM - ndim + i] * aux_i;
                                                aux_i *= catarr->pshape[i];
                                            }

                                            int64_t itr_p = 0;
                                            int64_t itr_i = shaper[CATERVA_MAXDIM - 1];

                                            for (int i = CATERVA_MAXDIM - 2; i >= CATERVA_MAXDIM - ndim; --i) {
                                                itr_p += ii[i] * itr_i;
                                                itr_i *= shaper[i];
                                            }
                                            memcpy(&part_aux[aux_p * typesize],
                                                   &(((uint8_t *) itr->block)[itr_p * typesize]),
                                                   shaper[7] * typesize);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                int err = blosc2_schunk_append_buffer(itr->cont->catarr->sc, part_aux,
                                                      (size_t) catarr->psize * typesize);
                if (err < 0) {
                    return INA_ERROR(INA_ERR_FAILED);
                }
                memset(part_aux, 0, catarr->psize * catarr->sc->typesize);

                free(part_aux);
            }
        }
    }

    // Ceck if a external buffer is needed
    if (itr->external_buffer) {
        if (bufsize < itr->block_shape_size * typesize + BLOSC_MAX_OVERHEAD) {
            return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
        }
        itr->block = buffer;
        itr->block_pointer = (void **) &itr->block;
    }

    //update_index
    itr->cur_block_index[ndim - 1] = itr->nblock % (itr->cont_eshape[ndim - 1] / itr->block_shape[ndim - 1]);
    itr->cur_elem_index[ndim - 1] = itr->cur_block_index[ndim - 1] * itr->block_shape[ndim - 1];

    int64_t inc = itr->cont_eshape[ndim - 1] / itr->block_shape[ndim - 1];

    for (int i = ndim - 2; i >= 0; --i) {
        itr->cur_block_index[i] = itr->nblock % (inc * itr->cont_eshape[i] / itr->block_shape[i]) / (inc);
        itr->cur_elem_index[i] = itr->cur_block_index[i] * itr->block_shape[i];
        inc *= itr->cont_eshape[i] / itr->block_shape[i];
    }

    //calculate the buffer size
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
}


INA_API(int) iarray_iter_write_block_has_next(iarray_iter_write_block_t *itr)
{
    if ( itr->nblock == (itr->cont_esize / itr->block_shape_size)) {
        caterva_array_t *catarr = itr->cont->catarr;
        int8_t ndim = catarr->ndim;
        int64_t typesize = itr->cont->catarr->ctx->cparams.typesize;
        int64_t psizeb = itr->cur_block_size * typesize;
        if (itr->cont->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
            if (!itr->contiguous) {
                caterva_dims_t start = caterva_new_dims(itr->cur_elem_index, ndim);

                int64_t stop_[IARRAY_DIMENSION_MAX];
                for (int i = 0; i < ndim; ++i) {
                    stop_[i] = start.dims[i] + itr->cur_block_shape[i];
                }
                caterva_dims_t stop = caterva_new_dims(stop_, ndim);

                caterva_set_slice_buffer(catarr, itr->block, &start, &stop);
            }
        } else {

            // check if the part should be padded with 0s
            if (itr->cur_block_size == catarr->psize) {
                if (itr->compressed_chunk_buffer) {
                    int err = blosc2_schunk_append_chunk(catarr->sc, itr->block, false);
                    if (err < 0) {
                        // TODO: if the next call is not zero, it can be interpreted as there are more elements
                        return INA_ERROR(INA_ERR_FAILED);
                    }
                } else {
                    int err = blosc2_schunk_append_buffer(catarr->sc, itr->block, (size_t) psizeb);
                    if (err < 0) {
                        // TODO: if the next call is not zero, it can be interpreted as there are more elements
                        return INA_ERROR(INA_ERR_FAILED);
                    }
                }
            } else {
                uint8_t *part_aux = malloc((size_t) catarr->psize * typesize);
                memset(part_aux, 0, catarr->psize * typesize);

                //reverse part_shape
                int64_t shaper[CATERVA_MAXDIM];
                for (int i = 0; i < CATERVA_MAXDIM; ++i) {
                    if (i >= CATERVA_MAXDIM - ndim) {
                        shaper[i] = itr->cur_block_shape[i - CATERVA_MAXDIM + ndim];
                    } else {
                        shaper[i] = 1;
                    }
                }

                //copy buffer data to an aux buffer padded with 0's
                int64_t ii[CATERVA_MAXDIM];
                for (ii[0] = 0; ii[0] < shaper[0]; ++ii[0]) {
                    for (ii[1] = 0; ii[1] < shaper[1]; ++ii[1]) {
                        for (ii[2] = 0; ii[2] < shaper[2]; ++ii[2]) {
                            for (ii[3] = 0; ii[3] < shaper[3]; ++ii[3]) {
                                for (ii[4] = 0; ii[4] < shaper[4]; ++ii[4]) {
                                    for (ii[5] = 0; ii[5] < shaper[5]; ++ii[5]) {
                                        for (ii[6] = 0; ii[6] < shaper[6]; ++ii[6]) {

                                            int64_t aux_p = 0;
                                            int64_t aux_i = catarr->pshape[ndim - 1];

                                            for (int i = ndim - 2; i >= 0; --i) {
                                                aux_p += ii[CATERVA_MAXDIM - ndim + i] * aux_i;
                                                aux_i *= catarr->pshape[i];
                                            }

                                            int64_t itr_p = 0;
                                            int64_t itr_i = shaper[CATERVA_MAXDIM - 1];

                                            for (int i = CATERVA_MAXDIM - 2; i >= CATERVA_MAXDIM - ndim; --i) {
                                                itr_p += ii[i] * itr_i;
                                                itr_i *= shaper[i];
                                            }
                                            memcpy(&part_aux[aux_p * typesize],
                                                   &(((uint8_t *) itr->block)[itr_p * typesize]),
                                                   shaper[7] * typesize);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                int err = blosc2_schunk_append_buffer(itr->cont->catarr->sc, part_aux,
                                                      (size_t) catarr->psize * typesize);
                if (err < 0) {
                    // TODO: if the next call is not zero, it can be interpreted as there are more elements
                    return INA_ERROR(INA_ERR_FAILED);
                }
                memset(part_aux, 0, catarr->psize * catarr->sc->typesize);

                free(part_aux);
            }
        }
    }

    return itr->nblock < itr->total_blocks;
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
    INA_VERIFY_NOT_NULL(itr);
    *itr = (iarray_iter_write_block_t *)ina_mem_alloc(sizeof(iarray_iter_write_block_t));
    INA_RETURN_IF_NULL(itr);

    if (!cont->catarr->empty && cont->catarr->storage == CATERVA_STORAGE_BLOSC) {
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT); //TODO: Should we allow a rewrite a non-empty iarray cont
    }

    if (blockshape == NULL) {
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }

    if (cont->catarr->storage == CATERVA_STORAGE_BLOSC) {
        for (int i = 0; i < cont->dtshape->ndim; ++i) {
            if (blockshape[i] != cont->dtshape->pshape[i]) {
                return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
            }
        }
    }

    int64_t typesize = cont->catarr->ctx->cparams.typesize;

    caterva_dims_t shape = caterva_new_dims(cont->dtshape->shape, cont->dtshape->ndim);
    int err = caterva_update_shape(cont->catarr, &shape);

    if (cont->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        cont->catarr->buf = cont->catarr->ctx->alloc((size_t) cont->catarr->size * typesize);
    }

    if (err < 0) {
        return INA_ERROR(INA_ERR_FAILED);
    }

    (*itr)->compressed_chunk_buffer = false;  // the default is to pass uncompressed buffers
    (*itr)->val = value;
    (*itr)->ctx = ctx;
    (*itr)->cont = cont;
    (*itr)->cur_block_index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->cur_elem_index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->cur_block_shape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->block_shape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->cont_eshape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));

    (*itr)->cont_esize = 1;
    (*itr)->block_shape_size = 1;
    int64_t size = typesize;
    for (int i = 0; i < (*itr)->cont->dtshape->ndim; ++i) {
        (*itr)->block_shape[i] = blockshape[i];
        size *= (*itr)->block_shape[i];
        if (cont->catarr->eshape[i] % blockshape[i] == 0) {
            (*itr)->cont_eshape[i] = (cont->catarr->eshape[i] / blockshape[i]) * blockshape[i];
        } else {
            (*itr)->cont_eshape[i] = (cont->catarr->eshape[i] / blockshape[i] + 1) * blockshape[i];

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
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
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

    return INA_SUCCESS;
}


INA_API(void) iarray_iter_write_block_free(iarray_iter_write_block_t *itr)
{
    if (!itr->contiguous && !itr->external_buffer) {
        ina_mem_free(itr->block);
    }
    ina_mem_free(itr->block_shape);
    ina_mem_free(itr->cur_block_shape);
    ina_mem_free(itr->cur_block_index);
    ina_mem_free(itr->cur_elem_index);
    ina_mem_free(itr->cont_eshape);

    ina_mem_free(itr);
}


/*
 * Element-wise read iterator
 */

INA_API(ina_rc_t) iarray_iter_read_next(iarray_iter_read_t *itr)
{
    caterva_array_t *catarr = itr->cont->catarr;
    int ndim = catarr->ndim;

    int64_t typesize = itr->cont->catarr->ctx->cparams.typesize;

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
            INA_MUST_SUCCEED(_iarray_get_slice_buffer_no_copy(itr->ctx, itr->cont, (int64_t *) start_,
                                                              (int64_t *) stop_, (void **) &itr->part,
                                                              buflen * typesize));
        } else {
            INA_MUST_SUCCEED(iarray_get_slice_buffer(itr->ctx, itr->cont, (int64_t *) start_,
                                                     (int64_t *) stop_, itr->part,
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
}

/*
 * Function: iarray_iter_read_finished
 */

INA_API(int) iarray_iter_read_has_next(iarray_iter_read_t *itr)
{
    return itr->nelem < itr->cont_size;
}


INA_API(ina_rc_t) iarray_iter_read_new(iarray_context_t *ctx,
                                       iarray_iter_read_t **itr,
                                       iarray_container_t *cont,
                                       iarray_iter_read_value_t *val)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(cont);
    INA_VERIFY_NOT_NULL(itr);

    *itr = (iarray_iter_read_t*)ina_mem_alloc(sizeof(iarray_iter_read_t));
    INA_RETURN_IF_NULL(itr);

    (*itr)->ctx = ctx;
    (*itr)->cont = cont;
    if (cont->catarr->storage == CATERVA_STORAGE_BLOSC) {
        (*itr)->part = (uint8_t *) ina_mem_alloc((size_t) cont->catarr->psize * cont->catarr->sc->typesize);
    }
    (*itr)->elem_index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));

    (*itr)->block_shape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));

    (*itr)->cur_block_shape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->cur_block_index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));

    for (int i = 0; i < cont->dtshape->ndim; ++i) {
        (*itr)->block_shape[i] = cont->dtshape->pshape[i];
    }

    (*itr)->val = val;

    // Initialize element and block index
    for (int i = 0; i <IARRAY_DIMENSION_MAX; ++i) {
        (*itr)->cur_block_index[i] = 0;
    }

    // Initialize counters
    (*itr)->nelem = 0;
    (*itr)->nblock = 0;
    (*itr)->nelem_block = 0;

    // Initialize block_ params

    (*itr)->cont_size = 1;
    for (int i = 0; i < (*itr)->cont->dtshape->ndim; ++i) {
        (*itr)->cont_size *= (*itr)->cont->dtshape->shape[i];
    }

    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_read_free
 */

INA_API(void) iarray_iter_read_free(iarray_iter_read_t *itr)
{
    ina_mem_free(itr->elem_index);
    if (itr->cont->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
        ina_mem_free(itr->part);
    }
    ina_mem_free(itr->block_shape);
    ina_mem_free(itr->cur_block_shape);
    ina_mem_free(itr->cur_block_index);
    ina_mem_free(itr);
}


/*
 * Element by element write iterator
 */


INA_API(ina_rc_t) iarray_iter_write_next(iarray_iter_write_t *itr)
{
    caterva_array_t *catarr = itr->container->catarr;
    int ndim = catarr->ndim;
    int64_t typesize = itr->container->catarr->ctx->cparams.typesize;
    // check if a part is filled totally and append it

    if (itr->nelem_block == itr->cur_block_size - 1) {
        if (itr->container->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
            int err = blosc2_schunk_append_buffer(catarr->sc, itr->part,
                                                  (size_t) catarr->psize * typesize);
            if (err < 0) {
                return INA_ERROR(INA_ERR_FAILED);
            }

            int64_t inc = 1;
            itr->cur_block_size = 1;

            itr->nblock += 1;

            for (int i = ndim - 1; i >= 0; --i) {
                itr->cur_block_index[i] = itr->nblock % (inc * (catarr->eshape[i] / catarr->pshape[i])) / inc;
                inc *= (catarr->eshape[i] / catarr->pshape[i]);
                if ((itr->cur_block_index[i] + 1) * catarr->pshape[i] > catarr->shape[i]) {
                    itr->cur_block_shape[i] = catarr->shape[i] - itr->cur_block_index[i] * catarr->pshape[i];
                } else {
                    itr->cur_block_shape[i] = catarr->pshape[i];
                }
                itr->cur_block_size *= itr->cur_block_shape[i];
            }
            memset(itr->part, 0, catarr->psize * typesize);
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
        itr->elem_index[i] = ind_part_elem[i] + itr->cur_block_index[i] * catarr->pshape[i];
        itr->elem_flat_index += itr->elem_index[i] * inc_s;
        inc *= itr->cur_block_shape[i];
        inc_p *= catarr->pshape[i];
        inc_s *= catarr->shape[i];
    }
    itr->pointer = (void *)&(itr->part)[cont_pointer * typesize];

    itr->val->elem_pointer = itr->pointer;
    itr->val->elem_index = itr->elem_index;
    itr->val->elem_flat_index = itr->elem_flat_index;

    itr->nelem += 1;

    return INA_SUCCESS;
}

INA_API(int) iarray_iter_write_has_next(iarray_iter_write_t *itr)
{
    int64_t typesize = itr->container->catarr->ctx->cparams.typesize;
    if (itr->nelem == itr->container->catarr->size) {
        if (itr->container->catarr->storage == CATERVA_STORAGE_BLOSC) {
            blosc2_schunk_append_buffer(itr->container->catarr->sc, itr->part,
                                        (size_t) itr->container->catarr->psize * typesize);
        }
    }
    return itr->nelem < itr->container->catarr->size;
}

INA_API(ina_rc_t) iarray_iter_write_new(iarray_context_t *ctx,
                                        iarray_iter_write_t **itr,
                                        iarray_container_t *cont,
                                        iarray_iter_write_value_t *val)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(cont);
    INA_VERIFY_NOT_NULL(itr);

    *itr = (iarray_iter_write_t*)ina_mem_alloc(sizeof(iarray_iter_write_t));
    INA_RETURN_IF_NULL(itr);
    caterva_dims_t shape = caterva_new_dims(cont->dtshape->shape, cont->dtshape->ndim);
    int err = caterva_update_shape(cont->catarr, &shape);
    if (err < 0) {
        return INA_ERROR(INA_ERR_FAILED);
    }
    (*itr)->ctx = ctx;
    (*itr)->container = cont;
    if (cont->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        (*itr)->part = (uint8_t *) cont->catarr->ctx->alloc((size_t)cont->catarr->psize *
            cont->catarr->ctx->cparams.typesize);
        cont->catarr->buf = (*itr)->part;
    } else {
        (*itr)->part = (uint8_t *) ina_mem_alloc((size_t)cont->catarr->psize * cont->catarr->ctx->cparams.typesize);
    }

    (*itr)->elem_index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->cur_block_index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->cur_block_shape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));

    (*itr)->val = val;

    (*itr)->nelem = 0;
    (*itr)->nblock = 0;
    (*itr)->nelem_block = 0;
    (*itr)->elem_flat_index = 0;

    (*itr)->cur_block_size = (*itr)->container->catarr->psize;

    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        (*itr)->elem_index[i] = 0;
        (*itr)->cur_block_index[i] = 0;
        (*itr)->cur_block_shape[i] = (*itr)->container->catarr->pshape[i];
    }

    memset((*itr)->part, 0, cont->catarr->psize * cont->catarr->ctx->cparams.typesize);

    return INA_SUCCESS;
}


INA_API(void) iarray_iter_write_free(iarray_iter_write_t *itr)
{
    ina_mem_free(itr->elem_index);
    if (itr->container->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
        ina_mem_free(itr->part);
    }
    ina_mem_free(itr->cur_block_index);
    ina_mem_free(itr->cur_block_shape);
    ina_mem_free(itr);
}
