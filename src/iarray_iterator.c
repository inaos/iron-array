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
 * Element by element iterator
 *
 * Next functions are used to fill an iarray container element by element
 */

/*
 * Function: iarray_iter_write_init
 * -------------------------
 *   Set the iterator values to the first element
 *
 *   itr: an iterator
 */

INA_API(void) iarray_iter_write_init(iarray_iter_write_t *itr)
{
    itr->cont = 0;
    itr->cont_part = 0;
    itr->cont_part_elem = 0;

    itr->nelem = 0;

    itr->bsize = itr->container->catarr->psize;

    memset(itr->part, 0, itr->container->catarr->psize * itr->container->catarr->ctx->cparams.typesize);
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        itr->index[i] = 0;
        itr->part_index[i] = 0;
        itr->bshape[i] = itr->container->catarr->pshape[i];
    }
    itr->pointer = &itr->part[0];
}

/*
 * Function: iarray_iter_write_next
 * -------------------------
 *   Compute the next iterator element nad update the iterator with it
 *
 *   itr: an iterator
 */

INA_API(ina_rc_t) iarray_iter_write_next(iarray_iter_write_t *itr)
{
    caterva_array_t *catarr = itr->container->catarr;
    int ndim = catarr->ndim;
    int64_t typesize = itr->container->catarr->ctx->cparams.typesize;
    // check if a part is filled totally and append it

    if (itr->cont_part_elem  == itr->bsize - 1) {
        if (itr->container->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
            itr->cont = itr->container->catarr->size;
        } else {

            int err = blosc2_schunk_append_buffer(catarr->sc, itr->part,
                                                  (size_t) catarr->psize * typesize);
            if (err < 0) {
                return INA_ERROR(INA_ERR_FAILED);
            }
            itr->cont_part_elem = 0;
            itr->cont_part += 1;
            int64_t inc = 1;
            itr->bsize = 1;

            for (int i = ndim - 1; i >= 0; --i) {
                itr->part_index[i] = itr->cont_part % (inc * (catarr->eshape[i] / catarr->pshape[i])) / inc;
                inc *= (catarr->eshape[i] / catarr->pshape[i]);
                if ((itr->part_index[i] + 1) * catarr->pshape[i] > catarr->shape[i]) {
                    itr->bshape[i] = catarr->shape[i] - itr->part_index[i] * catarr->pshape[i];
                } else {
                    itr->bshape[i] = catarr->pshape[i];
                }
                itr->bsize *= itr->bshape[i];
            }
            memset(itr->part, 0, catarr->psize * typesize);
        }
    } else {
        itr->cont_part_elem += 1;
    }

    // jump to the next element
    itr->cont += 1;

    int64_t ind_part_elem[IARRAY_DIMENSION_MAX];
    int64_t cont_pointer = 0;

    int64_t inc = 1;
    int64_t inc_s = 1;
    int64_t inc_p = 1;

    itr->nelem = 0;

    for (int i = ndim - 1; i >= 0; --i) {
        ind_part_elem[i] = itr->cont_part_elem % (inc * itr->bshape[i]) / inc;
        cont_pointer += ind_part_elem[i] * inc_p;
        itr->index[i] = ind_part_elem[i] + itr->part_index[i] * catarr->pshape[i];
        itr->nelem += itr->index[i] * inc_s;
        inc *= itr->bshape[i];
        inc_p *= catarr->pshape[i];
        inc_s *= catarr->shape[i];
    }
    itr->pointer = (void *)&(itr->part)[cont_pointer * typesize];

    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_write_finished
 * -----------------------------
 *   Check if the iteration over a container is finished
 *
 *   itr: an iterator
 *
 *   return: 1 if iter is finished or 0 if not
 */

INA_API(int) iarray_iter_write_finished(iarray_iter_write_t *itr)
{
    return itr->cont >= itr->container->catarr->size;
}

/*
 * Function: iarray_iter_write_value
 * ------------------------
 *   Store in `val` some variables of the actual element
 *
 *   itr: an iterator
 *   val: a struct where data needed by the user is stored
 *     part_index: position in coord where the element is located in the container
 *     nelem: if the container is row-wise flatten, `nelem` is the element position in the container
 *     pointer: pointer to element position in memory. It's used to copy the element into the container
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(void) iarray_iter_write_value(iarray_iter_write_t *itr, iarray_iter_write_value_t *val)
{
    val->pointer = itr->pointer;
    val->index = itr->index;
    val->nelem = itr->nelem;
}

/*
 * Function: iarray_iter_write_new
 * ------------------------
 *   Create a new iterator
 *
 *   ctx: iarrat context
 *   container: the container used in the iterator
 *   itr: an iterator pointer
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(ina_rc_t) iarray_iter_write_new(iarray_context_t *ctx, iarray_container_t *container, iarray_iter_write_t **itr)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(itr);

    *itr = (iarray_iter_write_t*)ina_mem_alloc(sizeof(iarray_iter_write_t));
    INA_RETURN_IF_NULL(itr);
    caterva_dims_t shape = caterva_new_dims(container->dtshape->shape, container->dtshape->ndim);
    int err = caterva_update_shape(container->catarr, &shape);
    if (err < 0) {
        return INA_ERROR(INA_ERR_FAILED);
    }
    (*itr)->ctx = ctx;
    (*itr)->container = container;
    if (container->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        (*itr)->part = (uint8_t *) container->catarr->ctx->alloc((size_t)container->catarr->psize *
            container->catarr->ctx->cparams.typesize);
        container->catarr->buf = (*itr)->part;
    } else {
        (*itr)->part = (uint8_t *) ina_mem_alloc((size_t)container->catarr->psize * container->catarr->ctx->cparams.typesize);
    }

    (*itr)->index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->part_index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->bshape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));


    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_write_free
 * -------------------------
 *   Free an iterator structure
 *
 *   itr: an iterator
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(void) iarray_iter_write_free(iarray_iter_write_t *itr)
{
    ina_mem_free(itr->index);
    if (itr->container->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
        ina_mem_free(itr->part);
    }
    ina_mem_free(itr->part_index);
    ina_mem_free(itr->bshape);
    ina_mem_free(itr);
}

/*
 * Partition by partition iterator
 *
 * Unlike the previous, the next collection of functions are used to fill an iarray container part by part
 */

/*
 * Function: iarray_iter_write_part_init
 * -------------------------------
 *   Set the iterator values to the first element
 *
 *   itr: an iterator
 */

INA_API(void) iarray_iter_write_part_init(iarray_iter_write_part_t *itr)
{
    int8_t ndim = itr->container->dtshape->ndim;
    caterva_array_t *catarr = itr->container->catarr;

    itr->cont = 0;
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        itr->part_index[i] = 0;
        itr->part_shape[i] = itr->shape[i];
    }
    itr->part_size = itr->shape_size;

    //update_index
    itr->part_index[ndim - 1] = itr->cont % (itr->eshape[ndim - 1] / itr->shape[ndim - 1]);
    itr->elem_index[ndim - 1] = itr->part_index[ndim - 1] * itr->shape[ndim - 1];

    int64_t inc = itr->eshape[ndim - 1] / itr->shape[ndim - 1];

    for (int i = ndim - 2; i >= 0; --i) {
        itr->part_index[i] = itr->cont % (inc * itr->eshape[i] / itr->shape[i]) / (inc);
        itr->elem_index[i] = itr->part_index[i] * itr->shape[i];
        inc *= itr->eshape[i] / itr->shape[i];
    }

    //calculate the buffer size
    itr->part_size = 1;
    for (int i = 0; i < ndim; ++i) {
        if ((itr->part_index[i] + 1) * itr->shape[i] > catarr->shape[i]) {
            itr->part_shape[i] = catarr->shape[i] - itr->eshape[i] + itr->shape[i];
        } else {
            itr->part_shape[i] = itr->shape[i];
        }
        itr->part_size *= itr->part_shape[i];
    }
}

/*
 * Function: iarray_iter_write_part_next
 * -------------------------------
 *   Update the iterator to next element
 *
 *   itr: an iterator
 */

INA_API(ina_rc_t) iarray_iter_write_part_next(iarray_iter_write_part_t *itr)
{
    caterva_array_t *catarr = itr->container->catarr;
    int8_t ndim = catarr->ndim;
    int64_t typesize = itr->container->catarr->ctx->cparams.typesize;
    int64_t psizeb = itr->part_size * typesize;

    if (itr->container->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        caterva_dims_t start = caterva_new_dims(itr->elem_index, ndim);

        int64_t stop_[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < ndim; ++i) {
            stop_[i] = start.dims[i] + itr->part_shape[i];
        }
        caterva_dims_t stop = caterva_new_dims(stop_, ndim);

        caterva_set_slice_buffer(catarr, itr->part, &start, &stop);
    } else {

        // check if the part should be padded with 0s
        if (itr->part_size == catarr->psize) {
            int err = blosc2_schunk_append_buffer(catarr->sc, itr->part, (size_t) psizeb);
            if (err < 0) {
                return INA_ERROR(INA_ERR_FAILED);
            }
        } else {
            uint8_t *part_aux = malloc((size_t) catarr->psize * typesize);
            memset(part_aux, 0, catarr->psize * typesize);

            //reverse part_shape
            int64_t shaper[CATERVA_MAXDIM];
            for (int i = 0; i < CATERVA_MAXDIM; ++i) {
                if (i >= CATERVA_MAXDIM - ndim) {
                    shaper[i] = itr->part_shape[i - CATERVA_MAXDIM + ndim];
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
                                               &(itr->part[itr_p * typesize]),
                                               shaper[7] * typesize);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            int err = blosc2_schunk_append_buffer(itr->container->catarr->sc, part_aux,
                                                  (size_t) catarr->psize * typesize);
            memset(part_aux, 0, catarr->psize * catarr->sc->typesize);
            if (err < 0) {
                return INA_ERROR(INA_ERR_FAILED);
            }

            free(part_aux);
        }
    }
    itr->cont += 1;

    //update_index
    itr->part_index[ndim - 1] = itr->cont % (itr->eshape[ndim - 1] / itr->shape[ndim - 1]);
    itr->elem_index[ndim - 1] = itr->part_index[ndim - 1] * itr->shape[ndim - 1];

    int64_t inc = itr->eshape[ndim - 1] / itr->shape[ndim - 1];

    for (int i = ndim - 2; i >= 0; --i) {
        itr->part_index[i] = itr->cont % (inc * itr->eshape[i] / itr->shape[i]) / (inc);
        itr->elem_index[i] = itr->part_index[i] * itr->shape[i];
        inc *= itr->eshape[i] / itr->shape[i];
    }

    //calculate the buffer size
    itr->part_size = 1;
    for (int i = 0; i < ndim; ++i) {
        if ((itr->part_index[i] + 1) * itr->shape[i] > catarr->shape[i]) {
            itr->part_shape[i] = catarr->shape[i] - itr->eshape[i] + itr->shape[i];
        } else {
            itr->part_shape[i] = itr->shape[i];
        }
        itr->part_size *= itr->part_shape[i];
    }

    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_write_part_finished
 * -----------------------------------
 *   Check if the iterator is finished
 *
 *   itr: an iterator
 *
 *   return: 1 if iter is finished or 0 if not
 */

INA_API(int) iarray_iter_write_part_finished(iarray_iter_write_part_t *itr)
{
    return itr->cont >= itr->esize / itr->shape_size;
}

/*
 * Function: iarray_iter_write_part_value
 * --------------------------------
 *   Store in `val` parameter some variables of the actual part
 *
 *   itr: an iterator
 *   val: a struct where data needed by the user is stored
 *     part_index: position in coord where the part is located in the container
 *     nelem: if the parts are row-wise listed, `nelem` is the part position in this list
 *     elem_index: position in coord where the first element of the part is located in the container
 *     part_shape: is the actual part part_shape. It should be used to compute the part size
 *     pointer: pointer to the first part element position in memory. It's used to copy the part into the container
 *
 *   return: INA_SUCCESS or an error code
 */

INA_API(void) iarray_iter_write_part_value(iarray_iter_write_part_t *itr, iarray_iter_write_part_value_t *val)
{
    val->pointer = itr->pointer;
    val->part_index = itr->part_index;
    val->elem_index = itr->elem_index;
    val->nelem = itr->cont;
    val->part_shape = itr->part_shape;
}

/*
 * Function: iarray_iter_write_part_new
 * ------------------------------
 *   Create a new iterator
 *
 *   ctx: iarray context
 *   container: the container used in the iterator
 *   itr: an iterator
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(ina_rc_t) iarray_iter_write_part_new(iarray_context_t *ctx, iarray_container_t *container,
                                             iarray_iter_write_part_t **itr,
                                             const int64_t *blockshape)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(itr);
    *itr = (iarray_iter_write_part_t*)ina_mem_alloc(sizeof(iarray_iter_write_part_t));
    INA_RETURN_IF_NULL(itr);

    if (blockshape != NULL & container->catarr->storage == CATERVA_STORAGE_BLOSC) {
        return INA_ERROR(INA_ERR_FAILED);
    }

    if (blockshape == NULL) {
        blockshape = container->dtshape->pshape;
    }
    int64_t typesize = container->catarr->ctx->cparams.typesize;

    caterva_dims_t shape = caterva_new_dims(container->dtshape->shape, container->dtshape->ndim);
    int err = caterva_update_shape(container->catarr, &shape);
    container->catarr->buf = container->catarr->ctx->alloc((size_t) container->catarr->size * typesize);

    if (err < 0) {
        return INA_ERROR(INA_ERR_FAILED);
    }

    (*itr)->ctx = ctx;
    (*itr)->container = container;

    (*itr)->part_index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->elem_index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->part_shape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->shape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->eshape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));

    (*itr)->esize = 1;
    (*itr)->shape_size = 1;
    int64_t size = typesize;
    for (int i = 0; i < (*itr)->container->dtshape->ndim; ++i) {
        (*itr)->shape[i] = blockshape[i];
        size *= (*itr)->shape[i];
        if (container->catarr->eshape[i] % blockshape[i] == 0) {
            (*itr)->eshape[i] = (container->catarr->eshape[i] / blockshape[i]) * blockshape[i];
        } else {
            (*itr)->eshape[i] = (container->catarr->eshape[i] / blockshape[i] + 1) * blockshape[i];

        }
        (*itr)->esize *= (*itr)->eshape[i];
        (*itr)->shape_size *= (*itr)->shape[i];
    }

    (*itr)->part = (uint8_t *) ina_mem_alloc((size_t) size * typesize);
    (*itr)->pointer = &(*itr)->part[0];

    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_write_part_free
 * -------------------------------
 *   Free an iterator structure
 *
 *   itr: an iterator
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(void) iarray_iter_write_part_free(iarray_iter_write_part_t *itr)
{
    ina_mem_free(itr->part_index);
    ina_mem_free(itr->elem_index);
    ina_mem_free(itr->part_shape);
    ina_mem_free(itr->part);
    ina_mem_free(itr->shape);
    ina_mem_free(itr->eshape);

    ina_mem_free(itr);
}

/*
 * Matmul iterator
 *
 * Internal iterator used to perform easily matrix-matrix or vector-matrix multiplications by blocks
 *
 */


/*
 * Function: iarray_iter_matmul_init
 * --------------------------------
 *   Set the iterator values to the first element
 *
 *   itr: an iterator
 */

void _iarray_iter_matmul_init(iarray_iter_matmul_t *itr)
{
    itr->cont = 0;
    itr->npart1 = 0;
    itr->npart2 = 0;
}

/*
 * Function: iarray_iter_matmul_next
 * --------------------------------
 *   Update the block to be used of each container
 *
*   itr: an iterator
 */

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

/*
 * Function: iarray_iter_matmul_finished
 * ------------------------------------
 *   Check if the iterator is finished
 *
 *   itr: an iterator
 *
 *   return: 1 if iter is finished or 0 if not
 */

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

/*
 * Function: iarray_iter_matmul_new
 * ------------------------
 *   Create a matmul iterator
 *
 *   ctx: iarray context
 *   itr: an iterator
 *
*   return: INA_SUCCESS or an error code
 */

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

/*
 * Function: iarray_iter_matmul_free
 * --------------------------------
 *   Free an iterator structure
 *
 *   itr: an iterator
 *
*   return: INA_SUCCESS or an error code
 */

void _iarray_iter_matmul_free(iarray_iter_matmul_t *itr)
{
    ina_mem_free(itr);
}

/*
 * Element by element read iterator
 */

/*
 * Function: iarray_iter_read_init
 */

INA_API(void) iarray_iter_read_init(iarray_iter_read_t *itr)
{
    // Initialize element and block index
    for (int i = 0; i <IARRAY_DIMENSION_MAX; ++i) {
        itr->elem_index[i] = 0;
        itr->block_index[i] = 0;
    }

    // Initialize counters
    itr->elem_cont = 0;
    itr->block_cont = 0;
    itr->elem_cont_block = 0;

    // Initialize block_ params
    int64_t stop_[IARRAY_DIMENSION_MAX];
    int64_t buflen = 1;

    itr->block_size = 1;
    itr->c_size = 1;
    for (int i = 0; i < itr->container->dtshape->ndim; ++i) {
        itr->block_shape[i] = itr->shape[i];
        itr->block_size *= itr->block_shape[i];
        stop_[i] = itr->elem_index[i] + itr->shape[i];
        buflen *= itr->shape[i];
        itr->c_size *= itr->container->dtshape->shape[i];
    }

    // Decompress first block
    INA_MUST_SUCCEED(iarray_get_slice_buffer(itr->ctx, itr->container, (int64_t *) itr->elem_index,
                                             (int64_t *) stop_, itr->part,
                                             buflen * itr->container->catarr->ctx->cparams.typesize));
}

/*
 * Function: iarray_iter_read_next
 */

INA_API(ina_rc_t) iarray_iter_read_next(iarray_iter_read_t *itr)
{
    caterva_array_t *catarr = itr->container->catarr;
    int ndim = catarr->ndim;

    int64_t typesize = itr->container->catarr->ctx->cparams.typesize;

    // check if a block is readed totally and decompress next
    if (itr->elem_cont_block  == itr->block_size - 1) {

        if (itr->container->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
            itr->elem_cont = itr->c_size;
            return INA_SUCCESS;
        } else {
            if (itr->elem_cont == itr->c_size - 1) {
                itr->elem_cont++;
                return INA_SUCCESS;
            }

            // Update block counter
            itr->block_cont += 1;

            // Calculate aux variables
            int64_t aux[IARRAY_DIMENSION_MAX];
            for (int i = ndim - 1; i >= 0; --i) {
                if (itr->container->dtshape->shape[i] % itr->shape[i] == 0) {
                    aux[i] = itr->container->dtshape->shape[i] / itr->shape[i];
                } else {
                    aux[i] = itr->container->dtshape->shape[i] / itr->shape[i] + 1;
                }
            }

            // Calculate the start of the next block
            int64_t start_[IARRAY_DIMENSION_MAX];

            int64_t inc = 1;
            for (int i = ndim - 1; i >= 0; --i) {
                start_[i] = itr->block_cont % (aux[i] * inc) / inc;
                itr->block_index[i] = start_[i];
                start_[i] *= itr->shape[i];
                itr->elem_index[i] = start_[i];
                inc *= aux[i];
            }

            // Calculate the stop of the next block
            int64_t stop_[IARRAY_DIMENSION_MAX];
            int64_t buflen = 1;
            itr->block_size = 1;
            for (int i = ndim - 1; i >= 0; --i) {
                if (start_[i] + itr->shape[i] <= itr->container->dtshape->shape[i]) {
                    stop_[i] = start_[i] + itr->shape[i];
                } else {
                    stop_[i] = itr->container->dtshape->shape[i];
                }
                itr->block_shape[i] = stop_[i] - start_[i];
                itr->block_size *= itr->block_shape[i];
                buflen *= itr->shape[i];
            }

            // Decompress the next block
            INA_MUST_SUCCEED(iarray_get_slice_buffer(itr->ctx, itr->container, (int64_t *) start_,
                                                     (int64_t *) stop_, itr->part, buflen * typesize));

            itr->elem_cont_block = 0;
        }
    } else {
        // Go to next element of the block if it is not read totally
        itr->elem_cont_block += 1;
    }

    // jump to the next element
    itr->elem_cont += 1;

    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_read_finished
 */

INA_API(int) iarray_iter_read_finished(iarray_iter_read_t *itr)
{
    return itr->elem_cont >= itr->c_size;
}

/*
 * Function: iarray_iter_read_value
 */

INA_API(void) iarray_iter_read_value(iarray_iter_read_t *itr, iarray_iter_read_value_t *val)
{
    int64_t typesize = itr->container->catarr->ctx->cparams.typesize;

    int8_t ndim = itr->container->dtshape->ndim;
    int64_t *c_shape = itr->container->dtshape->shape;
    int64_t ind_part_elem[IARRAY_DIMENSION_MAX];
    int64_t inc = 1;
    int64_t inc_s = 1;

    itr->nelem = 0;
    for (int i = ndim - 1; i >= 0; --i) {
        ind_part_elem[i] = itr->elem_cont_block % (inc * itr->block_shape[i]) / inc;
        itr->index[i] = ind_part_elem[i] + itr->block_index[i] * itr->shape[i];
        itr->nelem += itr->index[i] * inc_s;
        inc_s *= c_shape[i];
        inc *= itr->block_shape[i];
    }
    itr->pointer = (void *)&(itr->part)[itr->elem_cont_block * typesize];

    val->index = itr->index;
    val->pointer = itr->pointer;
    val->nelem = itr->nelem;
}

/*
 * Function: iarray_iter_read_new
 */

INA_API(ina_rc_t) iarray_iter_read_new(iarray_context_t *ctx, iarray_container_t *container,
                                       iarray_iter_read_t **itr)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(itr);

    *itr = (iarray_iter_read_t*)ina_mem_alloc(sizeof(iarray_iter_read_t));
    INA_RETURN_IF_NULL(itr);

    (*itr)->ctx = ctx;
    (*itr)->container = container;
    if (container->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        (*itr)->part = container->catarr->buf;
    } else {
        (*itr)->part = (uint8_t *) ina_mem_alloc((size_t) container->catarr->psize * container->catarr->sc->typesize);
    }
    (*itr)->index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));

    (*itr)->shape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));

    (*itr)->block_shape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->block_index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->elem_index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));

    for (int i = 0; i < container->dtshape->ndim; ++i) {
        (*itr)->shape[i] = container->dtshape->pshape[i];
    }

    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_read_free
 */

INA_API(void) iarray_iter_read_free(iarray_iter_read_t *itr)
{
    ina_mem_free(itr->index);
    if (itr->container->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
        ina_mem_free(itr->part);
    }
    ina_mem_free(itr->shape);
    ina_mem_free(itr->block_shape);
    ina_mem_free(itr->block_index);
    ina_mem_free(itr->elem_index);
    ina_mem_free(itr);
}


/*
 * Read iterator by blocks
 *
 * Iterator that allows read an iarray container by blocks (the blocksize is specified by the user)
 */

/*
 * Function: iarray_iter_read_block_init
 */

INA_API(void) iarray_iter_read_block_init(iarray_iter_read_block_t *itr)
{
    int64_t typesize = itr->container->catarr->ctx->cparams.typesize;

    for (int i = 0; i <IARRAY_DIMENSION_MAX; ++i) {
        itr->elem_index[i] = 0;
        itr->block_index[i] = 0;
    }
    itr->cont = 0;

    int64_t stop_[IARRAY_DIMENSION_MAX];
    int64_t buflen = 1;

    itr->block_size = 1;
    for (int i = 0; i < itr->container->dtshape->ndim; ++i) {
        itr->block_shape[i] = itr->shape[i];
        itr->block_size *= itr->block_shape[i];
        stop_[i] = itr->elem_index[i] + itr->shape[i];
        buflen *= itr->shape[i];
    }

    INA_MUST_SUCCEED(iarray_get_slice_buffer(itr->ctx, itr->container, (int64_t *) itr->elem_index,
                                             (int64_t *) stop_, itr->part,
                                             buflen * typesize));
}

/*
 * Function: iarray_iter_read_block_next
 */

INA_API(ina_rc_t) iarray_iter_read_block_next(iarray_iter_read_block_t *itr)
{
    int64_t typesize = itr->container->catarr->ctx->cparams.typesize;

    int8_t ndim = itr->container->dtshape->ndim;
    itr->cont += 1;

    int64_t aux[IARRAY_DIMENSION_MAX];
    for (int i = ndim - 1; i >= 0; --i) {
        if (itr->container->dtshape->shape[i] % itr->shape[i] == 0) {
            aux[i] = itr->container->dtshape->shape[i] / itr->shape[i];
        } else {
            aux[i] = itr->container->dtshape->shape[i] / itr->shape[i] + 1;
        }
    }

    int64_t start_[IARRAY_DIMENSION_MAX];

    int64_t inc = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        start_[i] = itr->cont % (aux[i] * inc) / inc;
        itr->block_index[i] = start_[i];
        start_[i] *= itr->shape[i];
        itr->elem_index[i] = start_[i];
        inc *= aux[i];
    }

    int64_t stop_[IARRAY_DIMENSION_MAX];
    int64_t buflen = 1;
    itr->block_size = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if(start_[i] + itr->shape[i] <= itr->container->dtshape->shape[i]) {
            stop_[i] = start_[i] + itr->shape[i];
        } else {
            stop_[i] = itr->container->dtshape->shape[i];
        }
        itr->block_shape[i] = stop_[i] - start_[i];
        itr->block_size *= itr->block_shape[i];
        buflen *= itr->shape[i];
    }

    INA_MUST_SUCCEED(iarray_get_slice_buffer(itr->ctx, itr->container, (int64_t *) start_,
                                             (int64_t *) stop_, itr->part, buflen * typesize));

    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_read_block_finished
 */

INA_API(int) iarray_iter_read_block_finished(iarray_iter_read_block_t *itr)
{
    int64_t size = 1;
    for (int i = 0; i < itr->container->dtshape->ndim; ++i) {
        if(itr->container->dtshape->shape[i] % itr->shape[i] == 0) {
            size *= itr->container->dtshape->shape[i] / itr->shape[i];
        } else {
            size *= itr->container->dtshape->shape[i] / itr->shape[i] + 1;
        }
    }
    return itr->cont >= size;
}

/*
 * Function: iarray_iter_read_block_value
 */

INA_API(void) iarray_iter_read_block_value(iarray_iter_read_block_t *itr,
                                           iarray_iter_read_block_value_t *val)
{
    val->pointer = itr->pointer;
    val->block_index = itr->block_index;
    val->elem_index = itr->elem_index;
    val->nelem = itr->cont;
    val->block_shape = itr->block_shape;
}

/*
 * Function: iarray_iter_read_block_new
 */

INA_API(ina_rc_t) iarray_iter_read_block_new(iarray_context_t *ctx, iarray_container_t *container,
                                             iarray_iter_read_block_t **itr,
                                             const int64_t *blockshape)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(itr);
    *itr = (iarray_iter_read_block_t*) ina_mem_alloc(sizeof(iarray_iter_read_block_t));
    INA_RETURN_IF_NULL(itr);

    if (blockshape == NULL) {
        blockshape = container->dtshape->shape;
    }

    int64_t typesize = container->catarr->ctx->cparams.typesize;

    (*itr)->ctx = ctx;
    (*itr)->container = container;
    (*itr)->shape = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));
    (*itr)->block_shape = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));
    (*itr)->block_index = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));
    (*itr)->elem_index = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));

    int64_t size = typesize;
    for (int i = 0; i < (*itr)->container->dtshape->ndim; ++i) {
        (*itr)->shape[i] = blockshape[i];
        size *= (*itr)->shape[i];
    }

    (*itr)->part = ina_mem_alloc((size_t) size);
    (*itr)->pointer = &((*itr)->part[0]);

    // Create a cache in the underlying container so as to accelerate the getting of a slice
    INA_FAIL_IF(container->catarr->part_cache.data != NULL);
    INA_FAIL_IF(container->catarr->part_cache.nchunk != -1);
    // TODO: Using ina_mem_alloc instead of ina_mempool_dalloc makes the
    //  `./perf_vectors -I -e 3 -c 5` bench to fail.  Investigate more.
    // container->catarr->part_cache.data = ina_mem_alloc((size_t)size);
    // memset(container->catarr->part_cache.data, 0, (size_t)size);
    container->catarr->part_cache.data = ina_mempool_dalloc(ctx->mp, (size_t)size);

    return INA_SUCCESS;

fail:
    return ina_err_get_rc();

}

/*
 * Function: iarray_iter_read_block_free
 */

INA_API(void) iarray_iter_read_block_free(iarray_iter_read_block_t *itr)
{
    ina_mem_free(itr->shape);
    ina_mem_free(itr->block_shape);
    ina_mem_free(itr->block_index);
    ina_mem_free(itr->elem_index);
    ina_mem_free(itr->part);

    //ina_mem_free(itr->container->catarr->part_cache.data);  // TODO: investigate (see above)
    itr->container->catarr->part_cache.data = NULL;  // reset to NULL here (the memory pool will be reset later)
    itr->container->catarr->part_cache.nchunk = -1;  // means no valid cache yet
    ina_mem_free(itr);
}


/*
 * Read iterator by blocks 2 version
 *
 * Iterator that allows read an iarray container by blocks (the blocksize is specified by the user)
 */


/*
 * Function: iarray_iter_read_block_next
 */

INA_API(ina_rc_t) iarray_iter_read_block2_next(iarray_iter_read_block2_t *itr)
{
    int64_t typesize = itr->cont->catarr->ctx->cparams.typesize;
    int8_t ndim = itr->cont->dtshape->ndim;

    // Calculate the start of the desired block
    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t inc = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        start_[i] = itr->nblock % (itr->aux[i] * inc) / inc;
        itr->act_block_index[i] = start_[i];
        start_[i] *= itr->block_shape[i];
        itr->act_elem_index[i] = start_[i];
        inc *= itr->aux[i];
    }

    // Calculate the stop of the desired block
    int64_t stop_[IARRAY_DIMENSION_MAX];
    int64_t actual_block_size = 1;
    itr->act_block_size = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if(start_[i] + itr->block_shape[i] <= itr->cont->dtshape->shape[i]) {
            stop_[i] = start_[i] + itr->block_shape[i];
        } else {
            stop_[i] = itr->cont->dtshape->shape[i];
        }
        itr->actual_block_shape[i] = stop_[i] - start_[i];
        itr->act_block_size *= itr->actual_block_shape[i];
        actual_block_size *= itr->block_shape[i];
    }

    // Get the desired block
    INA_MUST_SUCCEED(iarray_get_slice_buffer(itr->ctx, itr->cont, (int64_t *) start_,
                                             (int64_t *) stop_, itr->part, actual_block_size * typesize));

    // Update the structure that user can see
    itr->val->pointer = itr->pointer;
    itr->val->block_index = itr->act_block_index;
    itr->val->elem_index = itr->act_elem_index;
    itr->val->nelem = itr->nblock;
    itr->val->block_shape = itr->actual_block_shape;
    itr->val->block_size = actual_block_size;
    // Increment the block counter
    itr->nblock += 1;

    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_read_block_finished
 */

INA_API(int) iarray_iter_read_block2_has_next(iarray_iter_read_block2_t *itr)
{
    return itr->nblock < itr->total_blocks;
}


/*
 * Function: iarray_iter_read_block_new
 */

INA_API(ina_rc_t) iarray_iter_read_block2_new(iarray_context_t *ctx,
                                              iarray_iter_read_block2_t **itr,
                                              iarray_container_t *cont,
                                              const int64_t *blockshape,
                                              iarray_iter_read_block2_value_t *val)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(itr);
    *itr = (iarray_iter_read_block2_t*) ina_mem_alloc(sizeof(iarray_iter_read_block2_t));
    INA_RETURN_IF_NULL(itr);

    (*itr)->ctx = ctx;

    INA_VERIFY_NOT_NULL(cont);
    (*itr)->cont = cont;
    int64_t typesize = (*itr)->cont->catarr->ctx->cparams.typesize;

    if (blockshape == NULL) {
        blockshape = cont->dtshape->shape;
    }

    (*itr)->aux = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));
    (*itr)->block_shape = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));
    (*itr)->actual_block_shape = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));
    (*itr)->act_block_index = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));
    (*itr)->act_elem_index = (int64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));

    // Create a buffer where data is stored to pass it to the user
    int64_t block_size = typesize;
    for (int i = 0; i < cont->dtshape->ndim; ++i) {
        (*itr)->block_shape[i] = blockshape[i];
        block_size *= (*itr)->block_shape[i];
    }
    (*itr)->part = ina_mem_alloc((size_t) block_size);
    (*itr)->pointer = &((*itr)->part[0]);

    (*itr)->val = val;

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
        (*itr)->act_elem_index[i] = 0;
        (*itr)->act_block_index[i] = 0;
    }
    (*itr)->nblock = 0;

    // Create a cache in the underlying container so as to accelerate the getting of a slice
    // INA_FAIL_IF(container->catarr->part_cache.data != NULL);
    // INA_FAIL_IF(container->catarr->part_cache.nchunk != -1);
    // TODO: Using ina_mem_alloc instead of ina_mempool_dalloc makes the
    //  `./perf_vectors -I -e 3 -c 5` bench to fail.  Investigate more.
    // container->catarr->part_cache.data = ina_mem_alloc((size_t)size);
    // memset(container->catarr->part_cache.data, 0, (size_t)size);
    //container->catarr->part_cache.data = ina_mempool_dalloc(ctx->mp, (size_t)size);

    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_read_block_free
 */

INA_API(void) iarray_iter_read_block2_free(iarray_iter_read_block2_t *itr)
{
    ina_mem_free(itr->block_shape);
    ina_mem_free(itr->actual_block_shape);
    ina_mem_free(itr->act_block_index);
    ina_mem_free(itr->act_elem_index);
    ina_mem_free(itr->part);

    //ina_mem_free(itr->container->catarr->part_cache.data);  // TODO: investigate (see above)
    itr->cont->catarr->part_cache.data = NULL;  // reset to NULL here (the memory pool will be reset later)
    itr->cont->catarr->part_cache.nchunk = -1;  // means no valid cache yet
    ina_mem_free(itr);
}


/*
 * Partition by partition iterator
 *
 * Unlike the previous, the next collection of functions are used to fill an iarray container part by part
 */


/*
 * Function: iarray_iter_write_part_next
 * -------------------------------
 *   Update the iterator to next element
 *
 *   itr: an iterator
 */

INA_API(ina_rc_t) iarray_iter_write_block2_next(iarray_iter_write_part_t *itr) {
    caterva_array_t *catarr = itr->container->catarr;
    int8_t ndim = catarr->ndim;
    int64_t typesize = itr->container->catarr->ctx->cparams.typesize;
    int64_t psizeb = itr->part_size * typesize;

    if (itr->cont != 0) {

        printf("Append data\n");
        if (itr->container->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
            caterva_dims_t start = caterva_new_dims(itr->elem_index, ndim);

            int64_t stop_[IARRAY_DIMENSION_MAX];
            for (int i = 0; i < ndim; ++i) {
                stop_[i] = start.dims[i] + itr->part_shape[i];
            }
            caterva_dims_t stop = caterva_new_dims(stop_, ndim);

            caterva_set_slice_buffer(catarr, itr->part, &start, &stop);
        } else {

            // check if the part should be padded with 0s
            if (itr->part_size == catarr->psize) {
                int err = blosc2_schunk_append_buffer(catarr->sc, itr->part, (size_t) psizeb);
                if (err < 0) {
                    return INA_ERROR(INA_ERR_FAILED);
                }
            } else {
                uint8_t *part_aux = malloc((size_t) catarr->psize * typesize);
                memset(part_aux, 0, catarr->psize * typesize);

                //reverse part_shape
                int64_t shaper[CATERVA_MAXDIM];
                for (int i = 0; i < CATERVA_MAXDIM; ++i) {
                    if (i >= CATERVA_MAXDIM - ndim) {
                        shaper[i] = itr->part_shape[i - CATERVA_MAXDIM + ndim];
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
                                                   &(itr->part[itr_p * typesize]),
                                                   shaper[7] * typesize);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                int err = blosc2_schunk_append_buffer(itr->container->catarr->sc, part_aux,
                                                      (size_t) catarr->psize * typesize);
                memset(part_aux, 0, catarr->psize * catarr->sc->typesize);
                if (err < 0) {
                    return INA_ERROR(INA_ERR_FAILED);
                }

                free(part_aux);
            }
        }
    }
    //update_index
    itr->part_index[ndim - 1] = itr->cont % (itr->eshape[ndim - 1] / itr->shape[ndim - 1]);
    itr->elem_index[ndim - 1] = itr->part_index[ndim - 1] * itr->shape[ndim - 1];

    int64_t inc = itr->eshape[ndim - 1] / itr->shape[ndim - 1];

    for (int i = ndim - 2; i >= 0; --i) {
        itr->part_index[i] = itr->cont % (inc * itr->eshape[i] / itr->shape[i]) / (inc);
        itr->elem_index[i] = itr->part_index[i] * itr->shape[i];
        inc *= itr->eshape[i] / itr->shape[i];
    }

    //calculate the buffer size
    itr->part_size = 1;
    for (int i = 0; i < ndim; ++i) {
        if ((itr->part_index[i] + 1) * itr->shape[i] > catarr->shape[i]) {
            itr->part_shape[i] = catarr->shape[i] - itr->eshape[i] + itr->shape[i];
        } else {
            itr->part_shape[i] = itr->shape[i];
        }
        itr->part_size *= itr->part_shape[i];
    }

    itr->cont += 1;

    itr->val->pointer = itr->pointer;
    itr->val->part_index = itr->part_index;
    itr->val->elem_index = itr->elem_index;
    itr->val->nelem = itr->cont;
    itr->val->part_shape = itr->part_shape;

    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_write_part_finished
 * -----------------------------------
 *   Check if the iterator is finished
 *
 *   itr: an iterator
 *
 *   return: 1 if iter is finished or 0 if not
 */

INA_API(int) iarray_iter_write_block2_has_next(iarray_iter_write_part_t *itr)
{
    printf("Block %llu of %llu\n", itr->cont, itr->esize / itr->shape_size);
    if ( itr->cont == (itr->esize / itr->shape_size)) {
        printf("Append data\n");
        caterva_array_t *catarr = itr->container->catarr;
        int8_t ndim = catarr->ndim;
        int64_t typesize = itr->container->catarr->ctx->cparams.typesize;
        int64_t psizeb = itr->part_size * typesize;
        if (itr->container->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
            caterva_dims_t start = caterva_new_dims(itr->elem_index, ndim);

            int64_t stop_[IARRAY_DIMENSION_MAX];
            for (int i = 0; i < ndim; ++i) {
                stop_[i] = start.dims[i] + itr->part_shape[i];
            }
            caterva_dims_t stop = caterva_new_dims(stop_, ndim);

            caterva_set_slice_buffer(catarr, itr->part, &start, &stop);
        } else {

            // check if the part should be padded with 0s
            if (itr->part_size == catarr->psize) {
                blosc2_schunk_append_buffer(catarr->sc, itr->part, (size_t) psizeb);
            } else {
                uint8_t *part_aux = malloc((size_t) catarr->psize * typesize);
                memset(part_aux, 0, catarr->psize * typesize);

                //reverse part_shape
                int64_t shaper[CATERVA_MAXDIM];
                for (int i = 0; i < CATERVA_MAXDIM; ++i) {
                    if (i >= CATERVA_MAXDIM - ndim) {
                        shaper[i] = itr->part_shape[i - CATERVA_MAXDIM + ndim];
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
                                                   &(itr->part[itr_p * typesize]),
                                                   shaper[7] * typesize);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                blosc2_schunk_append_buffer(itr->container->catarr->sc, part_aux,
                                            (size_t) catarr->psize * typesize);
                memset(part_aux, 0, catarr->psize * catarr->sc->typesize);


                free(part_aux);
            }
        }
    }
    return itr->cont < (itr->esize / itr->shape_size);
}


/*
 * Function: iarray_iter_write_part_new
 * ------------------------------
 *   Create a new iterator
 *
 *   ctx: iarray context
 *   container: the container used in the iterator
 *   itr: an iterator
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(ina_rc_t) iarray_iter_write_block2_new(iarray_context_t *ctx,
                                             iarray_iter_write_part_t **itr,
                                             iarray_container_t *container,
                                             const int64_t *blockshape,
                                             iarray_iter_write_block2_value_t *val)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(itr);
    *itr = (iarray_iter_write_part_t*)ina_mem_alloc(sizeof(iarray_iter_write_part_t));
    INA_RETURN_IF_NULL(itr);

    if (blockshape != NULL & container->catarr->storage == CATERVA_STORAGE_BLOSC) {
        return INA_ERROR(INA_ERR_FAILED);
    }

    if (blockshape == NULL) {
        blockshape = container->dtshape->pshape;
    }
    int64_t typesize = container->catarr->ctx->cparams.typesize;

    caterva_dims_t shape = caterva_new_dims(container->dtshape->shape, container->dtshape->ndim);
    int err = caterva_update_shape(container->catarr, &shape);
    container->catarr->buf = container->catarr->ctx->alloc((size_t) container->catarr->size * typesize);

    if (err < 0) {
        return INA_ERROR(INA_ERR_FAILED);
    }

    (*itr)->val = val;
    (*itr)->ctx = ctx;
    (*itr)->container = container;

    (*itr)->part_index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->elem_index = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->part_shape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->shape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));
    (*itr)->eshape = (int64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(int64_t));

    (*itr)->esize = 1;
    (*itr)->shape_size = 1;
    int64_t size = typesize;
    for (int i = 0; i < (*itr)->container->dtshape->ndim; ++i) {
        (*itr)->shape[i] = blockshape[i];
        size *= (*itr)->shape[i];
        if (container->catarr->eshape[i] % blockshape[i] == 0) {
            (*itr)->eshape[i] = (container->catarr->eshape[i] / blockshape[i]) * blockshape[i];
        } else {
            (*itr)->eshape[i] = (container->catarr->eshape[i] / blockshape[i] + 1) * blockshape[i];

        }
        (*itr)->esize *= (*itr)->eshape[i];
        (*itr)->shape_size *= (*itr)->shape[i];
    }

    (*itr)->part = (uint8_t *) ina_mem_alloc((size_t) size * typesize);
    (*itr)->pointer = &(*itr)->part[0];

    int8_t ndim = (*itr)->container->dtshape->ndim;
    caterva_array_t *catarr = (*itr)->container->catarr;

    (*itr)->cont = 0;
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        (*itr)->part_index[i] = 0;
        (*itr)->part_shape[i] = (*itr)->shape[i];
    }
    (*itr)->part_size = (*itr)->shape_size;

    //update_index
    (*itr)->part_index[ndim - 1] = (*itr)->cont % ((*itr)->eshape[ndim - 1] / (*itr)->shape[ndim - 1]);
    (*itr)->elem_index[ndim - 1] = (*itr)->part_index[ndim - 1] * (*itr)->shape[ndim - 1];

    int64_t inc = (*itr)->eshape[ndim - 1] / (*itr)->shape[ndim - 1];

    for (int i = ndim - 2; i >= 0; --i) {
        (*itr)->part_index[i] = (*itr)->cont % (inc * (*itr)->eshape[i] / (*itr)->shape[i]) / (inc);
        (*itr)->elem_index[i] = (*itr)->part_index[i] * (*itr)->shape[i];
        inc *= (*itr)->eshape[i] / (*itr)->shape[i];
    }

    //calculate the buffer size
    (*itr)->part_size = 1;
    for (int i = 0; i < ndim; ++i) {
        if (((*itr)->part_index[i] + 1) * (*itr)->shape[i] > catarr->shape[i]) {
            (*itr)->part_shape[i] = catarr->shape[i] - (*itr)->eshape[i] + (*itr)->shape[i];
        } else {
            (*itr)->part_shape[i] = (*itr)->shape[i];
        }
        (*itr)->part_size *= (*itr)->part_shape[i];
    }
    
    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_write_part_free
 * -------------------------------
 *   Free an iterator structure
 *
 *   itr: an iterator
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(void) iarray_iter_write_block2_free(iarray_iter_write_part_t *itr)
{
    ina_mem_free(itr->part_index);
    ina_mem_free(itr->elem_index);
    ina_mem_free(itr->part_shape);
    ina_mem_free(itr->part);
    ina_mem_free(itr->shape);
    ina_mem_free(itr->eshape);

    ina_mem_free(itr);
}