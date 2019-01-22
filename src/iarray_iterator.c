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

    memset(itr->part, 0, itr->container->catarr->psize * itr->container->catarr->sc->typesize);
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

    // check if a part is filled totally and append it

    if (itr->cont_part_elem  == itr->bsize - 1) {
        int err = blosc2_schunk_append_buffer(catarr->sc, itr->part, catarr->psize * catarr->sc->typesize);
        if (err < 0) {
            return INA_ERROR(INA_ERR_FAILED);
        }
        itr->cont_part_elem = 0;
        itr->cont_part += 1;
        uint64_t inc = 1;
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
        memset(itr->part, 0, catarr->psize * catarr->sc->typesize);
    } else {
        itr->cont_part_elem += 1;
    }

    // jump to the next element
    itr->cont += 1;

    uint64_t ind_part_elem[IARRAY_DIMENSION_MAX];
    uint64_t cont_pointer = 0;

    uint64_t inc = 1;
    uint64_t inc_s = 1;
    uint64_t inc_p = 1;

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
    itr->pointer = (void *)&(itr->part)[cont_pointer * catarr->sc->typesize];

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
    int err = caterva_update_shape(container->catarr, shape);
    if (err < 0) {
        return INA_ERROR(INA_ERR_FAILED);
    }
    (*itr)->ctx = ctx;
    (*itr)->container = container;
    (*itr)->part = (uint8_t *) ina_mem_alloc(container->catarr->psize * container->catarr->sc->typesize);
    (*itr)->index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));
    (*itr)->part_index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));
    (*itr)->bshape = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));


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
    ina_mem_free(itr->part);
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
    itr->cont = 0;
    memset(itr->part, 0, itr->container->catarr->psize * itr->container->catarr->sc->typesize);
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        itr->part_index[i] = 0;
        itr->part_shape[i] = itr->container->dtshape->pshape[i];
    }
    itr->part_size = itr->container->catarr->psize;
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
    int ndim = catarr->ndim;

    uint64_t psizeb = itr->part_size * catarr->sc->typesize;

    // check if the part should be padded with 0s
    if ( itr->part_size == catarr->psize ) {
        int err = blosc2_schunk_append_buffer(catarr->sc, itr->part, psizeb);
        if (err < 0) {
            return INA_ERROR(INA_ERR_FAILED);
        }
    } else {
        uint8_t *part_aux = malloc(catarr->psize * catarr->sc->typesize);

        //reverse part_shape
        uint64_t shaper[CATERVA_MAXDIM];
        for (int i = 0; i < CATERVA_MAXDIM; ++i) {
            if(i >= CATERVA_MAXDIM - ndim) {
                shaper[i] = itr->part_shape[i - CATERVA_MAXDIM + ndim];
            } else {
                shaper[i] = 1;
            }
        }

        //copy buffer data to an aux buffer padded with 0's
        uint64_t ii[CATERVA_MAXDIM];
        for (ii[0] = 0; ii[0] < shaper[0]; ++ii[0]) {
            for (ii[1] = 0; ii[1] < shaper[1]; ++ii[1]) {
                for (ii[2] = 0; ii[2] < shaper[2]; ++ii[2]) {
                    for (ii[3] = 0; ii[3] < shaper[3]; ++ii[3]) {
                        for (ii[4] = 0; ii[4] < shaper[4]; ++ii[4]) {
                            for (ii[5] = 0; ii[5] < shaper[5]; ++ii[5]) {
                                for (ii[6] = 0; ii[6] < shaper[6]; ++ii[6]) {

                                    uint64_t aux_p = 0;
                                    uint64_t aux_i = catarr->pshape[ndim - 1];

                                    for (int i = ndim - 2; i >= 0; --i) {
                                        aux_p += ii[CATERVA_MAXDIM - ndim + i] * aux_i;
                                        aux_i *= catarr->pshape[i];
                                    }

                                    uint64_t itr_p = 0;
                                    uint64_t itr_i = shaper[CATERVA_MAXDIM - 1];

                                    for (int i = CATERVA_MAXDIM - 2; i >= CATERVA_MAXDIM - ndim; --i) {
                                        itr_p += ii[i] * itr_i;
                                        itr_i *= shaper[i];
                                    }
                                    memcpy(&part_aux[aux_p * catarr->sc->typesize], &(itr->part[itr_p * catarr->sc->typesize]), shaper[7] * catarr->sc->typesize);
                                }
                            }
                        }
                    }
                }
            }
        }

        int err = blosc2_schunk_append_buffer(itr->container->catarr->sc, part_aux, catarr->psize * catarr->sc->typesize);
        if (err < 0) {
            return INA_ERROR(INA_ERR_FAILED);
        }

        free(part_aux);
    }
    itr->cont += 1;

    //update_index
    itr->part_index[ndim - 1] = itr->cont % (catarr->eshape[ndim - 1] / catarr->pshape[ndim - 1]);
    uint64_t inc = catarr->eshape[ndim - 1] / catarr->pshape[ndim - 1];

    for (int i = ndim - 2; i >= 0; --i) {
        itr->part_index[i] = itr->cont % (inc * catarr->eshape[i] / catarr->pshape[i]) / (inc);
        itr->elem_index[i] = itr->part_index[i] * catarr->pshape[i];
        inc *= catarr->eshape[i] / catarr->pshape[i];
    }

    //calculate the buffer size
    itr->part_size = 1;
    for (int i = 0; i < ndim; ++i) {
        if ((itr->part_index[i] + 1) * catarr->pshape[i] > catarr->shape[i]) {
            itr->part_shape[i] = catarr->shape[i] - catarr->eshape[i] + catarr->pshape[i];
        } else {
            itr->part_shape[i] = catarr->pshape[i];
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
    return itr->cont >= itr->container->catarr->esize / itr->container->catarr->psize;
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
                                             iarray_iter_write_part_t **itr)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(itr);
    *itr = (iarray_iter_write_part_t*)ina_mem_alloc(sizeof(iarray_iter_write_part_t));
    INA_RETURN_IF_NULL(itr);
    caterva_dims_t shape = caterva_new_dims(container->dtshape->shape, container->dtshape->ndim);
    int err = caterva_update_shape(container->catarr, shape);
    if (err < 0) {
        return INA_ERROR(INA_ERR_FAILED);
    }
    (*itr)->ctx = ctx;
    (*itr)->container = container;
    (*itr)->part = (uint8_t *) ina_mem_alloc(container->catarr->psize * container->catarr->sc->typesize);
    (*itr)->part_index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));
    (*itr)->elem_index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));
    (*itr)->pointer = &(*itr)->part[0];
    (*itr)->part_shape = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));

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
    uint64_t B0 = itr->B0;
    uint64_t B1 = itr->B1;
    uint64_t M = itr->M;
    uint64_t N = itr->N;
    uint64_t K = itr->K;

    itr->cont++;

    uint64_t n, k, m;

    if (itr->container2->catarr->ndim == 1) {
        m = itr->cont / ((K/B1)) % (M/B0);
        k = itr->cont % (K/B1);

        itr->npart1 = (m * (K/B1) + k);
        itr->npart2 = k;

    } else {
        m = itr->cont / ((K/B1) * (N/B0)) % (M/B0);
        k = itr->cont % (K/B1);
        n = itr->cont / ((K/B1)) % (N/B0);

        itr->npart1 = (m * (K/B1) + k);
        itr->npart2 = (k * (N/B0) + n);
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
    uint64_t B0 = itr->B0;
    uint64_t B1 = itr->B1;
    uint64_t M = itr->M;
    uint64_t N = itr->N;
    uint64_t K = itr->K;

    if (itr->container2->catarr->ndim == 1) {
        return itr->cont >= (M/B0) * (K/B1);
    }

    return itr->cont >= (M/B0) * (N/B0) * (K/B1);
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
                                 uint64_t *bshape, iarray_iter_matmul_t **itr)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(c1);
    INA_VERIFY_NOT_NULL(c2);
    INA_VERIFY_NOT_NULL(bshape);
    INA_VERIFY_NOT_NULL(itr);

    // Verify that block shape is < than container shapes
    for (int i = 0; i < c1->dtshape->ndim; ++i) {
        if (c1->dtshape->shape[i] <= bshape[i]) {
            return INA_ERROR(INA_ERR_FAILED);
        }
    }
    for (int i = 0; i < c2->dtshape->ndim; ++i) {
        if (c2->dtshape->shape[i] <= bshape[c1->dtshape->ndim - 1 - i]) {
            return INA_ERROR(INA_ERR_FAILED);
        }
    }

    *itr = (iarray_iter_matmul_t*)ina_mem_alloc(sizeof(iarray_iter_matmul_t));
    INA_RETURN_IF_NULL(itr);
    (*itr)->ctx = ctx;
    (*itr)->container1 = c1;
    (*itr)->container2 = c2;
    (*itr)->B0 = bshape[0];
    (*itr)->B1 = bshape[1];
    if (c1->dtshape->shape[0] % bshape[0] == 0) {
        (*itr)->M = c1->dtshape->shape[0];
    } else {
        (*itr)->M = (c1->dtshape->shape[0] / bshape[0] + 1) * bshape[0];
    }
    if (c1->dtshape->shape[1] % bshape[1] == 0) {
        (*itr)->K = c1->dtshape->shape[1];
    } else {
        (*itr)->K = (c1->dtshape->shape[1] / bshape[1] + 1) * bshape[1];
    }

    if (c2->dtshape->shape[1] % bshape[0] == 0) {
        (*itr)->N = c2->dtshape->shape[1];
    } else {
        (*itr)->N = (c2->dtshape->shape[1] / bshape[0] + 1) * bshape[0];
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
    caterva_array_t *catarr = itr->container->catarr;

    itr->cont = 0;
    itr->cont_part = 0;
    itr->cont_part_elem = 0;

    itr->nelem = 0;

    itr->bsize = itr->container->catarr->psize;

    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        itr->index[i] = 0;
        itr->part_index[i] = 0;
        itr->bshape[i] = itr->container->catarr->pshape[i];
    }
    itr->pointer = &itr->part[0];

    blosc2_schunk_decompress_chunk(catarr->sc, 0, itr->part, catarr->psize * catarr->sc->typesize);
}

/*
 * Function: iarray_iter_read_next
 */

INA_API(ina_rc_t) iarray_iter_read_next(iarray_iter_read_t *itr)
{
    caterva_array_t *catarr = itr->container->catarr;
    int ndim = catarr->ndim;

    // check if a part is filled totally and append it

    if (itr->cont_part_elem  == itr->bsize - 1) {
        if(itr->cont == catarr->size - 1) {
            itr->cont++;
            return INA_SUCCESS;
        }
        int err = blosc2_schunk_decompress_chunk(catarr->sc, (int) itr->cont_part + 1, itr->part, catarr->psize * catarr->sc->typesize);
        if (err < 0) {
            return INA_ERROR(INA_ERR_FAILED);
        }
        itr->cont_part_elem = 0;
        itr->cont_part += 1;
        uint64_t inc = 1;
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
    } else {
        itr->cont_part_elem += 1;
    }

    // jump to the next element
    itr->cont += 1;

    uint64_t ind_part_elem[IARRAY_DIMENSION_MAX];
    uint64_t cont_pointer = 0;

    uint64_t inc = 1;
    uint64_t inc_s = 1;
    uint64_t inc_p = 1;

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
    itr->pointer = (void *)&(itr->part)[cont_pointer * catarr->sc->typesize];

    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_read_finished
 */

INA_API(int) iarray_iter_read_finished(iarray_iter_read_t *itr)
{
    return itr->cont >= itr->container->catarr->size;
}

/*
 * Function: iarray_iter_read_value
 */

INA_API(void) iarray_iter_read_value(iarray_iter_read_t *itr, iarray_iter_read_value_t *val)
{
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

    *itr = (iarray_iter_write_t*)ina_mem_alloc(sizeof(iarray_iter_write_t));
    INA_RETURN_IF_NULL(itr);
    caterva_dims_t shape = caterva_new_dims(container->dtshape->shape, container->dtshape->ndim);
    int err = caterva_update_shape(container->catarr, shape);
    if (err < 0) {
        return INA_ERROR(INA_ERR_FAILED);
    }
    (*itr)->ctx = ctx;
    (*itr)->container = container;
    (*itr)->part = (uint8_t *) ina_mem_alloc(container->catarr->psize * container->catarr->sc->typesize);
    (*itr)->index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));
    (*itr)->part_index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));
    (*itr)->bshape = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));


    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_read_free
 */

INA_API(void) iarray_iter_read_free(iarray_iter_read_t *itr)
{
    ina_mem_free(itr->index);
    ina_mem_free(itr->part);
    ina_mem_free(itr->part_index);
    ina_mem_free(itr->bshape);
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
    for (int i = 0; i <IARRAY_DIMENSION_MAX; ++i) {
        itr->elem_index[i] = 0;
        itr->block_index[i] = 0;
    }
    itr->cont = 0;

    uint64_t stop_[IARRAY_DIMENSION_MAX];
    uint64_t buflen = 1;

    itr->block_size = 1;
    for (int i = 0; i < itr->container->dtshape->ndim; ++i) {
        itr->block_shape[i] = itr->shape[i];
        itr->block_size *= itr->block_shape[i];
        stop_[i] = itr->elem_index[i] + itr->shape[i];
        buflen *= itr->shape[i];
    }

    INA_MUST_SUCCEED(iarray_slice_buffer(itr->ctx, itr->container, (int64_t *) itr->elem_index,
                     (int64_t *) stop_, itr->part, buflen * sizeof(double)));
}

/*
 * Function: iarray_iter_read_block_next
 */

INA_API(ina_rc_t) iarray_iter_read_block_next(iarray_iter_read_block_t *itr)
{
    uint8_t ndim = itr->container->dtshape->ndim;
    caterva_array_t *catarr = itr->container->catarr;
    itr->cont += 1;

    uint64_t aux[IARRAY_DIMENSION_MAX];
    for (int i = ndim - 1; i >= 0; --i) {
        if (catarr->shape[i] % itr->shape[i] == 0) {
            aux[i] = catarr->shape[i] / itr->shape[i];
        } else {
            aux[i] = catarr->shape[i] / itr->shape[i] + 1;
        }
    }

    uint64_t start_[IARRAY_DIMENSION_MAX];

    uint64_t inc = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        start_[i] = itr->cont % (aux[i] * inc) / inc;
        itr->block_index[i] = start_[i];
        start_[i] *= itr->shape[i];
        itr->elem_index[i] = start_[i];
        inc *= aux[i];
    }

    uint64_t stop_[IARRAY_DIMENSION_MAX];
    uint64_t buflen = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if(start_[i] + itr->shape[i] <= catarr->shape[i]) {
            stop_[i] = start_[i] + itr->shape[i];
        } else {
            stop_[i] = catarr->shape[i];
        }
        itr->block_shape[i] = stop_[i] - start_[i];
        itr->block_size *= itr->block_shape[i];
        buflen *= itr->shape[i];
    }

    INA_MUST_SUCCEED(iarray_slice_buffer(itr->ctx, itr->container, (int64_t *) start_,
                     (int64_t *) stop_, itr->part, buflen * catarr->sc->typesize));

    return INA_SUCCESS;
}

/*
 * Function: iarray_iter_read_block_finished
 */

INA_API(int) iarray_iter_read_block_finished(iarray_iter_read_block_t *itr)
{
    uint64_t size = 1;
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
                                             iarray_iter_read_block_t **itr, uint64_t *blockshape)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(itr);
    *itr = (iarray_iter_read_block_t*) ina_mem_alloc(sizeof(iarray_iter_read_block_t));
    INA_RETURN_IF_NULL(itr);

    (*itr)->ctx = ctx;
    (*itr)->container = container;
    (*itr)->shape = (uint64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(uint64_t));
    (*itr)->block_shape = (uint64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(uint64_t));
    (*itr)->block_index = (uint64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(uint64_t));
    (*itr)->elem_index = (uint64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(uint64_t));

    uint64_t size = 1;
    for (int i = 0; i < (*itr)->container->dtshape->ndim; ++i) {
        (*itr)->shape[i] = blockshape[i];
        size *= (*itr)->shape[i];
    }
    if ((*itr)->container->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        (*itr)->part = ina_mem_alloc(size * sizeof(double));
    } else {
        (*itr)->part = ina_mem_alloc(size * sizeof(float));
    }
    (*itr)->pointer = &((*itr)->part[0]);
    return INA_SUCCESS;
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
    ina_mem_free(itr);
}
