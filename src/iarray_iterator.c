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
 * ELEMENT BY ELEMENT ITERATOR
 *
 * Next functions are used to fill an iarray container element by element
 */

/*
 * Function: _update_itr_index (private)
 * -------------------------------------
 *   (internal) Update the index and the nelem of an iterator
 *
 *   itr: an iterator
 */

void _update_itr_index(iarray_context_t *ctx, iarray_itr_t *itr)
{
    caterva_array_t *catarr = itr->container->catarr;

    int ndim = catarr->ndim;

    uint64_t cont2 = itr->cont % catarr->psize; // element position in the chunk

    // set element index (in the chunk)
    itr->index[ndim - 1] = cont2 % catarr->pshape[ndim-1];
    uint64_t inc = catarr->pshape[ndim - 1];
    for (int i = ndim - 2; i >= 0; --i) {
        itr->index[i] = cont2 % (inc * catarr->pshape[i]) / inc;
        inc *= catarr->pshape[i];
    }

    // set element index (in entire container)
    uint64_t nchunk = itr->cont / catarr->psize;
    uint64_t aux_nchunk[CATERVA_MAXDIM];
    aux_nchunk[ndim - 1] = catarr->eshape[ndim - 1] / catarr->pshape[ndim - 1];
    for (int k = ndim - 2; k >= 0; --k) {
        aux_nchunk[k] = aux_nchunk[k + 1] * (catarr->eshape[k] / catarr->pshape[k]);
    }
    for (int j = 0; j < ndim; ++j) {
        itr->index[j] += nchunk % aux_nchunk[j] / (aux_nchunk[j] / (catarr->eshape[j] / catarr->pshape[j])) * catarr->pshape[j];
    }

    // set element pointer
    if (itr->container->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        itr->pointer = (void *)&((double*)itr->part)[cont2];
    } else{
        itr->pointer = (void *)&((float*)itr->part)[cont2];
    }

    // set element nelem
    itr->nelem = 0;
    inc = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        itr->nelem += itr->index[i] * inc;
        inc *= itr->container->dtshape->shape[i];
    }
}

/*
 * Function: iarray_itr_init
 * -------------------------
 *   Set the iterator values to the first element
 *
 *   itr: an iterator
 */

INA_API(void) iarray_itr_init(iarray_context_t *ctx, iarray_itr_t *itr)
{
    itr->cont = 0;
    itr->nelem = 0;
    memset(itr->part, 0, itr->container->catarr->psize * itr->container->catarr->sc->typesize);
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        itr->index[i] = 0;
    }
    itr->pointer = &itr->part[0];
}

/*
 * Function: iarray_itr_next
 * -------------------------
 *   Compute the next iterator element nad update the iterator with it
 *
 *   itr: an iterator
 */

INA_API(ina_rc_t) iarray_itr_next(iarray_context_t *ctx, iarray_itr_t *itr)
{
    caterva_array_t *catarr = itr->container->catarr;
    int ndim = catarr->ndim;

    // jump to the next element
    itr->cont += 1;
    _update_itr_index(ctx, itr);

    // check if the element is out of the container (pad positions)
    uint64_t aux_inc[CATERVA_MAXDIM];
    aux_inc[ndim - 1] = 1;
    for (int m = ndim - 2; m >= 0; --m) {
        aux_inc[m] = catarr->pshape[m + 1] * aux_inc[m + 1];
    }
    for (int l = ndim - 1; l >= 0; --l) {
        if (itr->index[l] >= catarr->shape[l]) {
            itr->cont += (catarr->eshape[l] - catarr->shape[l]) * aux_inc[l];
            _update_itr_index(ctx, itr);
        }
    }

    // check if a chunk is filled totally and append it
    if (itr->cont % catarr->psize == 0) {
        int err = blosc2_schunk_append_buffer(catarr->sc, itr->part, catarr->psize * catarr->sc->typesize);
        if (err < 0) {
            return INA_ERR_FAILED;
        }
        memset(itr->part, 0, catarr->psize * catarr->sc->typesize);
    }

    _update_itr_index(ctx, itr);
    return INA_SUCCESS;
}

/*
 * Function: iarray_itr_finished
 * -----------------------------
 *   Check if the iteration over a container is finished
 *
 *   itr: an iterator
 *
 *   return: 1 if iter is finished or 0 if not
 */

INA_API(int) iarray_itr_finished(iarray_context_t *ctx, iarray_itr_t *itr)
{
    return itr->cont >= itr->container->catarr->esize;
}

/*
 * Function: iarray_itr_value
 * ------------------------
 *   Store in `val` some variables of the actual element
 *
 *   itr: an iterator
 *   val: a struct where data needed by the user is stored
 *     index: position in coord where the element is placed in the container
 *     nelem: if the container is row-wise flatten, `nelem` is the element position in the container
 *     pointer: pointer to element position in memory. It's used to copy the element into the container
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(void) iarray_itr_value(iarray_context_t *ctx, iarray_itr_t *itr, iarray_itr_value_t *val)
{
    val->pointer = itr->pointer;
    val->index = itr->index;
    val->nelem = itr->nelem;
}

/*
 * Function: iarray_itr_new
 * ------------------------
 *   Create a new iterator
 *
 *   container: the container used in the iterator
 *   itr: an iterator pointer
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(ina_rc_t) iarray_itr_new(iarray_context_t *ctx, iarray_container_t *container, iarray_itr_t **itr)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(itr);

    *itr = (iarray_itr_t*)ina_mem_alloc(sizeof(iarray_itr_t));
    INA_RETURN_IF_NULL(itr);
    caterva_dims_t shape = caterva_new_dims(container->dtshape->shape, container->dtshape->ndim);
    int err = caterva_update_shape(container->catarr, shape);
    if (err < 0) {
        return INA_ERR_FAILED;
    }
    (*itr)->container = container;
    (*itr)->part = (uint8_t *) ina_mem_alloc(container->catarr->psize * container->catarr->sc->typesize);
    (*itr)->index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));

    return INA_SUCCESS;
}

/*
 * Function: iarray_itr_free
 * -------------------------
 *   Free an iterator structure
 *
 *   itr: an iterator
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(void) iarray_itr_free(iarray_context_t *ctx, iarray_itr_t *itr)
{
    ina_mem_free(itr->index);
    ina_mem_free(itr->part);
    ina_mem_free(itr);
}

/*
 * CHUNK BY CHUNK ITERATOR
 *
 * Unlike the previous, the next collection of functions are used to fill an iarray container chunk by chunk
 */

/*
 * Function: _update_itr_index (private)
 * -------------------------------------
 *   Update the index and the nelem of an iterator
 *
 *   itr: an iterator
 */

void _update_itr_chunk_index(iarray_context_t *ctx, iarray_itr_t *itr)
{
    caterva_array_t *catarr = itr->container->catarr;

    int ndim = catarr->ndim;

    uint64_t cont2 = itr->cont % catarr->psize; // element position in the chunk

    // set element index (in the chunk)
    itr->index[ndim - 1] = cont2 % catarr->pshape[ndim-1];
    uint64_t inc = catarr->pshape[ndim - 1];
    for (int i = ndim - 2; i >= 0; --i) {
        itr->index[i] = cont2 % (inc * catarr->pshape[i]) / inc;
        inc *= catarr->pshape[i];
    }

    // set element index (in entire container)
    uint64_t nchunk = itr->cont / catarr->psize;
    uint64_t aux_nchunk[CATERVA_MAXDIM];
    aux_nchunk[ndim - 1] = catarr->eshape[ndim - 1] / catarr->pshape[ndim - 1];
    for (int k = ndim - 2; k >= 0; --k) {
        aux_nchunk[k] = aux_nchunk[k + 1] * (catarr->eshape[k] / catarr->pshape[k]);
    }
    for (int j = 0; j < ndim; ++j) {
        itr->index[j] += nchunk % aux_nchunk[j] / (aux_nchunk[j] / (catarr->eshape[j] / catarr->pshape[j])) * catarr->pshape[j];
    }

    // set element pointer
    if (itr->container->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        itr->pointer = (void *)&((double*)itr->part)[cont2];
    } else{
        itr->pointer = (void *)&((float*)itr->part)[cont2];
    }

    // set element nelem
    itr->nelem = 0;
    inc = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        itr->nelem += itr->index[i] * inc;
        inc *= itr->container->dtshape->shape[i];
    }
}

/*
 * Function: iarray_itr_chunk_init
 * -------------------------------
 *   Set the iterator values to the first element
 *
 *   itr: an iterator
 */

INA_API(void) iarray_itr_chunk_init(iarray_context_t *ctx, iarray_itr_chunk_t *itr)
{
    itr->cont = 0;
    memset(itr->part, 0, itr->container->catarr->psize * itr->container->catarr->sc->typesize);
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        itr->index[i] = 0;
        itr->shape[i] = itr->container->dtshape->pshape[i];
    }
    itr->size = itr->container->catarr->psize;
}

/*
 * Function: iarray_itr_chunk_next
 * -------------------------------
 *   Update the iterator to next element
 *
 *   itr: an iterator
 */

INA_API(ina_rc_t) iarray_itr_chunk_next(iarray_context_t *ctx, iarray_itr_chunk_t *itr)
{
    caterva_array_t *catarr = itr->container->catarr;
    int ndim = catarr->ndim;

    uint64_t psizeb = itr->size * catarr->sc->typesize;

    // check if the chunk should be padded with 0s
    if ( itr->size == catarr->psize ) {
        int err = blosc2_schunk_append_buffer(catarr->sc, itr->part, psizeb);
        if (err < 0) {
            return INA_ERR_FAILED;
        }
    } else {
        uint8_t *part_aux = malloc(catarr->psize * catarr->sc->typesize);

        //reverse shape
        uint64_t shaper[CATERVA_MAXDIM];
        for (int i = 0; i < CATERVA_MAXDIM; ++i) {
            if(i >= CATERVA_MAXDIM - ndim) {
                shaper[i] = itr->shape[i - CATERVA_MAXDIM + ndim];
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
            return INA_ERR_FAILED;
        }

        free(part_aux);
    }
    itr->cont += 1;

    //update_index
    itr->index[ndim - 1] = itr->cont % (catarr->eshape[ndim - 1] / catarr->pshape[ndim - 1]);
    uint64_t inc = catarr->eshape[ndim - 1] / catarr->pshape[ndim - 1];

    for (int i = ndim - 2; i >= 0; --i) {
        itr->index[i] = itr->cont % (inc * catarr->eshape[i] / catarr->pshape[i]) / (inc);
        itr->el_index[i] = itr->index[i] * catarr->pshape[i];
        inc *= catarr->eshape[i] / catarr->pshape[i];
    }

    //calculate the buffer size
    itr->size = 1;
    for (int i = 0; i < ndim; ++i) {
        if ((itr->index[i] + 1) * catarr->pshape[i] > catarr->shape[i]) {
            itr->shape[i] = catarr->shape[i] - catarr->eshape[i] + catarr->pshape[i];
        } else {
            itr->shape[i] = catarr->pshape[i];
        }
        itr->size *= itr->shape[i];
    }

    return INA_SUCCESS;
}

/*
 * Function: iarray_itr_chunk_finished
 * -----------------------------------
 *   Check if the iterator is finished
 *
 *   itr: an iterator
 *
 *   return: 1 if iter is finished or 0 if not
 */

INA_API(int) iarray_itr_chunk_finished(iarray_context_t *ctx, iarray_itr_chunk_t *itr)
{
    return itr->cont >= itr->container->catarr->esize / itr->container->catarr->psize;
}

/*
 * Function: iarray_itr_chunk_value
 * --------------------------------
 *   Store in `val` parameter some variables of the actual chunk
 *
 *   itr: an iterator
 *   val: a struct where data needed by the user is stored
 *     index: position in coord where the chunk is placed in the container
 *     nelem: if the chunks are row-wise listed, `nelem` is the chunk position in this list
 *     el_index: position in coord where the first element of the chunk is placed in the container
 *     shape: is the actual chunk shape. It should be used to compute the chunk size
 *     pointer: pointer to the first chunk element position in memory. It's used to copy the chunk into the container
 *
 *   return: INA_SUCCESS or an error code
 */

INA_API(void) iarray_itr_chunk_value(iarray_context_t *ctx, iarray_itr_chunk_t *itr, iarray_itr_chunk_value_t *val)
{
    val->pointer = itr->pointer;
    val->index = itr->index;
    val->el_index = itr->el_index;
    val->nelem = itr->cont;
    val->shape = itr->shape;
}

/*
 * Function: iarray_itr_chunk_new
 * ------------------------------
 *   Create a new iterator
 *
 *   container: the container used in the iterator
 *   itr: an iterator
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(ina_rc_t) iarray_itr_chunk_new(iarray_context_t *ctx, iarray_container_t *container, iarray_itr_chunk_t **itr)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(itr);
    *itr = (iarray_itr_chunk_t*)ina_mem_alloc(sizeof(iarray_itr_chunk_t));
    INA_RETURN_IF_NULL(itr);
    caterva_dims_t shape = caterva_new_dims(container->dtshape->shape, container->dtshape->ndim);
    int err = caterva_update_shape(container->catarr, shape);
    if (err < 0) {
        return INA_ERR_FAILED;
    }
    (*itr)->container = container;
    (*itr)->part = (uint8_t *) ina_mem_alloc(container->catarr->psize * container->catarr->sc->typesize);
    (*itr)->index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));
    (*itr)->el_index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));
    (*itr)->pointer = &(*itr)->part[0];
    (*itr)->shape = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));

    return INA_SUCCESS;
}

/*
 * Function: iarray_itr_chunk_free
 * -------------------------------
 *   Free an iterator structure
 *
 *   itr: an iterator
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(void) iarray_itr_chunk_free(iarray_context_t *ctx, iarray_itr_chunk_t *itr)
{
    ina_mem_free(itr->index);
    ina_mem_free(itr->el_index);
    ina_mem_free(itr->shape);
    ina_mem_free(itr->part);
    ina_mem_free(itr);
}

/*
 * MATMUL ITERATOR
 *
 * Internal iterator used to perform easily matrix-matrix or vector-matrix multiplications by blocks
 *
 */


/*
 * Function: iarray_itr_matmul_init
 * --------------------------------
 *   Set the iterator values to the first element
 *
 *   itr: an iterator
 */

void _iarray_itr_matmul_init(iarray_context_t *ctx, iarray_itr_matmul_t *itr)
{
    itr->cont = 0;
    itr->nchunk1 = 0;
    itr->nchunk2 = 0;
}

/*
 * Function: iarray_itr_matmul_next
 * --------------------------------
 *   Update the block to be used of each container
 *
 *   itr: an iterator
 */

void _iarray_itr_matmul_next(iarray_context_t *ctx, iarray_itr_matmul_t *itr)
{
    uint64_t P = itr->container1->catarr->pshape[0];
    uint64_t M = itr->container1->catarr->eshape[0];
    uint64_t N = itr->container2->catarr->eshape[1];
    uint64_t K = itr->container1->catarr->eshape[1];

    itr->cont++;

    uint64_t n, k, m;

    if (itr->container2->catarr->ndim == 1) {
        m = itr->cont / ((K/P)) % (M/P);
        k = itr->cont % (K/P);

        itr->nchunk1 = (m * (K/P) + k);
        itr->nchunk2 = k;

    } else {
        m = itr->cont / ((K/P) * (N/P)) % (M/P);
        k = itr->cont % (K/P);
        n = itr->cont / ((K/P)) % (N/P);

        itr->nchunk1 = (m * (K/P) + k);
        itr->nchunk2 = (k * (N/P) + n);
    }
}

/*
 * Function: iarray_itr_matmul_finished
 * ------------------------------------
 *   Check if the iterator is finished
 *
 *   itr: an iterator
 *
 *   return: 1 if iter is finished or 0 if not
 */

int _iarray_itr_matmul_finished(iarray_context_t *ctx, iarray_itr_matmul_t *itr)
{
    uint64_t P = itr->container1->catarr->pshape[0];
    uint64_t M = itr->container1->catarr->eshape[0];
    uint64_t N = itr->container2->catarr->eshape[1];
    uint64_t K = itr->container1->catarr->eshape[1];

    if (itr->container1->catarr->ndim == 1) {
        return itr->cont >= (M/P) * (N/P);
    }

    if (itr->container2->catarr->ndim == 1) {
        return itr->cont >= (M/P) * (K/P);
    }

    return itr->cont >= (M/P) * (N/P) * (K/P);
}

/*
 * Function: iarray_itr_matmul_new
 * ------------------------
 *   Create a matmul iterator
 *
 *   itr: an iterator
 *
*   return: INA_SUCCESS or an error code
 */

ina_rc_t _iarray_itr_matmul_new(iarray_context_t *ctx, iarray_container_t *c1, iarray_container_t *c2,
                                iarray_itr_matmul_t **itr)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(c1);
    INA_VERIFY_NOT_NULL(c2);
    INA_VERIFY_NOT_NULL(itr);

    *itr = (iarray_itr_matmul_t*)ina_mem_alloc(sizeof(iarray_itr_matmul_t));
    INA_RETURN_IF_NULL(itr);
    (*itr)->container1 = c1;
    (*itr)->container2 = c2;

    return INA_SUCCESS;
}

/*
 * Function: iarray_itr_matmul_free
 * --------------------------------
 *   Free an iterator structure
 *
 *   itr: an iterator
 *
*   return: INA_SUCCESS or an error code
 */

void _iarray_itr_matmul_free(iarray_context_t *ctx, iarray_itr_matmul_t *itr)
{
    ina_mem_free(itr);
}

/*
 * READ ITERATOR BY BLOCKS
 *
 * Iterator that allows read an iarray container by blocks (the blocksize is specified by the user)
 *
 */

/*
 * Function: iarray_itr_chunk_read_init
 * ------------------------------
 *   Set the iterator values to the first element
 *
 *   itr: an iterator
 */

INA_API(void) iarray_itr_chunk_read_init(iarray_context_t *ctx, iarray_itr_read_t *itr)
{
    itr->size = 1;
    for (int i = 0; i < itr->container->dtshape->ndim; ++i) {
        itr->index[i] = 0;
    }
    itr->cont = 0;

    //ToDo: Implement a iarray function that stores slicing result into a buffer
    uint64_t stop_[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        itr->pshape[i] = itr->shape[i];
        stop_[i] = itr->index[i] + itr->shape[i];
    }
    iarray_slice_buffer(ctx, itr->container, itr->index, stop_, itr->part);
}

/*
 * Function: iarray_itr_chunk_read_next
 * ------------------------------
 *   Update the iterator to next element
 *
 *   itr: an iterator
 */

INA_API(ina_rc_t) iarray_itr_chunk_read_next(iarray_context_t *ctx, iarray_itr_read_t *itr)
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
        start_[i] *= itr->shape[i];
        itr->index[i] = start_[i];
        inc *= aux[i];
    }

    uint64_t stop_[IARRAY_DIMENSION_MAX];

    for (int i = ndim - 1; i >= 0; --i) {
        if(start_[i] + itr->shape[i] <= catarr->shape[i]) {
            stop_[i] = start_[i] + itr->shape[i];
        } else {
            stop_[i] = catarr->shape[i];
        }
        itr->pshape[i] = stop_[i] - start_[i];
    }

    iarray_slice_buffer(ctx, itr->container, start_, stop_, itr->part);

    return INA_SUCCESS;
}

/*
 * Function: iarray_itr_chunk_read_finished
 * ----------------------------------
 *   Check if the iterator is finished
 *
 *   itr: an iterator
 *
 *   return: 1 if iter is finished or 0 if not
 */

INA_API(int) iarray_itr_chunk_read_finished(iarray_context_t *ctx, iarray_itr_read_t *itr)
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
 * Function: iarray_itr_chunk_read_value
 * -------------------------------
 *   Store in `val` parameter some variables of the actual block
 *
 *   itr: an iterator
 *   val: a struct where data needed by the user is stored
 *     index: position in coord where the chunk is placed in the container
 *     nelem: if the chunks are row-wise listed, `nelem` is the chunk position in this list
 *     el_index: position in coord where the first element of the chunk is placed in the container
 *     shape: is the actual chunk shape. It should be used to compute the chunk size
 *     pointer: pointer to the first chunk element position in memory. It's used to copy the chunk into the container
 *
 *   return: INA_SUCCESS or an error code
 */

INA_API(void) iarray_itr_chunk_read_value(iarray_context_t *ctx, iarray_itr_read_t *itr, iarray_itr_read_value_t *val)
{
    val->index = itr->index;
    val->shape = itr->pshape;
    val->pointer = itr->pointer;
}

/*
 * Function: iarray_itr_chunk_read_new
 * -----------------------------
 *   Create a new iterator
 *
 *   container: the container used in the iterator
 *   itr: an iterator
 *   blockshape: shape of each block
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(ina_rc_t) iarray_itr_chunk_read_new(iarray_context_t *ctx, iarray_container_t *container,
                                            iarray_itr_read_t **itr, uint64_t *blockshape)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);
    INA_VERIFY_NOT_NULL(itr);
    *itr = (iarray_itr_read_t*) ina_mem_alloc(sizeof(iarray_itr_read_t));
    INA_RETURN_IF_NULL(itr);

    (*itr)->container = container;
    (*itr)->size = 1;
    (*itr)->shape = (uint64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(uint64_t));
    (*itr)->pshape = (uint64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(uint64_t));
    (*itr)->index = (uint64_t *) ina_mem_alloc(IARRAY_DIMENSION_MAX * sizeof(uint64_t));

    for (int i = 0; i < (*itr)->container->dtshape->ndim; ++i) {
        (*itr)->shape[i] = blockshape[i];
        (*itr)->size *= (*itr)->shape[i];
    }
    if ((*itr)->container->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        (*itr)->part = ina_mem_alloc((*itr)->size * sizeof(double));
    } else {
        (*itr)->part = ina_mem_alloc((*itr)->size * sizeof(float));
    }
    (*itr)->pointer = &((*itr)->part[0]);
    return INA_SUCCESS;
}

/*
 * Function: iarray_itr_chunk_read_free
 * -------------------------------
 *   Free an iterator structure
 *
 *   itr: an iterator
 *
*   return: INA_SUCCESS or an error code
 */

INA_API(void) iarray_itr_chunk_read_free(iarray_context_t *ctx, iarray_itr_read_t *itr)
{
    ina_mem_free(itr->shape);
    ina_mem_free(itr->pshape);
    ina_mem_free(itr->index);
    ina_mem_free(itr->part);
    ina_mem_free(itr);
}
