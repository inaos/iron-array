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
 * Function: _update_itr_index (private)
 * -------------------------------------
 *   Update the index and the nelem of an iterator
 *
 *   itr: an iterator
 */

void _update_itr_index(iarray_itr_t *itr)
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

INA_API(void) iarray_itr_init(iarray_itr_t *itr)
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
 *   Compute the next iterator element
 *
 *   itr: an iterator
 */

INA_API(void) iarray_itr_next(iarray_itr_t *itr)
{
    caterva_array_t *catarr = itr->container->catarr;
    int ndim = catarr->ndim;

    // jump to the next element
    itr->cont += 1;
    _update_itr_index(itr);

    // check if the element is out of the container (pad positions)
    uint64_t aux_inc[CATERVA_MAXDIM];
    aux_inc[ndim - 1] = 1;
    for (int m = ndim - 2; m >= 0; --m) {
        aux_inc[m] = catarr->pshape[m + 1] * aux_inc[m + 1];
    }
    for (int l = ndim - 1; l >= 0; --l) {
        if (itr->index[l] >= catarr->shape[l]) {
            itr->cont += (catarr->eshape[l] - catarr->shape[l]) * aux_inc[l];
            _update_itr_index(itr);
        }
    }

    // check if a chunk is filled totally and append it
    if (itr->cont % catarr->psize == 0) {
        blosc2_schunk_append_buffer(catarr->sc, itr->part, catarr->psize * catarr->sc->typesize);
        memset(itr->part, 0, catarr->psize * catarr->sc->typesize);
    }

    _update_itr_index(itr);
}

/*
 * Function: iarray_itr_finished
 * -----------------------------
 *   Check if the iterator is finished
 *
 *   itr: an iterator
 *
 *   returns: 1 if iter is finished or 0 if not
 */

INA_API(int) iarray_itr_finished(iarray_itr_t *itr)
{
    return itr->cont >= itr->container->catarr->esize;
}

/*
 * Function: iarray_itr_value
 * ------------------------
 *   Create a new iterator
 *
 *   itr: an iterator
 *   val: a struct where data needed by the user is stored
 *
 *   returns: an error code
 */

INA_API(ina_rc_t) iarray_itr_value(iarray_itr_t *itr, iarray_itr_value_t *val)
{
    val->pointer = itr->pointer;
    val->index = itr->index;
    val->nelem = itr->nelem;

    return 0;
}

/*
 * Function: iarray_itr_new
 * ------------------------
 *   Create a new iterator
 *
 *   container: the container used in the iterator
 *   itr: an iterator
 *
 *   returns: an error code
 */

INA_API(ina_rc_t) iarray_itr_new(iarray_context_t *ctx, iarray_container_t *container, iarray_itr_t **itr)
{
    *itr = (iarray_itr_t*)ina_mem_alloc(sizeof(iarray_itr_t));
    INA_RETURN_IF_NULL(itr);
    caterva_update_shape(container->catarr, *container->shape);
    (*itr)->container = container;
    (*itr)->part = (uint8_t *) ina_mem_alloc(container->catarr->psize * container->catarr->sc->typesize);

    (*itr)->index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));

    return 0;
}

/*
 * Function: iarray_itr_free
 * -------------------------
 *   Free an iterator structure
 *
 *   itr: an iterator
 *
 *   returns: an error code
 */

INA_API(ina_rc_t) iarray_itr_free(iarray_context_t *ctx, iarray_itr_t *itr)
{
    ina_mem_free(itr->index);
    ina_mem_free(itr->part);
    ina_mem_free(itr);
    return 0;
}

// CHUNK BY CHUNK ITERATOR

/*
 * Function: _update_itr_index (private)
 * -------------------------------------
 *   Update the index and the nelem of an iterator
 *
 *   itr: an iterator
 */

void _update_itr_chunk_index(iarray_itr_t *itr)
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
 * -------------------------
 *   Set the iterator values to the first element
 *
 *   itr: an iterator
 */

INA_API(void) iarray_itr_chunk_init(iarray_itr_chunk_t *itr)
{
    itr->cont = 0;
    memset(itr->part, 0, itr->container->catarr->psize * itr->container->catarr->sc->typesize);
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        itr->index[i] = 0;
        itr->size = itr->container->catarr->psize;
        itr->shape[i] = itr->container->dtshape->partshape[i];
    }
}

/*
 * Function: iarray_itr_next
 * -------------------------
 *   Compute the next iterator element
 *
 *   itr: an iterator
 */

INA_API(void) iarray_itr_chunk_next(iarray_itr_chunk_t *itr)
{
    uint64_t psizeb = itr->size * itr->container->catarr->sc->typesize;
    if ( itr->size == itr->container->catarr->psize ) {
        //blosc2_schunk_append_buffer(itr->container->catarr->sc, itr->part, psizeb);
    } else {
        uint8_t *part_aux = malloc(psizeb);

        uint64_t ii[CATERVA_MAXDIM];

        for (ii[0] = 0; ii[0] < itr->shape[0]; ++ii[0]) {
            for (ii[1] = 0; ii[1] < itr->shape[1]; ++ii[1]) {
                for (ii[2] = 0; ii[2] < itr->shape[2]; ++ii[2]) {
                    for (ii[3] = 0; ii[3] < itr->shape[3]; ++ii[3]) {
                        for (ii[4] = 0; ii[4] < itr->shape[4]; ++ii[4]) {
                            for (ii[5] = 0; ii[5] < itr->shape[5]; ++ii[5]) {
                                for (ii[6] = 0; ii[6] < itr->shape[6]; ++ii[6]) {

                                    uint64_t aux_p = 0;
                                    uint64_t itr_p = 0;

                                    //memcpy(&part_aux[aux_p], &(itr->part)[itr_p], itr->shape[7]);
                                }
                            }
                        }
                    }
                }
            }
        }
        //blosc2_schunk_append_buffer(itr->container->catarr->sc, itr->part, psizeb);
    }

    caterva_array_t *catarr = itr->container->catarr;
    int ndim = catarr->ndim;

    // jump to the next element
    itr->cont += 1;



}

/*
 * Function: iarray_itr_chunk_finished
 * -----------------------------
 *   Check if the iterator is finished
 *
 *   itr: an iterator
 *
 *   returns: 1 if iter is finished or 0 if not
 */

INA_API(int) iarray_itr_chunk_finished(iarray_itr_chunk_t *itr)
{
    return itr->cont >= itr->container->catarr->esize / itr->container->catarr->psize;
}

/*
 * Function: iarray_itr_value
 * ------------------------
 *   Create a new iterator
 *
 *   itr: an iterator
 *   val: a struct where data needed by the user is stored
 *
 *   returns: an error code
 */

INA_API(ina_rc_t) iarray_itr_chunk_value(iarray_itr_chunk_t *itr, iarray_itr_chunk_value_t *val)
{
    val->pointer = itr->pointer;
    val->index = itr->index;
    val->nelem = itr->cont;

    return 0;
}

/*
 * Function: iarray_itr_chunk_new
 * ------------------------
 *   Create a new iterator
 *
 *   container: the container used in the iterator
 *   itr: an iterator
 *
 *   returns: an error code
 */

INA_API(ina_rc_t) iarray_itr_chunk_new(iarray_context_t *ctx, iarray_container_t *container, iarray_itr_chunk_t **itr)
{
    *itr = (iarray_itr_chunk_t*)ina_mem_alloc(sizeof(iarray_itr_chunk_t));
    INA_RETURN_IF_NULL(itr);
    caterva_update_shape(container->catarr, *container->shape);
    (*itr)->container = container;
    (*itr)->part = (uint8_t *) ina_mem_alloc(container->catarr->psize * container->catarr->sc->typesize);

    (*itr)->index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));
    (*itr)->pointer = &(*itr)->part[0];
    (*itr)->shape = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));

    return 0;
}

/*
 * Function: iarray_itr_chunk_free
 * -------------------------
 *   Free an iterator structure
 *
 *   itr: an iterator
 *
 *   returns: an error code
 */

INA_API(ina_rc_t) iarray_itr_chunk_free(iarray_context_t *ctx, iarray_itr_chunk_t *itr)
{
    ina_mem_free(itr->index);
    ina_mem_free(itr->shape);
    ina_mem_free(itr->part);
    ina_mem_free(itr);
    return 0;
}

// MATMUL ITERATOR

/*
 * Function: iarray_itr_matmul_init
 * --------------------------------
 *   Set the iterator values to the first element
 *
 *   itr: an iterator
 */

ina_rc_t iarray_itr_matmul_init(iarray_itr_matmul_t *itr)
{
    itr->cont = 0;
    itr->nchunk1 = 0;
    itr->nchunk2 = 0;
    return 0;
}

/*
 * Function: iarray_itr_matmul_next
 * --------------------------------
 *   Compute the next iterator element
 *
 *   itr: an iterator
 */

ina_rc_t iarray_itr_matmul_next(iarray_itr_matmul_t *itr)
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

    return 0;
}

/*
 * Function: iarray_itr_matmul_finished
 * ------------------------------------
 *   Check if the iterator is finished
 *
 *   itr: an iterator
 *
 *   returns: 1 if iter is finished or 0 if not
 */

int iarray_itr_matmul_finished(iarray_itr_matmul_t *itr)
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
 *   Free an iterator structure
 *
 *   itr: an iterator
 *
 *   returns: an error code
 */

ina_rc_t iarray_itr_matmul_new(iarray_context_t *ctx, iarray_container_t *c1, iarray_container_t *c2, iarray_itr_matmul_t **itr)
{
    *itr = (iarray_itr_matmul_t*)ina_mem_alloc(sizeof(iarray_itr_matmul_t));
    INA_RETURN_IF_NULL(itr);
    (*itr)->container1 = c1;
    (*itr)->container2 = c2;

    return 0;
}

/*
 * Function: iarray_itr_matmul_free
 * --------------------------------
 *   Free an iterator structure
 *
 *   itr: an iterator
 *
 *   returns: an error code
 */

ina_rc_t iarray_itr_matmul_free(iarray_context_t *ctx, iarray_itr_matmul_t *itr)
{
    ina_mem_free(itr);
    return 0;
}
