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
 *
 */

void _update_itr_index(iarray_itr_t *itr) 
{
    caterva_array_t *catarr = itr->container->catarr;

    int ndim = catarr->ndim;

    uint64_t cont2 = itr->cont % catarr->csize; // element position in the chunk

    // set element index (in the chunk)
    itr->index[ndim - 1] = cont2 % catarr->pshape[ndim-1];
    uint64_t inc = catarr->pshape[ndim - 1];
    for (int i = ndim - 2; i >= 0; --i) {
        itr->index[i] = cont2 % (inc * catarr->pshape[i]) / inc;
        inc *= catarr->pshape[i];
    }

    // set element index (in entire container)
    uint64_t nchunk = itr->cont / catarr->csize;
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
 *
 */

INA_API(void) iarray_itr_init(iarray_itr_t *itr)
{
    itr->cont = 0;
    itr->nelem = 0;
    memset(itr->part, 0, itr->container->catarr->csize * itr->container->catarr->sc->typesize);
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
 *
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
    if (itr->cont % catarr->csize == 0) {
        blosc2_schunk_append_buffer(catarr->sc, itr->part, catarr->csize * catarr->sc->typesize);
        memset(itr->part, 0, catarr->csize * catarr->sc->typesize);
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
 * Function: iarray_itr_new
 * ------------------------
 *   Create a new iterator
 *
 *   container: the container used in the iterator
 *   itr: an iterator
 *
 *   returns: an error code
 */

INA_API(ina_rc_t) iarray_itr_new(iarray_container_t *container, iarray_itr_t **itr)
{
    *itr = (iarray_itr_t*)ina_mem_alloc(sizeof(iarray_itr_t));
    INA_RETURN_IF_NULL(itr);
    caterva_update_shape(container->catarr, *container->shape);
    (*itr)->container = container;
    (*itr)->part = (uint8_t *) ina_mem_alloc(container->catarr->csize * container->catarr->sc->typesize);

    (*itr)->index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));

    return 0;
}

/*
 * Function: iarray_itr_new
 * ------------------------
 *   Free an iterator structure
 *
 *   itr: an iterator
 *
 *   returns: an error code
 */

INA_API(ina_rc_t) iarray_itr_free(iarray_itr_t *itr)
{
    ina_mem_free(itr->index);
    ina_mem_free(itr->part);
    ina_mem_free(itr);
    return 0;
}
// MATMUL ITERATOR

INA_API(void) iarray_itr_matmul_init(iarray_itr_matmul_t *itr)
{
    itr->cont = 0;
    itr->nchunk1 = 0;
    itr->nchunk2 = 0;
}

INA_API(void) iarray_itr_matmul_next(iarray_itr_matmul_t *itr)
{
    uint64_t P = itr->container1->catarr->pshape[0];
    uint64_t M = itr->container1->catarr->eshape[0];
    uint64_t N = itr->container2->catarr->eshape[1];
    uint64_t K = itr->container1->catarr->eshape[1];

    itr->cont++;

    uint64_t m = itr->cont / ((K/P) * (N/P)) % (M/P);
    uint64_t n = itr->cont / ((K/P)) % (N/P);
    uint64_t k = itr->cont % (K/P);

    itr->nchunk1 = (m * (K/P) + k);
    itr->nchunk2 = (k * (N/P) + n);

}

INA_API(int) iarray_itr_matmul_finished(iarray_itr_matmul_t *itr)
{
    uint64_t P = itr->container1->catarr->pshape[0];
    uint64_t M = itr->container1->catarr->eshape[0];
    uint64_t N = itr->container2->catarr->eshape[1];
    uint64_t K = itr->container1->catarr->eshape[1];

    return itr->cont >= (M/P) * (N/P) * (K/P);
}

INA_API(ina_rc_t) iarray_itr_matmul_new(iarray_container_t *c1, iarray_container_t *c2, iarray_itr_matmul_t **itr)
{
    *itr = (iarray_itr_matmul_t*)ina_mem_alloc(sizeof(iarray_itr_matmul_t));
    INA_RETURN_IF_NULL(itr);
    (*itr)->container1 = c1;
    (*itr)->container2 = c2;

    return 0;
}

INA_API(ina_rc_t) iarray_itr_matmul_free(iarray_itr_matmul_t *itr)
{
    ina_mem_free(itr);
    return 0;
}
