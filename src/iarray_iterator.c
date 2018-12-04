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

void _update_itr_index(iarray_itr_t *itr) 
{

    caterva_array_t *catarr = itr->container->catarr;

    int ndim = catarr->ndim;

    uint64_t cont2 = itr->cont % catarr->csize;
    itr->index[ndim - 1] = cont2 % catarr->pshape[ndim-1];
    uint64_t inc = catarr->pshape[ndim - 1];

    for (int i = ndim - 2; i >= 0; --i) {
        itr->index[i] = cont2 % (inc * catarr->pshape[i]) / inc;
        inc *= catarr->pshape[i];
    }

    uint64_t nchunk = itr->cont / catarr->csize;

    uint64_t aux_nchunk[CATERVA_MAXDIM];

    aux_nchunk[ndim - 1] = catarr->eshape[ndim - 1] / catarr->pshape[ndim - 1];
    for (int k = ndim - 2; k >= 0; --k) {
        aux_nchunk[k] = aux_nchunk[k + 1] * (catarr->eshape[k] / catarr->pshape[k]);
    }
    for (int j = 0; j < ndim; ++j) {
        itr->index[j] += nchunk % aux_nchunk[j] / (aux_nchunk[j] / (catarr->eshape[j] / catarr->pshape[j])) * catarr->pshape[j];
    }

    if (itr->container->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        itr->pointer = (void *)&((double*)itr->part)[cont2];
    } else{
        itr->pointer = (void *)&((float*)itr->part)[cont2];
    }

    itr->nelem = 0;
    inc = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        itr->nelem += itr->index[i] * inc;
        inc *= itr->container->dtshape->shape[i];
    }
}


void _iarray_itr_init(iarray_itr_t *itr)
{
    itr->cont = 0;
    itr->nelem = 0;
    memset(itr->part, 0, itr->container->catarr->csize * itr->container->catarr->sc->typesize);
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        itr->index[i] = 0;
    }
    itr->pointer = &itr->part[0];
}

void _iarray_itr_next(iarray_itr_t *itr)
{

    caterva_array_t *catarr = itr->container->catarr;
    int ndim = catarr->ndim;

    itr->cont += 1;

    _update_itr_index(itr);

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

    if (itr->cont % catarr->csize == 0) {
        blosc2_schunk_append_buffer(catarr->sc, itr->part, catarr->csize * catarr->sc->typesize);
        memset(itr->part, 0, catarr->csize * catarr->sc->typesize);
    }

    _update_itr_index(itr);
}


int _iarray_itr_finished(iarray_itr_t *itr)
{
    return itr->cont >= itr->container->catarr->esize;
}


INA_API(ina_rc_t) iarray_itr_new(iarray_container_t *container, iarray_itr_t **itr)
{
    *itr = (iarray_itr_t*)ina_mem_alloc(sizeof(iarray_itr_t));
    INA_RETURN_IF_NULL(itr);
    caterva_update_shape(container->catarr, *container->shape);
    (*itr)->container = container;
    (*itr)->part = (uint8_t *) ina_mem_alloc(container->catarr->csize * container->catarr->sc->typesize);

    (*itr)->index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));

    (*itr)->init = _iarray_itr_init;
    (*itr)->next = _iarray_itr_next;
    (*itr)->finished = _iarray_itr_finished;
    return 0;
}

INA_API(ina_rc_t) iarray_itr_free(iarray_itr_t *itr)
{
    ina_mem_free(itr->index);
    ina_mem_free(itr->part);
    ina_mem_free(itr);
    return 0;
}
