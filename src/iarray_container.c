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
#include "iarray_constructor.h"


INA_API(ina_rc_t) iarray_container_dtshape_equal(iarray_dtshape_t *a, iarray_dtshape_t *b)
{
    if (a->dtype != b->dtype) {
        return -1;
    }
    if (a->ndim != b->ndim) {
        return -1;
    }
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return -1;
        }
    }
    return 0;
}

INA_API(ina_rc_t) iarray_container_new(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    for (int i = 0; i < dtshape->ndim; ++i) {
        dtshape->offset[i] = 0;
    }

    return _iarray_container_new(ctx, dtshape, store, flags, container);
}

INA_API(ina_rc_t) iarray_get_slice(iarray_context_t *ctx,
                                   iarray_container_t *c,
                                   int64_t *start,
                                   int64_t *stop,
                                   uint64_t *pshape,
                                   iarray_store_properties_t *store,
                                   int flags,
                                   int view,
                                   iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);


    uint64_t start_[IARRAY_DIMENSION_MAX];
    uint64_t stop_[IARRAY_DIMENSION_MAX];

    uint64_t *offset = c->dtshape->offset;

    for (int i = 0; i < c->dtshape->ndim; ++i) {
        uint64_t of = offset[i];
        if (start[i] < 0) {
            start_[i] =  of + start[i] + c->dtshape->shape[i];
        } else{
            start_[i] = of + (uint64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[i] =  of + stop[i] + c->dtshape->shape[i];
        } else {
            stop_[i] = of + (uint64_t) stop[i];
        }
    }

    if (view == 1) { //TODO: Create a flag to indicate if a view is desired or not

        iarray_dtshape_t dtshape;
        dtshape.ndim = c->dtshape->ndim;
        dtshape.dtype = c->dtshape->dtype;

        for (int i = 0; i < dtshape.ndim; ++i) {
            dtshape.shape[i] = stop_[i] - start_[i];
            dtshape.pshape[i] = pshape[i];
            dtshape.offset[i] = start_[i];
        }

        _iarray_view_new(ctx, c, &dtshape, container);

        (*container)->view = 1;
        if (c->transposed == 1) {
            (*container)->transposed = 1;
        }

    } else {
        iarray_dtshape_t dtshape;

        dtshape.ndim = c->dtshape->ndim;
        dtshape.dtype = c->dtshape->dtype;

        for (int i = 0; i < dtshape.ndim; ++i) {
            dtshape.shape[i] = stop_[i] - start_[i];
            dtshape.pshape[i] = pshape[i];
            dtshape.offset[i] = 0;
        }

        // Check if matrix is transposed

        if (c->transposed == 1) {
            uint64_t aux_stop[IARRAY_DIMENSION_MAX];
            uint64_t aux_start[IARRAY_DIMENSION_MAX];

            for (int i = 0; i < c->dtshape->ndim; ++i) {
                aux_start[i] = start_[i];
                aux_stop[i] = stop_[i];
            }

            for (int i = 0; i < c->dtshape->ndim; ++i) {
                start_[i] = aux_start[c->dtshape->ndim - 1 - i];
                stop_[i] = aux_stop[c->dtshape->ndim - 1 - i];
            }
        }

        iarray_container_new(ctx, &dtshape, store, flags, container);

        if (c->transposed == 1) {
            (*container)->transposed = 1;
        }

        caterva_dims_t start__ = caterva_new_dims((uint64_t *) start_, c->dtshape->ndim);
        caterva_dims_t stop__ = caterva_new_dims((uint64_t *) stop_, c->dtshape->ndim);

        INA_FAIL_IF(caterva_get_slice((*container)->catarr, c->catarr, start__, stop__) != 0);
    }
    return INA_SUCCESS;

fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_get_slice_buffer(iarray_context_t *ctx,
                                          iarray_container_t *c,
                                          int64_t *start,
                                          int64_t *stop,
                                          void *buffer,
                                          uint64_t buflen)
{
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);

    uint8_t ndim = c->dtshape->ndim;
    uint64_t *off = c->dtshape->offset;

    uint64_t start_[IARRAY_DIMENSION_MAX];
    uint64_t stop_[IARRAY_DIMENSION_MAX];

    for (int i = 0; i < ndim; ++i) {
        if (start[i] < 0) {
            start_[i] = off[i] + start[i] + c->dtshape->shape[i];
        } else{
            start_[i] = off[i] + (uint64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[i] = off[i] + stop[i] + c->dtshape->shape[i];
        } else {
            stop_[i] = off[i] + (uint64_t) stop[i];
        }
    }

    if (c->transposed == 1) {
        uint64_t aux_stop[IARRAY_DIMENSION_MAX];
        uint64_t aux_start[IARRAY_DIMENSION_MAX];

        for (int i = 0; i < c->dtshape->ndim; ++i) {
            aux_start[i] = start_[i];
            aux_stop[i] = stop_[i];
        }

        for (int i = 0; i < c->dtshape->ndim; ++i) {
            start_[i] = aux_start[c->dtshape->ndim - 1 - i];
            stop_[i] = aux_stop[c->dtshape->ndim - 1 - i];
        }
    }

    int64_t pshape[IARRAY_DIMENSION_MAX];
    uint64_t psize = 1;
    for (int i = 0; i < ndim; ++i) {
        pshape[i] = stop_[i] - start_[i];
        psize *= pshape[i];
    }

    if (c->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        if (psize * sizeof(double) > buflen) {
            return INA_ERR_ERROR;
        }
    } else {
        if (psize * sizeof(float) > buflen) {
            return INA_ERR_ERROR;
        }
    }

    caterva_dims_t start__ = caterva_new_dims((uint64_t *) start_, ndim);
    caterva_dims_t stop__ = caterva_new_dims((uint64_t *) stop_, ndim);
    caterva_dims_t pshape_ = caterva_new_dims((uint64_t *) pshape, ndim);

    INA_FAIL_IF(caterva_get_slice_buffer(buffer, c->catarr, start__, stop__, pshape_) != 0);

    uint64_t rows = stop_[0] - start_[0];
    uint64_t cols = stop_[1] - start_[1];

    if (c->transposed == 1) {
        switch (c->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_dimatcopy('R', 'T', rows, cols, 1.0, (double *) buffer, cols, rows);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_simatcopy('R', 'T', rows, cols, 1.0,
                              (float *) buffer, cols, rows);
                break;
        }
    }

    return INA_SUCCESS;

    fail:
    return ina_err_get_rc();
}

ina_rc_t _iarray_get_slice_buffer(iarray_context_t *ctx,
                                  iarray_container_t *c,
                                  int64_t *start,
                                  int64_t *stop,
                                  uint64_t *pshape,
                                  void *buffer,
                                  uint64_t buflen)
{
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(pshape);

    uint8_t ndim = c->dtshape->ndim;

    uint64_t start_[IARRAY_DIMENSION_MAX];
    uint64_t stop_[IARRAY_DIMENSION_MAX];
    uint64_t pshape_[IARRAY_DIMENSION_MAX];

    for (int i = 0; i < ndim; ++i) {
        pshape_[i] = pshape[i];
        if (start[i] < 0) {
            start_[i] = start[i] + c->dtshape->shape[i];
        } else{
            start_[i] = (uint64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[i] = stop[i] + c->dtshape->shape[i];
        } else {
            stop_[i] = (uint64_t) stop[i];
        }
    }

    if (c->transposed == 1) {
        uint64_t aux_stop[IARRAY_DIMENSION_MAX];
        uint64_t aux_start[IARRAY_DIMENSION_MAX];
        uint64_t aux_pshape[IARRAY_DIMENSION_MAX];

        for (int i = 0; i < c->dtshape->ndim; ++i) {
            aux_start[i] = start_[i];
            aux_stop[i] = stop_[i];
            aux_pshape[i] = pshape[i];
        }

        for (int i = 0; i < c->dtshape->ndim; ++i) {
            start_[i] = aux_start[c->dtshape->ndim - 1 - i];
            stop_[i] = aux_stop[c->dtshape->ndim - 1 - i];
            pshape_[i] = aux_pshape[c->dtshape->ndim - 1 - i];
        }
    }

    uint64_t psize = 1;
    for (int i = 0; i < ndim; ++i) {
        psize *= pshape[i];
    }

    if (c->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        if (psize * sizeof(double) > buflen) {
            return INA_ERR_ERROR;
        }
    } else {
        if (psize * sizeof(float) > buflen) {
            return INA_ERR_ERROR;
        }
    }

    caterva_dims_t start__ = caterva_new_dims((uint64_t *) start_, ndim);
    caterva_dims_t stop__ = caterva_new_dims((uint64_t *) stop_, ndim);
    caterva_dims_t pshape__ = caterva_new_dims(pshape_, ndim);

    memset(buffer, 0, buflen);

    INA_FAIL_IF(caterva_get_slice_buffer(buffer, c->catarr, start__, stop__, pshape__) != 0);

    /*
    if (c->transposed == 1) {
        uint64_t rows = pshape[1];
        uint64_t cols = pshape[0];
        switch (c->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_dimatcopy('R', 'T', rows, cols, 1.0, (double *) buffer, cols, rows);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_simatcopy('R', 'T', rows, cols, 1.0, (float *) buffer, cols, rows);
                break;
        }
    }
    */

    return INA_SUCCESS;

fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_squeeze(iarray_context_t *ctx,
                                 iarray_container_t *container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);

    INA_FAIL_IF(caterva_squeeze(container->catarr) != 0);
    uint8_t inc = 0;
    if (container->dtshape->ndim != container->catarr->ndim) {
        container->dtshape->ndim = (uint8_t) container->catarr->ndim;
        for (int i = 0; i < container->catarr->ndim; ++i) {
            if (container->dtshape->shape[i] != container->catarr->shape[i]) {
                inc += 1;
            }
            container->dtshape->shape[i] = container->catarr->shape[i];
            container->dtshape->pshape[i] = container->catarr->pshape[i];
            container->dtshape->offset[i] = container->dtshape->offset[i + inc];
        }
    }

    return INA_SUCCESS;

fail:
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_container_info(iarray_container_t *c,
    uint64_t *nbytes,
    uint64_t *cbytes)
{
    INA_VERIFY_NOT_NULL(c);

    *nbytes = (uint64_t) c->catarr->sc->nbytes;
    *cbytes = (uint64_t) c->catarr->sc->cbytes;

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_container_almost_equal(iarray_container_t *a, iarray_container_t *b, double tol) {
    if(a->dtshape->dtype != b->dtshape->dtype){
        return INA_ERR_FAILED;
    }
    if(a->dtshape->ndim != b->dtshape->ndim) {
        return INA_ERR_FAILED;
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        INA_TEST_ASSERT_EQUAL_UINT64(a->dtshape->shape[i], b->dtshape->shape[i]);
    }

    ina_rc_t retcode = INA_SUCCESS;
    int dtype = a->dtshape->dtype;
    int ndim = a->dtshape->ndim;

    // For the blocksize, choose the maximum of the partition shapes
    uint64_t *blocksize = malloc(ndim * sizeof(uint64_t));
    for (int i = 0; i < ndim; ++i) {
        blocksize[i] = INA_MAX(a->dtshape->pshape[i], b->dtshape->pshape[i]);
    }

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_t *ctx = NULL;
    iarray_context_new(&cfg, &ctx);
    iarray_iter_read_block_t *iter_a;
    iarray_iter_read_block_new(ctx, a, &iter_a, blocksize);
    iarray_iter_read_block_t *iter_b;
    iarray_iter_read_block_new(ctx, b, &iter_b, blocksize);

    for (iarray_iter_read_block_init(iter_a), iarray_iter_read_block_init(iter_b);
         !iarray_iter_read_block_finished(iter_a);
         iarray_iter_read_block_next(iter_a), iarray_iter_read_block_next(iter_b)) {

        iarray_iter_read_block_value_t val_a;
        iarray_iter_read_block_value(iter_a, &val_a);
        iarray_iter_read_block_value_t val_b;
        iarray_iter_read_block_value(iter_b, &val_b);

        uint64_t block_size = 1;
        for (int i = 0; i < ndim; ++i) {
            block_size *= val_a.block_shape[i];
        }

        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (uint64_t i = 0; i < block_size; ++i) {
                double vdiff = fabs(((double *)val_a.pointer)[i] - ((double *)val_b.pointer)[i]) / ((double *)val_a.pointer)[i];
                if (vdiff > tol) {
                    printf("%f, %f\n", ((double *)val_a.pointer)[i], ((double *)val_b.pointer)[i]);
                    printf("Values differ in nelem: %llu (diff: %f)\n", i + val_a.nelem * block_size, vdiff);
                    retcode = INA_ERR_FAILED;
                    goto failed;
                }
            }
        }
        else {
            for (uint64_t i = 0; i < block_size; ++i) {
                float vdiff = fabsf(((float *)val_a.pointer)[i] - ((float *)val_b.pointer)[i]) / ((float *)val_a.pointer)[i];
                if (vdiff > tol) {
                    printf("%f, %f\n", ((float *)val_a.pointer)[i], ((float *)val_b.pointer)[i]);
                    printf("Values differ in nelem: %llu (diff: %f)\n", i, vdiff);
                    retcode = INA_ERR_FAILED;
                    goto failed;
                }
            }
        }
    }

failed:
    iarray_iter_read_block_free(iter_a);
    iarray_iter_read_block_free(iter_b);
    free(blocksize);

    return retcode;
}

INA_API(void) iarray_container_free(iarray_context_t *ctx, iarray_container_t **container)
{
    INA_VERIFY_FREE(container);

    if ((*container)->view = 1) {
        INA_MEM_FREE_SAFE((*container)->dtshape);
    } else {
        if ((*container)->catarr != NULL) {
            caterva_free_array((*container)->catarr);
        }
        INA_MEM_FREE_SAFE((*container)->frame);
        INA_MEM_FREE_SAFE((*container)->cparams);
        INA_MEM_FREE_SAFE((*container)->dparams);
        INA_MEM_FREE_SAFE((*container)->dtshape);
        INA_MEM_FREE_SAFE(*container);
    }
}

INA_API(ina_rc_t) iarray_container_gt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_container_lt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_container_gte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_container_lte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_container_eq(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}
