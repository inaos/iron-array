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
        return INA_ERR_FALSE;
    }
    if (a->ndim != b->ndim) {
        return INA_ERR_FALSE;
    }
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return INA_ERR_FALSE;
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

    return _iarray_container_new(ctx, dtshape, store, flags, container);
}

INA_API(ina_rc_t) iarray_get_slice(iarray_context_t *ctx,
                                   iarray_container_t *c,
                                   int64_t *start,
                                   int64_t *stop,
                                   const int64_t *pshape,
                                   iarray_store_properties_t *store,
                                   int flags,
                                   bool view,
                                   iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);

    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t stop_[IARRAY_DIMENSION_MAX];

    int64_t *offset = c->auxshape->offset;

    for (int i = 0; i < c->dtshape->ndim; ++i) {
        if (start[i] < 0) {
            start_[i] =  offset[i] + start[i] + c->dtshape->shape[i];
        } else{
            start_[i] = offset[i] + (int64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[i] =  offset[i] + stop[i] + c->dtshape->shape[i];
        } else {
            stop_[i] = offset[i] + (int64_t) stop[i];
        }
    }

    if (view) {

        iarray_dtshape_t dtshape;
        dtshape.ndim = c->dtshape->ndim;
        dtshape.dtype = c->dtshape->dtype;

        for (int i = 0; i < dtshape.ndim; ++i) {
            dtshape.shape[i] = stop_[i] - start_[i];
            dtshape.pshape[i] = pshape[i];
        }

        _iarray_view_new(ctx, c, &dtshape, start_, container);

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
        }

        // Check if matrix is transposed
        if (c->transposed) {
            int64_t aux_stop[IARRAY_DIMENSION_MAX];
            int64_t aux_start[IARRAY_DIMENSION_MAX];

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

        if (c->transposed) {
            (*container)->transposed = true;
        }

        caterva_dims_t start__ = caterva_new_dims((int64_t *) start_, c->dtshape->ndim);
        caterva_dims_t stop__ = caterva_new_dims((int64_t *) stop_, c->dtshape->ndim);

        INA_FAIL_IF(caterva_get_slice((*container)->catarr, c->catarr, &start__, &stop__) != 0);
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
                                          int64_t buflen)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);

    int8_t ndim = c->dtshape->ndim;
    int64_t *offset = c->auxshape->offset;
    int8_t *index = c->auxshape->index;

    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t stop_[IARRAY_DIMENSION_MAX];

    for (int i = 0; i < c->catarr->ndim; ++i) {
        start_[i] = 0 + offset[i];
        stop_[i] = 1 + offset[i];
    }

    for (int i = 0; i < ndim; ++i) {
        if (start[i] < 0) {
            start_[index[i]] += start[i] + c->dtshape->shape[i];
        } else{
            start_[index[i]] += (int64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[index[i]] += stop[i] + c->dtshape->shape[i] - 1;
        } else {
            stop_[index[i]] += (int64_t) stop[i] - 1;
        }
    }

    if (c->transposed) {
        int64_t aux_stop[IARRAY_DIMENSION_MAX];
        int64_t aux_start[IARRAY_DIMENSION_MAX];

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
    int64_t psize = 1;
    for (int i = 0; i < c->catarr->ndim; ++i) {
        pshape[i] = stop_[i] - start_[i];
        psize *= pshape[i];
    }

    if (c->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        if (psize * (int64_t)sizeof(double) > buflen) {
            return INA_ERR_ERROR;
        }
    } else {
        if (psize * (int64_t)sizeof(float) > buflen) {
            return INA_ERR_ERROR;
        }
    }

    caterva_dims_t start__ = caterva_new_dims((int64_t *) start_, c->catarr->ndim);
    caterva_dims_t stop__ = caterva_new_dims((int64_t *) stop_, c->catarr->ndim);
    caterva_dims_t pshape_ = caterva_new_dims((int64_t *) pshape, c->catarr->ndim);

    INA_FAIL_IF(caterva_get_slice_buffer(buffer, c->catarr, &start__, &stop__, &pshape_) != 0);

    size_t rows = (size_t)stop_[0] - start_[0];
    size_t cols = (size_t)stop_[1] - start_[1];
    if (c->transposed) {
        switch (c->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_dimatcopy('R', 'T', rows, cols, 1.0, (double *) buffer, cols, rows);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_simatcopy('R', 'T', rows, cols, 1.0, (float *) buffer, cols, rows);
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    return INA_SUCCESS;

    fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) _iarray_get_slice_buffer_no_copy(iarray_context_t *ctx,
                                                   iarray_container_t *c,
                                                   int64_t *start,
                                                   int64_t *stop,
                                                   void **buffer,
                                                   int64_t buflen)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);

    int8_t ndim = c->dtshape->ndim;
    int64_t *offset = c->auxshape->offset;
    int8_t *index = c->auxshape->index;

    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t stop_[IARRAY_DIMENSION_MAX];

    for (int i = 0; i < c->catarr->ndim; ++i) {
        start_[i] = 0 + offset[i];
        stop_[i] = 1 + offset[i];
    }

    for (int i = 0; i < ndim; ++i) {
        if (start[i] < 0) {
            start_[index[i]] += start[i] + c->dtshape->shape[i];
        } else{
            start_[index[i]] += (int64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[index[i]] += stop[i] + c->dtshape->shape[i] - 1;
        } else {
            stop_[index[i]] += (int64_t) stop[i] - 1;
        }
    }

    if (c->transposed) {
        int64_t aux_stop[IARRAY_DIMENSION_MAX];
        int64_t aux_start[IARRAY_DIMENSION_MAX];

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
    int64_t psize = 1;
    for (int i = 0; i < c->catarr->ndim; ++i) {
        pshape[i] = stop_[i] - start_[i];
        psize *= pshape[i];
    }

    if (c->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        if (psize * (int64_t)sizeof(double) > buflen) {
            return INA_ERR_ERROR;
        }
    } else {
        if (psize * (int64_t)sizeof(float) > buflen) {
            return INA_ERR_ERROR;
        }
    }

    caterva_dims_t start__ = caterva_new_dims((int64_t *) start_, c->catarr->ndim);
    caterva_dims_t stop__ = caterva_new_dims((int64_t *) stop_, c->catarr->ndim);
    caterva_dims_t pshape_ = caterva_new_dims((int64_t *) pshape, c->catarr->ndim);

    INA_FAIL_IF(caterva_get_slice_buffer_no_copy(buffer, c->catarr, &start__, &stop__, &pshape_) != 0);

    //printf("GS %p\n", buffer);

    return INA_SUCCESS;

    fail:
    return ina_err_get_rc();
}

ina_rc_t _iarray_get_slice_buffer(iarray_context_t *ctx,
                                  iarray_container_t *c,
                                  int64_t *start,
                                  int64_t *stop,
                                  int64_t *pshape,
                                  void *buffer,
                                  int64_t buflen)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(pshape);


    int8_t ndim = c->dtshape->ndim;
    int64_t *offset = c->auxshape->offset;
    int8_t *index = c->auxshape->index;

    int64_t start_[IARRAY_DIMENSION_MAX];
    int64_t stop_[IARRAY_DIMENSION_MAX];
    int64_t pshape_[IARRAY_DIMENSION_MAX];

    for (int i = 0; i < c->catarr->ndim; ++i) {
        start_[i] = 0 + offset[i];
        stop_[i] = 1 + offset[i];
        pshape_[i] = 1;
    }

    for (int i = 0; i < ndim; ++i) {
        pshape_[i] = pshape[i];
        if (start[i] < 0) {
            start_[index[i]] += start[i] + c->dtshape->shape[i];
        } else{
            start_[index[i]] += (int64_t) start[i];
        }
        if (stop[i] < 0) {
            stop_[index[i]] += stop[i] + c->dtshape->shape[i] - 1;
        } else {
            stop_[index[i]] += (int64_t) stop[i] - 1;
        }
    }

    if (c->transposed) {
        int64_t aux_stop[IARRAY_DIMENSION_MAX];
        int64_t aux_start[IARRAY_DIMENSION_MAX];
        int64_t aux_pshape[IARRAY_DIMENSION_MAX];

        for (int i = 0; i < c->catarr->ndim; ++i) {
            aux_start[i] = start_[i];
            aux_stop[i] = stop_[i];
            aux_pshape[i] = pshape_[i];
        }

        for (int i = 0; i < c->catarr->ndim; ++i) {
            start_[i] = aux_start[c->catarr->ndim - 1 - i];
            stop_[i] = aux_stop[c->catarr->ndim - 1 - i];
            pshape_[i] = aux_pshape[c->catarr->ndim - 1 - i];
        }
    }

    int64_t psize = 1;
    for (int i = 0; i < ndim; ++i) {
        psize *= pshape[i];
    }

    switch (c->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if (psize * (int64_t)sizeof(double) > buflen)
                return INA_ERR_ERROR;
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if (psize * (int64_t)sizeof(float) > buflen)
                return INA_ERR_ERROR;
            break;
        default:
            return INA_ERR_EXCEEDED;
    }

    caterva_dims_t start__ = caterva_new_dims((int64_t *) start_, c->catarr->ndim);
    caterva_dims_t stop__ = caterva_new_dims((int64_t *) stop_, c->catarr->ndim);
    caterva_dims_t pshape__ = caterva_new_dims(pshape_, c->catarr->ndim);

    memset(buffer, 0, buflen);

    INA_FAIL_IF(caterva_get_slice_buffer(buffer, c->catarr, &start__, &stop__, &pshape__) != 0);

    return INA_SUCCESS;

fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_squeeze(iarray_context_t *ctx,
                                 iarray_container_t *container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);

    uint8_t inc = 0;

    if (!container->view) {
        INA_FAIL_IF(caterva_squeeze(container->catarr) != 0);

        if (container->dtshape->ndim != container->catarr->ndim) {
            container->dtshape->ndim = (uint8_t) container->catarr->ndim;
            for (int i = 0; i < container->catarr->ndim; ++i) {
                if (container->dtshape->shape[i] != container->catarr->shape[i]) {
                    inc += 1;
                }
                container->dtshape->shape[i] = container->catarr->shape[i];
                container->dtshape->pshape[i] = container->catarr->pshape[i];
                container->auxshape->shape_wos[i] = container->catarr->shape[i];
                container->auxshape->pshape_wos[i] = container->catarr->pshape[i];
                container->auxshape->offset[i] = container->auxshape->offset[i + inc];
            }
        }
    } else {
        inc = 0;
        for (int i = 0; i < container->dtshape->ndim; ++i) {
            if (container->dtshape->shape[i] == 1) {
                inc ++;
            } else {
                container->dtshape->shape[i - inc] = container->dtshape->shape[i];
                container->dtshape->pshape[i - inc] = container->dtshape->pshape[i];
                container->auxshape->index[i - inc] = (uint8_t) i;
            }
        }
        container->dtshape->ndim -= inc;
    }

    return INA_SUCCESS;

fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_get_dtshape(iarray_context_t *ctx,
                                     iarray_container_t *c,
                                     iarray_dtshape_t *dtshape)
{
    INA_UNUSED(ctx);
    dtshape->ndim = c->dtshape->ndim;
    dtshape->dtype = c->dtshape->dtype;
    for (int i = 0; i < c->dtshape->ndim; ++i) {
        dtshape->shape[i] = c->dtshape->shape[i];
        dtshape->pshape[i] = c->dtshape->pshape[i];
    }
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_container_info(iarray_container_t *c, int64_t *nbytes, int64_t *cbytes)
{
    INA_VERIFY_NOT_NULL(c);

    if (c->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        *nbytes = c->catarr->size * c->catarr->ctx->cparams.typesize;
        *cbytes = *nbytes;
    }
    else {
        *nbytes = c->catarr->sc->nbytes;
        *cbytes = c->catarr->sc->cbytes;
    }

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_container_almost_equal(iarray_container_t *a, iarray_container_t *b, double tol) {
    if (a->dtshape->dtype != b->dtshape->dtype){
        return INA_ERR_FAILED;
    }
    if (a->dtshape->ndim != b->dtshape->ndim) {
        return INA_ERR_FAILED;
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        INA_TEST_ASSERT_EQUAL_INT64(a->dtshape->shape[i], b->dtshape->shape[i]);
    }

    ina_rc_t retcode = INA_SUCCESS;
    int dtype = a->dtshape->dtype;
    int ndim = a->dtshape->ndim;

    // For the blocksize, choose the maximum of the partition shapes
    int64_t *blocksize = malloc(ndim * sizeof(int64_t));
    for (int i = 0; i < ndim; ++i) {
        blocksize[i] = INA_MAX(a->dtshape->pshape[i], b->dtshape->pshape[i]);
    }

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_t *ctx = NULL;
    iarray_context_new(&cfg, &ctx);
    iarray_iter_read_block_t *iter_a;
    iarray_iter_read_block_value_t val_a;
    iarray_iter_read_block_new(ctx, &iter_a, a, blocksize, &val_a, false);
    iarray_iter_read_block_t *iter_b;
    iarray_iter_read_block_value_t val_b;
    iarray_iter_read_block_new(ctx, &iter_b, b, blocksize, &val_b, false);

    while (iarray_iter_read_block_has_next(iter_a)) {
        iarray_iter_read_block_next(iter_a, NULL, 0);
        iarray_iter_read_block_next(iter_b, NULL, 0);

        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (int64_t i = 0; i < val_a.block_size; ++i) {
                double vdiff = fabs(((double *)val_a.pointer)[i] - ((double *)val_b.pointer)[i]) / ((double *)val_a.pointer)[i];
                if (vdiff > tol) {
                    printf("%f, %f\n", ((double *)val_a.pointer)[i], ((double *)val_b.pointer)[i]);
                    printf("Values differ in nelem: %ld (diff: %f)\n", (long)(i + val_a.nblock * val_a.block_size), vdiff);
                    retcode = INA_ERR_FAILED;
                    goto failed;
                }
            }
        }
        else {
            for (int64_t i = 0; i < val_a.block_size; ++i) {
                float vdiff = fabsf(((float *)val_a.pointer)[i] - ((float *)val_b.pointer)[i]) / ((float *)val_a.pointer)[i];
                if (vdiff > tol) {
                    printf("%f, %f\n", ((float *)val_a.pointer)[i], ((float *)val_b.pointer)[i]);
                    printf("Values differ in nelem: %ld (diff: %f)\n", (long)(i + val_a.nblock * val_a.block_size), vdiff);
                    retcode = INA_ERR_FAILED;
                    goto failed;
                }
            }
        }
    }
    iarray_context_free(&ctx);

failed:
    iarray_iter_read_block_free(iter_a);
    iarray_iter_read_block_free(iter_b);
    free(blocksize);

    return retcode;
}

INA_API(void) iarray_container_free(iarray_context_t *ctx, iarray_container_t **container)
{
    INA_UNUSED(ctx);
    INA_VERIFY_FREE(container);

    if ((*container)->view) {
        INA_MEM_FREE_SAFE((*container)->dtshape);
    } else {
        if ((*container)->catarr != NULL) {
            caterva_free_array((*container)->catarr);
        }
        if ((*container)->frame) {
            blosc2_free_frame((*container)->frame);
        }
        INA_MEM_FREE_SAFE((*container)->cparams);
        INA_MEM_FREE_SAFE((*container)->dparams);
        INA_MEM_FREE_SAFE((*container)->dtshape);
        INA_MEM_FREE_SAFE((*container)->auxshape);
        INA_MEM_FREE_SAFE(*container);
    }
}

INA_API(ina_rc_t) iarray_container_gt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_container_lt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_container_gte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_container_lte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_container_eq(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERR_NOT_IMPLEMENTED;
}
