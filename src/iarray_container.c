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
    ina_rc_t rc;
    if (a->dtype != b->dtype) {
        printf("Dtypes are not equals\n");
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }
    if (a->ndim != b->ndim) {
        printf("Dims are not equals\n");
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_NDIM));
    }
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        if (a->shape[i] != b->shape[i]) {
            printf("Shapes are not equals\n");
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_SHAPE));
        }
    }
    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
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
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(container);

    ina_rc_t rc;

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

    for (int i = 0; i < c->dtshape->ndim; ++i) {
        if (start_[i] >= stop_[i]) {
            printf("Start is bigger than stop\n");
            INA_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
        }
        if (pshape[i] > stop_[i] - start_[i]){
            printf("pshape is bigger than shape\n");
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_PSHAPE));
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

        INA_FAIL_IF_ERROR(_iarray_view_new(ctx, c, &dtshape, start_, container));

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

        INA_FAIL_IF_ERROR(iarray_container_new(ctx, &dtshape, store, flags, container));

        if (c->transposed) {
            (*container)->transposed = true;
        }

        caterva_dims_t start__ = caterva_new_dims((int64_t *) start_, c->dtshape->ndim);
        caterva_dims_t stop__ = caterva_new_dims((int64_t *) stop_, c->dtshape->ndim);

        if (caterva_get_slice((*container)->catarr, c->catarr, &start__, &stop__) != 0) {
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
        }
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_container_free(ctx, container);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}

INA_API(ina_rc_t) iarray_set_slice(iarray_context_t *ctx,
                                   iarray_container_t *c,
                                   const int64_t *start,
                                   const int64_t *stop,
                                   iarray_container_t *slice)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(slice);

    ina_rc_t rc;

    if (c->dtshape->dtype != slice->dtshape->dtype) {
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }
    if (c->dtshape->ndim != slice->dtshape->ndim) {
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_NDIM));
    }

    int typesize = slice->catarr->ctx->cparams.typesize;
    int64_t buflen = slice->catarr->size;

    uint8_t *buffer;
    if (slice->catarr->storage == CATERVA_STORAGE_BLOSC) {
        buffer = ina_mem_alloc(buflen * typesize);
        INA_FAIL_IF_ERROR(iarray_to_buffer(ctx, slice, buffer, buflen * typesize));
    } else {
        buffer = slice->catarr->buf;
    }

    INA_FAIL_IF_ERROR(iarray_set_slice_buffer(ctx, c, start,stop, buffer, buflen * typesize));

    if (slice->catarr->storage == CATERVA_STORAGE_BLOSC) {
        INA_MEM_FREE_SAFE(buffer);
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    if (slice->catarr->storage == CATERVA_STORAGE_BLOSC) {
        INA_MEM_FREE_SAFE(buffer);
    }
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}


INA_API(ina_rc_t) iarray_get_slice_buffer(iarray_context_t *ctx,
                                          iarray_container_t *c,
                                          const int64_t *start,
                                          const int64_t *stop,
                                          void *buffer,
                                          const int64_t buflen)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(buffer);

    ina_rc_t rc;

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

    for (int i = 0; i < c->dtshape->ndim; ++i) {
        if (start_[i] >= stop_[i]) {
            printf("Start is bigger than stop\n");
            INA_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
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
            printf("The buffer size is not enough\n");
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
        }
    } else {
        if (psize * (int64_t)sizeof(float) > buflen) {
            printf("The buffer size is not enough\n");
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
        }
    }

    caterva_dims_t start__ = caterva_new_dims((int64_t *) start_, c->catarr->ndim);
    caterva_dims_t stop__ = caterva_new_dims((int64_t *) stop_, c->catarr->ndim);
    caterva_dims_t pshape_ = caterva_new_dims((int64_t *) pshape, c->catarr->ndim);

    if (caterva_get_slice_buffer(buffer, c->catarr, &start__, &stop__, &pshape_) != 0) {
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
    }

    size_t rows = (size_t)stop_[0] - start_[0];
    size_t cols = (size_t)stop_[1] - start_[1];
    if (c->transposed) {
        switch (c->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_dimatcopy('R', 'T', rows, cols, 1.0, (double *) buffer, cols, rows);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_simatcopy('R', 'T', rows, cols, 1.0f, (float *) buffer, cols, rows);
                break;
            default:
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}


INA_API(ina_rc_t) iarray_set_slice_buffer(iarray_context_t *ctx,
                                          iarray_container_t *c,
                                          const int64_t *start,
                                          const int64_t *stop,
                                          void *buffer,
                                          const int64_t buflen)
{
    // TODO: make use of buflen so as to avoid exceeding the buffer boundaries
    INA_UNUSED(ctx);
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);
    INA_VERIFY_NOT_NULL(buffer);

    ina_rc_t rc;

    if (c->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_STORAGE));
    }

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

    for (int i = 0; i < c->catarr->ndim; ++i) {
        if (start_[i] < 0) {
            INA_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
        }
        if (stop_[i] < start_[i]) {
            INA_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
        }
        if (c->catarr->shape[i] < stop_[i]) {
            INA_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
        }
    }

    if (c->transposed) {
        size_t rows = (size_t)stop_[0] - start_[0];
        size_t cols = (size_t)stop_[1] - start_[1];
        switch (c->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_dimatcopy('R', 'T', rows, cols, 1.0, (double *) buffer, cols, rows);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_simatcopy('R', 'T', rows, cols, 1.0f, (float *) buffer, cols, rows);
                break;
            default:
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
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

    int64_t psize = 1;
    for (int i = 0; i < c->catarr->ndim; ++i) {
        psize *= stop_[i] - start_[i];
    }

    switch (c->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if (psize * (int64_t)sizeof(double) > buflen)
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if (psize * (int64_t)sizeof(float) > buflen)
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        default:
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    caterva_dims_t start__ = caterva_new_dims(start_, c->dtshape->ndim);
    caterva_dims_t stop__ = caterva_new_dims(stop_, c->dtshape->ndim);
    int err = caterva_set_slice_buffer(c->catarr, buffer, &start__, &stop__);

    if (err != 0) {
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
    }
    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
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

    ina_rc_t rc;

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

    switch (c->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if (psize * (int64_t)sizeof(double) > buflen)
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if (psize * (int64_t)sizeof(float) > buflen)
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        default:
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    caterva_dims_t start__ = caterva_new_dims((int64_t *) start_, c->catarr->ndim);
    caterva_dims_t stop__ = caterva_new_dims((int64_t *) stop_, c->catarr->ndim);
    caterva_dims_t pshape_ = caterva_new_dims((int64_t *) pshape, c->catarr->ndim);

    if (caterva_get_slice_buffer_no_copy(buffer, c->catarr, &start__, &stop__, &pshape_) != 0) {
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
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

    ina_rc_t rc;

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
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if (psize * (int64_t)sizeof(float) > buflen)
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        default:
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    caterva_dims_t start__ = caterva_new_dims((int64_t *) start_, c->catarr->ndim);
    caterva_dims_t stop__ = caterva_new_dims((int64_t *) stop_, c->catarr->ndim);
    caterva_dims_t pshape__ = caterva_new_dims(pshape_, c->catarr->ndim);

    memset(buffer, 0, buflen);

    if (caterva_get_slice_buffer(buffer, c->catarr, &start__, &stop__, &pshape__) != 0) {
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}

INA_API(ina_rc_t) iarray_squeeze(iarray_context_t *ctx,
                                 iarray_container_t *container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);

    uint8_t inc = 0;

    if (!container->view) {
        if (caterva_squeeze(container->catarr) != 0) {
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
        }

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
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(dtshape);

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
    INA_VERIFY_NOT_NULL(nbytes);
    INA_VERIFY_NOT_NULL(cbytes);

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

INA_API(ina_rc_t) iarray_container_almost_equal(iarray_container_t *a, iarray_container_t *b, double tol)
{
    ina_rc_t rc;

    if (a->dtshape->dtype != b->dtshape->dtype){
        printf("Dtypes are not equals\n");
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }
    if (a->dtshape->ndim != b->dtshape->ndim) {
        printf("Dims are not equals\n");
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_NDIM));
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        if (a->dtshape->shape[i] != b->dtshape->shape[i]) {
            printf("Shapes are not equals");
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_SHAPE));
        }
    }

    int dtype = a->dtshape->dtype;
    int ndim = a->dtshape->ndim;

    // For the blocksize, choose the maximum of the partition shapes
    int64_t *blocksize = malloc(ndim * sizeof(int64_t));
    for (int i = 0; i < ndim; ++i) {
        blocksize[i] = INA_MAX(a->dtshape->pshape[i], b->dtshape->pshape[i]);
    }

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_t *ctx = NULL;
    INA_FAIL_IF_ERROR(iarray_context_new(&cfg, &ctx));
    iarray_iter_read_block_t *iter_a;
    iarray_iter_read_block_value_t val_a;
    INA_FAIL_IF_ERROR(iarray_iter_read_block_new(ctx, &iter_a, a, blocksize, &val_a, false));
    iarray_iter_read_block_t *iter_b;
    iarray_iter_read_block_value_t val_b;
    INA_FAIL_IF_ERROR(iarray_iter_read_block_new(ctx, &iter_b, b, blocksize, &val_b, false));

    while (INA_SUCCEED(iarray_iter_read_block_has_next(iter_a))) {
        INA_FAIL_IF_ERROR(iarray_iter_read_block_next(iter_a, NULL, 0));
        INA_FAIL_IF_ERROR(iarray_iter_read_block_next(iter_b, NULL, 0));

        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (int64_t i = 0; i < val_a.block_size; ++i) {
                double adiff = fabs(((double *)val_a.block_pointer)[i] - ((double *)val_b.block_pointer)[i]);
                double rdiff = fabs(((double *)val_a.block_pointer)[i] - ((double *)val_b.block_pointer)[i]) /
                    ((double *)val_a.block_pointer)[i];
                if (rdiff > tol) {
                    printf("%f, %f\n", ((double *)val_a.block_pointer)[i], ((double *)val_b.block_pointer)[i]);
                    printf("Values differ in nelem: %ld (diff: %f)\n",
                           (long)(i + val_a.nblock * val_a.block_size), adiff);
                    INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_ASSERTION_FAILED));
                }
            }
        }
        else {
            for (int64_t i = 0; i < val_a.block_size; ++i) {
                float adiff = fabsf(((float *)val_a.block_pointer)[i] - ((float *)val_b.block_pointer)[i]);
                float vdiff = fabsf(((float *)val_a.block_pointer)[i] - ((float *)val_b.block_pointer)[i]) /
                    ((float *)val_a.block_pointer)[i];
                if (vdiff > tol) {
                    printf("%f, %f\n", ((float *)val_a.block_pointer)[i], ((float *)val_b.block_pointer)[i]);
                    printf("Values differ in nelem: %ld (diff: %f)\n",
                           (long)(i + val_a.nblock * val_a.block_size), adiff);
                    INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_ASSERTION_FAILED));
                }
            }
        }
    }

    INA_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    iarray_context_free(&ctx);
    iarray_iter_read_block_free(&iter_a);
    iarray_iter_read_block_free(&iter_b);
    free(blocksize);
    return rc;
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
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}

INA_API(ina_rc_t) iarray_container_lt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}

INA_API(ina_rc_t) iarray_container_gte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}

INA_API(ina_rc_t) iarray_container_lte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}

INA_API(ina_rc_t) iarray_container_eq(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}
