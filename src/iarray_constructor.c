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



static ina_rc_t _iarray_container_fill_float(iarray_context_t *ctx, iarray_container_t *c, float value)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(c);

    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;

    IARRAY_RETURN_IF_FAILED(iarray_iter_write_new(ctx, &I, c, &val));

    while (INA_SUCCEED(iarray_iter_write_has_next(I))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_write_next(I));
        memcpy(val.elem_pointer, &value, sizeof(float));
    }
    IARRAY_ITER_FINISH();

    iarray_iter_write_free(&I);

    return INA_SUCCESS;
}


static ina_rc_t _iarray_container_fill_double(iarray_context_t *ctx, iarray_container_t *c, double value)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(c);

    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;

    IARRAY_RETURN_IF_FAILED(iarray_iter_write_new(ctx, &I, c, &val));

    while (INA_SUCCEED(iarray_iter_write_has_next(I))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_write_next(I));
        memcpy(val.elem_pointer, &value, sizeof(double));
    }
    IARRAY_ITER_FINISH();
    iarray_iter_write_free(&I);

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_arange(iarray_context_t *ctx,
                                iarray_dtshape_t *dtshape,
                                double start,
                                double stop,
                                double step,
                                iarray_storage_t *storage,
                                int flags,
                                iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    double contsize = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        contsize *= dtshape->shape[i];
    }

    double constant = (stop - start) / contsize;
    if (constant != step) {
        IARRAY_TRACE1(iarray.error, "The step parameter is invalid");
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }

    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, dtshape, storage, flags, container));

    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;

    IARRAY_RETURN_IF_FAILED(iarray_iter_write_new(ctx, &I, *container, &val));

    while (INA_SUCCEED(iarray_iter_write_has_next(I))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_write_next(I));

        int64_t i = 0;
        int64_t inc = 1;
        for (int j = dtshape->ndim - 1; j >= 0; --j) {
            i += val.elem_index[j] * inc;
            inc *= dtshape->shape[j];
        }

        if (dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
            double value = i * step + start;
            memcpy(val.elem_pointer, &value, sizeof(double));
        } else {
            float value = (float) (i * step + start);
            memcpy(val.elem_pointer, &value, sizeof(float));
        }
    }

    IARRAY_ITER_FINISH();
    iarray_iter_write_free(&I);

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_linspace(iarray_context_t *ctx,
                                  iarray_dtshape_t *dtshape,
                                  int64_t nelem,
                                  double start,
                                  double stop,
                                  iarray_storage_t *storage,
                                  int flags,
                                  iarray_container_t **container)
{

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    double contsize = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        contsize *= dtshape->shape[i];
    }

    if (contsize != nelem) {
        IARRAY_TRACE1(iarray.error, "The nelem parameter is invalid");
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }

    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, dtshape, storage, flags, container));

    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;

    IARRAY_RETURN_IF_FAILED(iarray_iter_write_new(ctx, &I, *container, &val));

    while (INA_SUCCEED(iarray_iter_write_has_next(I))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_write_next(I));

        int64_t i = 0;
        int64_t inc = 1;
        for (int j = dtshape->ndim - 1; j >= 0; --j) {
            i += val.elem_index[j] * inc;
            inc *= dtshape->shape[j];
        }

        if (dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
            double value = i * (stop - start) / (contsize - 1) + start;
            memcpy(val.elem_pointer, &value, sizeof(double));
        } else {
            float value = (float) (i * (stop - start) / (contsize - 1) + start);
            memcpy(val.elem_pointer, &value, sizeof(float));
        }
    }
    IARRAY_ITER_FINISH();
    iarray_iter_write_free(&I);

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_zeros(iarray_context_t *ctx,
                               iarray_dtshape_t *dtshape,
                               iarray_storage_t *storage,
                               int flags,
                               iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, dtshape, storage, flags, container));

    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            IARRAY_RETURN_IF_FAILED(_iarray_container_fill_double(ctx, *container, 0.0));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            IARRAY_RETURN_IF_FAILED(_iarray_container_fill_float(ctx, *container, 0.0f));
            break;
        default:
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }
    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_ones(iarray_context_t *ctx,
                              iarray_dtshape_t *dtshape,
                              iarray_storage_t *storage,
                              int flags,
                              iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, dtshape, storage, flags, container));

    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            IARRAY_RETURN_IF_FAILED(_iarray_container_fill_double(ctx, *container, 1.0));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            IARRAY_RETURN_IF_FAILED(_iarray_container_fill_float(ctx, *container, 1.0f));
            break;
        default:
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_fill_float(iarray_context_t *ctx,
                                    iarray_dtshape_t *dtshape,
                                    float value,
                                    iarray_storage_t *storage,
                                    int flags,
                                    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, dtshape, storage, flags, container));

    IARRAY_RETURN_IF_FAILED(_iarray_container_fill_float(ctx, *container, value));

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_fill_double(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     double value,
                                     iarray_storage_t *storage,
                                     int flags,
                                     iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, dtshape, storage, flags, container));

    IARRAY_RETURN_IF_FAILED(_iarray_container_fill_double(ctx, *container, value));

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_from_buffer(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     void *buffer,
                                     int64_t buflen,
                                     iarray_storage_t *storage,
                                     int flags,
                                     iarray_container_t **container)
{

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(buffer);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, dtshape, storage, flags, container));

    (*container)->catarr->empty = false;

    switch ((*container)->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if ((* container)->catarr->nitems * (int64_t) sizeof(double) > buflen) {
                IARRAY_TRACE1(iarray.error, "The size of the buffer is not enough");
                return INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER);
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if ((* container)->catarr->nitems * (int64_t) sizeof(float) > buflen) {
                IARRAY_TRACE1(iarray.error, "The size of the buffer is not enough");
                return INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER);
            }
            break;
        default:
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }

    caterva_config_t cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
    caterva_params_t params = {0};
    iarray_create_caterva_params(dtshape, &params);
    caterva_storage_t cat_storage = {0};
    iarray_create_caterva_storage(dtshape, storage, &cat_storage);
    caterva_context_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

    uint8_t *smeta = NULL;
    if (cat_storage.backend == CATERVA_STORAGE_BLOSC) {
        cat_storage.properties.blosc.nmetalayers = 1;
        cat_storage.properties.blosc.metalayers[0].name = "iarray";
        uint32_t smeta_len;
        blosc2_get_metalayer((*container)->catarr->sc, "iarray", &smeta, &smeta_len);
        cat_storage.properties.blosc.metalayers[0].sdata = smeta;
        cat_storage.properties.blosc.metalayers[0].size = smeta_len;
    }
    IARRAY_ERR_CATERVA(caterva_array_free(cat_ctx, &(*container)->catarr));

    IARRAY_ERR_CATERVA(caterva_array_from_buffer(cat_ctx, buffer, buflen, &params, &cat_storage, &(*container)->catarr));

    if (cat_storage.backend == CATERVA_STORAGE_BLOSC) {
        free(smeta);
    }
    (*container)->catarr->empty = false;

    IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_to_buffer(iarray_context_t *ctx,
                                   iarray_container_t *container,
                                   void *buffer,
                                   int64_t buflen)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(buffer);
    INA_VERIFY_NOT_NULL(container);

    int64_t size = 1;
    for (int i = 0; i < container->dtshape->ndim; ++i) {
        size *= container->dtshape->shape[i];
    }

    switch (container->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if (size * (int64_t) sizeof(double) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                return INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER);
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if (size * (int64_t) sizeof(float) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                return INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER);
            }
            break;
        default:
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }

    if (container->view) {
        int64_t start[IARRAY_DIMENSION_MAX];
        int64_t stop[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < container->dtshape->ndim; ++i) {
            start[i] = 0;
            stop[i] = container->dtshape->shape[i];
        }
        IARRAY_RETURN_IF_FAILED(iarray_get_slice_buffer(ctx, container, start, stop, buffer, buflen));
    } else {
        caterva_config_t cfg = {0};
        iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
        caterva_context_t *cat_ctx;
        IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));
        IARRAY_ERR_CATERVA(caterva_array_to_buffer(cat_ctx, container->catarr, buffer, buflen));
        IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));
    }

    if ((!container->view) && (container->transposed == 1)) {
        switch (container->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_dimatcopy('R', 'T', (size_t)container->dtshape->shape[1], (size_t)container->dtshape->shape[0], 1.0,
                              (double *) buffer, (size_t)container->dtshape->shape[0], (size_t)container->dtshape->shape[1]);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_simatcopy('R', 'T', (size_t)container->dtshape->shape[1], (size_t)container->dtshape->shape[0], 1.0f,
                              (float *) buffer, (size_t)container->dtshape->shape[0], (size_t)container->dtshape->shape[1]);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
        }
    }

    return INA_SUCCESS;
}


INA_API(bool) iarray_is_empty(iarray_container_t *container) {
    INA_VERIFY_NOT_NULL(container);
    if (container->catarr->empty) {
        return true;
    }
    return false;
}


INA_API(ina_rc_t) iarray_copy(iarray_context_t *ctx,
                              iarray_container_t *src,
                              bool view,
                              iarray_storage_t *storage,
                              int flags,
                              iarray_container_t **dest) {

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(src);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(dest);

    INA_UNUSED(flags);

    caterva_config_t cfg = {0};
    IARRAY_RETURN_IF_FAILED(iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg));
    caterva_context_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

    (*dest) = (iarray_container_t *) ina_mem_alloc(sizeof(iarray_container_t));
    (*dest)->dtshape = (iarray_dtshape_t *) ina_mem_alloc(sizeof(iarray_dtshape_t));
    ina_mem_cpy((*dest)->dtshape, src->dtshape, sizeof(iarray_dtshape_t));
    (*dest)->view = view;
    (*dest)->transposed = src->transposed;
    if ((*dest)->view) {
        (*dest)->storage = src->storage;
    } else {
        (*dest)->storage = (iarray_storage_t *) ina_mem_alloc(sizeof(iarray_storage_t));
        ina_mem_cpy((*dest)->storage, storage, sizeof(iarray_storage_t));
    }

    if (src->view && !view) {
        (*dest)->auxshape = (iarray_auxshape_t *) ina_mem_alloc(sizeof(iarray_auxshape_t));
        for (int i = 0; i < (*dest)->dtshape->ndim; ++i) {
            (*dest)->auxshape->offset[i] = 0;
            (*dest)->auxshape->index[i] = (int8_t) i;
            (*dest)->auxshape->shape_wos[i] = src->dtshape->shape[i];
            (*dest)->auxshape->pshape_wos[i] = storage->chunkshape[i];
            (*dest)->auxshape->bshape_wos[i] = storage->blockshape[i];
        }
    } else {
        (*dest)->auxshape = (iarray_auxshape_t *) ina_mem_alloc(sizeof(iarray_auxshape_t));
        ina_mem_cpy((*dest)->auxshape, src->auxshape, sizeof(iarray_auxshape_t));
    }

    if (view) {
        (*dest)->catarr = src->catarr;
    } else {
        caterva_params_t params = {0};
        IARRAY_RETURN_IF_FAILED(iarray_create_caterva_params(src->dtshape, &params));

        caterva_storage_t cat_storage = {0};
        IARRAY_RETURN_IF_FAILED(iarray_create_caterva_storage(src->dtshape, storage, &cat_storage));

        if (src->view) {
            int64_t *start = src->auxshape->offset;
            int64_t stop[IARRAY_DIMENSION_MAX];
            for (int i = 0; i < src->catarr->ndim; ++i) {
                stop[i] = src->auxshape->offset[i] + src->auxshape->shape_wos[i];
            }

            IARRAY_ERR_CATERVA(caterva_array_get_slice(cat_ctx, src->catarr, start, stop, &cat_storage, &(*dest)->catarr));
            IARRAY_ERR_CATERVA(caterva_array_squeeze(cat_ctx, (*dest)->catarr));
        } else {
            IARRAY_ERR_CATERVA(caterva_array_copy(cat_ctx, src->catarr, &cat_storage, &(*dest)->catarr));
        }
    }

    IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));

    return INA_SUCCESS;
}
