/*
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include "iarray_private.h"
#include <libiarray/iarray.h>


INA_API(ina_rc_t) iarray_empty(iarray_context_t *ctx,
                                iarray_dtshape_t *dtshape,
                                iarray_storage_t *storage,
                                iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, dtshape, storage, container));

    caterva_config_t cat_cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cat_cfg);
    caterva_ctx_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_ctx_new(&cat_cfg, &cat_ctx));

    caterva_params_t cat_params = {0};
    iarray_create_caterva_params(dtshape, &cat_params);

    caterva_storage_t cat_storage = {0};
    iarray_create_caterva_storage(dtshape, storage, &cat_storage);
    IARRAY_ERR_CATERVA(caterva_empty(cat_ctx, &cat_params, &cat_storage, &(*container)->catarr));
    free(cat_storage.metalayers[0].sdata);
    free(cat_storage.metalayers[0].name);

    IARRAY_ERR_CATERVA(caterva_ctx_free(&cat_ctx));

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_uninit(iarray_context_t *ctx,
                                iarray_dtshape_t *dtshape,
                                iarray_storage_t *storage,
                                iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, dtshape, storage, container));

    caterva_config_t cat_cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cat_cfg);
    caterva_ctx_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_ctx_new(&cat_cfg, &cat_ctx));

    caterva_params_t cat_params = {0};
    iarray_create_caterva_params(dtshape, &cat_params);

    caterva_storage_t cat_storage = {0};
    iarray_create_caterva_storage(dtshape, storage, &cat_storage);
    IARRAY_ERR_CATERVA(caterva_uninit(cat_ctx, &cat_params, &cat_storage, &(*container)->catarr));
    free(cat_storage.metalayers[0].sdata);
    free(cat_storage.metalayers[0].name);

    IARRAY_ERR_CATERVA(caterva_ctx_free(&cat_ctx));

    return INA_SUCCESS;
}

ina_rc_t iarray_fill(iarray_context_t *ctx,
                     iarray_dtshape_t *dtshape,
                     void *value,
                     iarray_storage_t *storage,
                     iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, dtshape, storage, container));

    caterva_config_t cat_cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cat_cfg);
    caterva_ctx_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_ctx_new(&cat_cfg, &cat_ctx));

    caterva_params_t cat_params = {0};
    iarray_create_caterva_params(dtshape, &cat_params);

    caterva_storage_t cat_storage = {0};
    iarray_create_caterva_storage(dtshape, storage, &cat_storage);

    IARRAY_ERR_CATERVA(caterva_full(cat_ctx, &cat_params, &cat_storage, value, &(*container)->catarr));
     free(cat_storage.metalayers[0].sdata);
     free(cat_storage.metalayers[0].name);

     IARRAY_ERR_CATERVA(caterva_ctx_free(&cat_ctx));

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_zeros(iarray_context_t *ctx,
                               iarray_dtshape_t *dtshape,
                               iarray_storage_t *storage,
                               iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, dtshape, storage, container));

    caterva_config_t cat_cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cat_cfg);
    caterva_ctx_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_ctx_new(&cat_cfg, &cat_ctx));

    caterva_params_t cat_params = {0};
    iarray_create_caterva_params(dtshape, &cat_params);

    caterva_storage_t cat_storage = {0};
    iarray_create_caterva_storage(dtshape, storage, &cat_storage);

    IARRAY_ERR_CATERVA(caterva_zeros(cat_ctx, &cat_params, &cat_storage, &(*container)->catarr));
    free(cat_storage.metalayers[0].sdata);
    free(cat_storage.metalayers[0].name);

    IARRAY_ERR_CATERVA(caterva_ctx_free(&cat_ctx));

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_ones(iarray_context_t *ctx,
                              iarray_dtshape_t *dtshape,
                              iarray_storage_t *storage,
                              iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE: {
            double value = 1;
            IARRAY_RETURN_IF_FAILED(iarray_fill(ctx, dtshape, &value, storage, container));
            break;
        }
        case IARRAY_DATA_TYPE_FLOAT: {
            float value = 1;
            IARRAY_RETURN_IF_FAILED(iarray_fill(ctx, dtshape, &value, storage, container));
            break;
        }
        case IARRAY_DATA_TYPE_INT64: {
            int64_t value = 1;
            IARRAY_RETURN_IF_FAILED(iarray_fill(ctx, dtshape, &value, storage, container));
            break;
        }
        case IARRAY_DATA_TYPE_INT32: {
            int32_t value = 1;
            IARRAY_RETURN_IF_FAILED(iarray_fill(ctx, dtshape, &value, storage, container));
            break;
        }
        case IARRAY_DATA_TYPE_INT16: {
            int16_t value = 1;
            IARRAY_RETURN_IF_FAILED(iarray_fill(ctx, dtshape, &value, storage, container));
            break;
        }
        case IARRAY_DATA_TYPE_INT8: {
            int8_t value = 1;
            IARRAY_RETURN_IF_FAILED(iarray_fill(ctx, dtshape, &value, storage, container));
            break;
        }
        case IARRAY_DATA_TYPE_UINT64: {
            uint64_t value = 1;
            IARRAY_RETURN_IF_FAILED(iarray_fill(ctx, dtshape, &value, storage, container));
            break;
        }
        case IARRAY_DATA_TYPE_UINT32: {
            uint32_t value = 1;
            IARRAY_RETURN_IF_FAILED(iarray_fill(ctx, dtshape, &value, storage, container));
            break;
        }
        case IARRAY_DATA_TYPE_UINT16: {
            uint16_t value = 1;
            IARRAY_RETURN_IF_FAILED(iarray_fill(ctx, dtshape, &value, storage, container));
            break;
        }
        case IARRAY_DATA_TYPE_UINT8: {
            uint8_t value = 1;
            IARRAY_RETURN_IF_FAILED(iarray_fill(ctx, dtshape, &value, storage, container));
            break;
        }
        case IARRAY_DATA_TYPE_BOOL: {
            bool value = true;
            IARRAY_RETURN_IF_FAILED(iarray_fill(ctx, dtshape, &value, storage, container));
            break;
        }
        default:
            IARRAY_TRACE1(iarray.error, "The data type is invalid for this operation");
            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_from_buffer(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     void *buffer,
                                     int64_t buflen,
                                     iarray_storage_t *storage,
                                     iarray_container_t **container)
{

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(buffer);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, dtshape, storage, container));

    int64_t nitems = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        nitems *= dtshape->shape[i];
    }

    if (nitems * (int64_t) (*container)->dtshape->dtype_size > buflen) {
        IARRAY_TRACE1(iarray.error, "The size of the buffer is not enough");
        return INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER);
    }

    caterva_config_t cat_cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cat_cfg);
    caterva_ctx_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_ctx_new(&cat_cfg, &cat_ctx));

    caterva_params_t cat_params = {0};
    iarray_create_caterva_params(dtshape, &cat_params);

    caterva_storage_t cat_storage = {0};
    iarray_create_caterva_storage(dtshape, storage, &cat_storage);

    IARRAY_ERR_CATERVA(caterva_from_buffer(cat_ctx, buffer, buflen, &cat_params, &cat_storage, &(*container)->catarr));
    free(cat_storage.metalayers[0].sdata);
    free(cat_storage.metalayers[0].name);

    IARRAY_ERR_CATERVA(caterva_ctx_free(&cat_ctx));

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_to_buffer(iarray_context_t *ctx,
                                   iarray_container_t *container,
                                   void *buffer,
                                   int64_t buflen)
{
    int64_t size = 1;
    for (int i = 0; i < container->dtshape->ndim; ++i) {
        size *= container->dtshape->shape[i];
    }
    if (size == 0) {
        return INA_SUCCESS;
    }

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(buffer);
    INA_VERIFY_NOT_NULL(container);

    if (size * (int64_t) container->dtshape->dtype_size > buflen) {
        IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
        return INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER);
    }

    if (container->container_viewed != NULL) {
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
        caterva_ctx_t *cat_ctx;
        IARRAY_ERR_CATERVA(caterva_ctx_new(&cfg, &cat_ctx));
        IARRAY_ERR_CATERVA(caterva_to_buffer(cat_ctx, container->catarr, buffer, buflen));
        IARRAY_ERR_CATERVA(caterva_ctx_free(&cat_ctx));
    }

    return INA_SUCCESS;
}


INA_API(bool) iarray_is_empty(iarray_container_t *container) {
    INA_VERIFY_NOT_NULL(container);
    return false;
}


INA_API(ina_rc_t) iarray_copy(iarray_context_t *ctx,
                              iarray_container_t *src,
                              bool view,
                              iarray_storage_t *storage,
                              iarray_container_t **dest) {

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(src);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(dest);

    if (src->container_viewed != NULL && view) {
        IARRAY_TRACE1(iarray.error, "IArray can not copy a view into another view");
        return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
    }

    int64_t start[IARRAY_DIMENSION_MAX], stop[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < src->dtshape->ndim; ++i) {
        start[i] = 0;
        stop[i] = src->dtshape->shape[i];
    }
    if (src->container_viewed != NULL) {
        IARRAY_RETURN_IF_FAILED(iarray_empty(ctx, src->dtshape, storage, dest));

        int64_t nelem = 1;
        for (int i = 0; i < src->dtshape->ndim; ++i) {
            nelem *= src->dtshape->shape[i];
        }
        if (nelem == 0) {
            return INA_SUCCESS;
        }

        int64_t iter_blockshape[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < src->dtshape->ndim; ++i) {
            iter_blockshape[i] = storage->chunkshape[i];
        }
        iarray_iter_read_block_t *iter_read;
        iarray_iter_read_block_value_t read_val;
        iarray_iter_read_block_new(ctx, &iter_read, src, iter_blockshape, &read_val,
                                   false);
        iarray_iter_write_block_t *iter_write;
        iarray_iter_write_block_value_t write_val;
        iarray_iter_write_block_new(ctx, &iter_write, *dest, iter_blockshape,
                                    &write_val,
                                   false);

        while (INA_SUCCEED(iarray_iter_write_block_has_next(iter_write)) &&
        INA_SUCCEED(iarray_iter_read_block_has_next(iter_read))) {
            IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_next(iter_read, NULL, 0));
            IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_next(iter_write,NULL, 0));
            memcpy(write_val.block_pointer, read_val.block_pointer, write_val.block_size *
            src->catarr->itemsize);
        }
        iarray_iter_read_block_free(&iter_read);
        iarray_iter_write_block_free(&iter_write);

    } else {
        IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, src->dtshape, storage, dest));

        caterva_config_t cat_cfg = {0};
        iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cat_cfg);
        caterva_ctx_t *cat_ctx;
        IARRAY_ERR_CATERVA(caterva_ctx_new(&cat_cfg, &cat_ctx));

        caterva_params_t cat_params = {0};
        iarray_create_caterva_params(src->dtshape, &cat_params);

        caterva_storage_t cat_storage = {0};

        iarray_create_caterva_storage(src->dtshape, storage, &cat_storage);

        IARRAY_ERR_CATERVA(caterva_copy(cat_ctx, src->catarr, &cat_storage, &(*dest)->catarr));

        free(cat_storage.metalayers[0].sdata);
        free(cat_storage.metalayers[0].name);

        IARRAY_ERR_CATERVA(caterva_ctx_free(&cat_ctx));

    }
    return INA_SUCCESS;
}
