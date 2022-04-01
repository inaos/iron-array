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

typedef struct {
    double start;
    double step;
    double base;
} iarray_constructor_logspace_info;


ina_rc_t iarray_constructor_logspace_fn(iarray_constructor_array_info_t *array_info,
                                      iarray_constructor_chunk_info_t *chunk_info,
                                      iarray_constructor_block_info_t *block_info,
                                      iarray_constructor_item_info_t *item_info,
                                      void *custom_info,
                                      void *custom_chunk_info,
                                      void *custom_block_info,
                                      uint8_t *item) {
    INA_UNUSED(array_info);
    INA_UNUSED(chunk_info);
    INA_UNUSED(block_info);
    INA_UNUSED(custom_chunk_info);
    INA_UNUSED(custom_block_info);

    iarray_constructor_logspace_info *logspace_info = custom_info;
    double start = logspace_info->start;
    double step = logspace_info->step;
    double base = logspace_info->base;
    iarray_dtshape_t *dtshape = array_info->a->dtshape;

    // logspace operation
    double val = pow(base, (double) item_info->index_flat * step + start);
    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE: {
            double value = (double) val;
            memcpy(item, &value, dtshape->dtype_size);
            break;
        }
        case IARRAY_DATA_TYPE_FLOAT: {
            float value = (float) val;
            memcpy(item, &value, dtshape->dtype_size);
            break;
        }
        case IARRAY_DATA_TYPE_INT64: {
            int64_t value = (int64_t) val;
            memcpy(item, &value, dtshape->dtype_size);
            break;
        }
        case IARRAY_DATA_TYPE_INT32: {
            int32_t value = (int32_t) val;
            memcpy(item, &value, dtshape->dtype_size);
            break;
        }
        case IARRAY_DATA_TYPE_INT16: {
            int16_t value = (int16_t) val;
            memcpy(item, &value, dtshape->dtype_size);
            break;
        }
        case IARRAY_DATA_TYPE_INT8: {
            int8_t value = (int8_t) val;
            memcpy(item, &value, dtshape->dtype_size);
            break;
        }
        case IARRAY_DATA_TYPE_UINT64: {
            uint64_t value = (uint64_t) val;
            memcpy(item, &value, dtshape->dtype_size);
            break;
        }
        case IARRAY_DATA_TYPE_UINT32: {
            uint32_t value = (uint32_t) val;
            memcpy(item, &value, dtshape->dtype_size);
            break;
        }
        case IARRAY_DATA_TYPE_UINT16: {
            uint16_t value = (uint16_t) val;
            memcpy(item, &value, dtshape->dtype_size);
            break;
        }
        case IARRAY_DATA_TYPE_UINT8: {
            uint8_t value = (uint8_t) val;
            memcpy(item, &value, dtshape->dtype_size);
            break;
        }
        default:
            INA_TRACE1(iarray.error, "The data type is invalid");
            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_logspace(iarray_context_t *ctx,
                                  iarray_dtshape_t *dtshape,
                                  double start,
                                  double stop,
                                  double base,
                                  iarray_storage_t *storage,
                                  iarray_container_t **container)
{
    int64_t nelem = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        nelem *= dtshape->shape[i];
    }

    iarray_constructor_logspace_info logspace_info;
    logspace_info.start = start;
    logspace_info.step = (stop - start) / ((double) nelem - 1);
    logspace_info.base = base;

    iarray_constructor_element_params_t elem_params = IARRAY_CONSTRUCTOR_ELEMENT_PARAMS_DEFAULT;
    elem_params.item_fn = iarray_constructor_logspace_fn;
    elem_params.constructor_info = &logspace_info;

    return iarray_constructor_element(ctx, dtshape, &elem_params, storage, container);
}
