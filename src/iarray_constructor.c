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

static ina_rc_t _iarray_container_fill_float(iarray_container_t *c, float value)
{
    caterva_fill(c->catarr, *c->shape, &value);
    return INA_SUCCESS;
}

static ina_rc_t _iarray_container_fill_double(iarray_container_t *c, double value)
{
    caterva_fill(c->catarr, *c->shape, &value);
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_arange(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    int start,
    int stop,
    int step,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    /* implement arange */

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_zeros(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            INA_FAIL_IF_ERROR(_iarray_container_fill_double(*container, 0.0));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            INA_FAIL_IF_ERROR(_iarray_container_fill_float(*container, 0.0f));
            break;
    }
    return INA_SUCCESS;
fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_ones(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    switch (dtshape->dtype) {
    case IARRAY_DATA_TYPE_DOUBLE:
        INA_FAIL_IF_ERROR(_iarray_container_fill_double(*container, 1.0));
        break;
    case IARRAY_DATA_TYPE_FLOAT:
        INA_FAIL_IF_ERROR(_iarray_container_fill_float(*container, 1.0f));
        break;
    }
    return INA_SUCCESS;
fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_fill_float(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    float value,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    INA_FAIL_IF_ERROR(_iarray_container_fill_float(*container, value));

    return INA_SUCCESS;

fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_fill_double(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    double value,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    INA_FAIL_IF_ERROR(_iarray_container_fill_double(*container, value));

    return INA_SUCCESS;

fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_from_buffer(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    void *buffer,
    size_t buffer_len,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(buffer);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    if (caterva_from_buffer((*container)->catarr, *(*container)->shape, buffer) != 0) {
        INA_ERROR(INA_ERR_FAILED);
        INA_FAIL_IF(1);
    }

    return INA_SUCCESS;

fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_to_buffer(iarray_context_t *ctx,
    iarray_container_t *container,
    void *buffer,
    size_t buffer_len)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(buffer);
    INA_VERIFY_NOT_NULL(container);

    if (caterva_to_buffer(container->catarr, buffer) != 0) {
        return INA_ERROR(INA_ERR_FAILED);
    }

    return INA_SUCCESS;
}


