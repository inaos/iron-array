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
    caterva_dims_t shape = caterva_new_dims(c->dtshape->shape, c->dtshape->ndim);
    caterva_fill(c->catarr, shape, &value);
    return INA_SUCCESS;
}

static ina_rc_t _iarray_container_fill_double(iarray_container_t *c, double value)
{
    caterva_dims_t shape = caterva_new_dims(c->dtshape->shape, c->dtshape->ndim);
    caterva_fill(c->catarr, shape, &value);
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

    // TODO: would it be interesting to add a `buffer_len` parameter to `caterva_from_buffer()`?
    caterva_dims_t shape = caterva_new_dims((*container)->dtshape->shape, (*container)->dtshape->ndim);
    if (caterva_from_buffer((*container)->catarr, shape, buffer) != 0) {
        INA_ERROR(INA_ERR_FAILED);
        INA_FAIL_IF(1);
    }

    return INA_SUCCESS;

fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}


static int32_t deserialize_meta(uint8_t *smeta, uint32_t smeta_len, iarray_data_type_t *dtype)
{
    uint8_t *pmeta = smeta;

    // We only have an entry with the datatype (enumerated < 128)
    *dtype = *pmeta;
    pmeta += 1;
    assert(pmeta - smeta == smeta_len);
    if (*dtype >= IARRAY_DATA_TYPE_MAX) {
        return INA_ERR_FAILED;
    }

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_from_file(iarray_context_t *ctx, iarray_store_properties_t *store,
                                   iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);

    caterva_ctx_t *cat_ctx = caterva_new_ctx(NULL, NULL, BLOSC_CPARAMS_DEFAULTS, BLOSC_DPARAMS_DEFAULTS);

    caterva_array_t *catarr = caterva_from_file(cat_ctx, store->id);
    if (catarr == NULL) {
        INA_ERROR(INA_ERR_FAILED);
        INA_FAIL_IF(1);
    }

    uint8_t *smeta;
    uint32_t smeta_len;
    blosc2_frame_get_namespace(catarr->sc->frame, "iarray", &smeta, &smeta_len);
    iarray_data_type_t dtype;
    deserialize_meta(smeta, smeta_len, &dtype);

    *container = (iarray_container_t*)ina_mem_alloc(sizeof(iarray_container_t));
    INA_RETURN_IF_NULL(container);
    (*container)->catarr = catarr;

    // Build the dtshape
    (*container)->dtshape = (iarray_dtshape_t*)ina_mem_alloc(sizeof(iarray_dtshape_t));
    iarray_dtshape_t* dtshape = (*container)->dtshape;
    dtshape->dtype = dtype;
    dtshape->ndim = catarr->ndim;
    for (int i = 0; i < catarr->ndim; ++i) {
        dtshape->shape[i] = catarr->shape[i];
        dtshape->pshape[i] = catarr->pshape[i];
    }

    // Populate the frame
    (*container)->frame = (blosc2_frame*)ina_mem_alloc(sizeof(blosc2_frame));
    INA_FAIL_IF((*container)->frame == NULL);
    ina_mem_cpy((*container)->frame, catarr->sc->frame, sizeof(blosc2_frame));

    // Populate compression parameters
    blosc2_cparams *cparams;
    blosc2_get_cparams(catarr->sc, &cparams);
    blosc2_cparams *cparams2 = (blosc2_cparams*)ina_mem_alloc(sizeof(blosc2_cparams));
    memcpy(cparams2, cparams, sizeof(blosc2_cparams));
    free(cparams);
    (*container)->cparams = cparams2;  // we need an INA-allocated struct (to match INA_MEM_FREE_SAFE)
    blosc2_dparams *dparams;
    blosc2_get_dparams(catarr->sc, &dparams);
    blosc2_dparams *dparams2 = (blosc2_dparams*)ina_mem_alloc(sizeof(blosc2_dparams));
    memcpy(dparams2, dparams, sizeof(blosc2_dparams));
    free(dparams);
    (*container)->dparams = dparams2;  // we need an INA-allocated struct (to match INA_MEM_FREE_SAFE)

    (*container)->transposed = false;  // TODO: complete this

    (*container)->store = ina_mem_alloc(sizeof(_iarray_container_store_t));
    INA_FAIL_IF((*container)->store == NULL);
    (*container)->store->id = ina_str_new_fromcstr(store->id);

    return INA_SUCCESS;

fail:
    caterva_free_ctx(cat_ctx);
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


