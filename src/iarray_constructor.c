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
    INA_VERIFY_NOT_NULL(c);


    caterva_dims_t shape = caterva_new_dims(c->dtshape->shape, c->dtshape->ndim);

    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;

    INA_FAIL_IF_ERROR(iarray_iter_write_new(ctx, &I, c, &val));

    while (iarray_iter_write_has_next(I)) {
        INA_FAIL_IF_ERROR(iarray_iter_write_next(I));
        memcpy(val.elem_pointer, &value, sizeof(float));
    }

    return INA_SUCCESS;

    fail:
    return ina_err_get_rc();
}


static ina_rc_t _iarray_container_fill_double(iarray_context_t *ctx, iarray_container_t *c, double value)
{
    INA_VERIFY_NOT_NULL(c);

    caterva_dims_t shape = caterva_new_dims(c->dtshape->shape, c->dtshape->ndim);

    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;

    INA_FAIL_IF_ERROR(iarray_iter_write_new(ctx, &I, c, &val));

    while (iarray_iter_write_has_next(I)) {
        INA_FAIL_IF_ERROR(iarray_iter_write_next(I));
        memcpy(val.elem_pointer, &value, sizeof(double));
    }

    return INA_SUCCESS;
    fail:
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_arange(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    double start,
    double stop,
    double step,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    double contsize = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        contsize *= dtshape->shape[i];
    }

    double constant = (stop - start) / contsize;
    if (constant != step) {
        INA_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
    }

    INA_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;

    INA_FAIL_IF_ERROR(iarray_iter_write_new(ctx, &I, *container, &val));

    while (iarray_iter_write_has_next(I)) {
        INA_FAIL_IF_ERROR(iarray_iter_write_next(I));

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
    iarray_iter_write_free(&I);

    return INA_SUCCESS;

fail:
    iarray_iter_write_free(&I);
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_linspace(iarray_context_t *ctx,
                                  iarray_dtshape_t *dtshape,
                                  int64_t nelem,
                                  double start,
                                  double stop,
                                  iarray_store_properties_t *store,
                                  int flags,
                                  iarray_container_t **container)
{

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    double contsize = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        contsize *= dtshape->shape[i];
    }

    if (contsize != nelem) {
        INA_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
    }

    INA_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;

    INA_FAIL_IF_ERROR(iarray_iter_write_new(ctx, &I, *container, &val));

    while (iarray_iter_write_has_next(I)) {
        INA_FAIL_IF_ERROR(iarray_iter_write_next(I));

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
    iarray_iter_write_free(&I);

    return INA_SUCCESS;

fail:
    iarray_iter_write_free(&I);
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
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

    INA_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            INA_FAIL_IF_ERROR(_iarray_container_fill_double(ctx, *container, 0.0));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            INA_FAIL_IF_ERROR(_iarray_container_fill_float(ctx, *container, 0.0f));
            break;
        default:
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
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

    INA_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    switch (dtshape->dtype) {
    case IARRAY_DATA_TYPE_DOUBLE:
        INA_FAIL_IF_ERROR(_iarray_container_fill_double(ctx, *container, 1.0));
        break;
    case IARRAY_DATA_TYPE_FLOAT:
        INA_FAIL_IF_ERROR(_iarray_container_fill_float(ctx, *container, 1.0f));
        break;
    default:
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
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

    INA_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    INA_FAIL_IF_ERROR(_iarray_container_fill_float(ctx, *container, value));

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

    INA_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    INA_FAIL_IF_ERROR(_iarray_container_fill_double(ctx, *container, value));

    return INA_SUCCESS;

fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_from_buffer(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     void *buffer,
                                     size_t buflen,
                                     iarray_store_properties_t *store,
                                     int flags,
                                     iarray_container_t **container)
{

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(buffer);
    INA_VERIFY_NOT_NULL(container);

    INA_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    switch ((*container)->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if ((* container)->catarr->size * (int64_t)sizeof(double) > buflen)
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if ((* container)->catarr->size * (int64_t)sizeof(float) > buflen)
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        default:
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    // TODO: would it be interesting to add a `buffer_len` parameter to `caterva_from_buffer()`?
    caterva_dims_t shape = caterva_new_dims((*container)->dtshape->shape, (*container)->dtshape->ndim);
    if (caterva_from_buffer((*container)->catarr, &shape, buffer) != 0) {
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
    }

    return INA_SUCCESS;

fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}


static ina_rc_t deserialize_meta(uint8_t *smeta, uint32_t smeta_len, iarray_data_type_t *dtype)
{
    INA_UNUSED(smeta_len);
    INA_VERIFY_NOT_NULL(smeta);
    INA_VERIFY_NOT_NULL(dtype);

    uint8_t *pmeta = smeta;

    // We only have an entry with the datatype (enumerated < 128)
    *dtype = *pmeta;
    pmeta += 1;
    assert(pmeta - smeta == smeta_len);
    if (*dtype >= IARRAY_DATA_TYPE_MAX) {
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    return INA_SUCCESS;
    fail:
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_from_file(iarray_context_t *ctx, iarray_store_properties_t *store,
                                   iarray_container_t **container, bool load_in_mem)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);

    caterva_ctx_t *cat_ctx = caterva_new_ctx(NULL, NULL, BLOSC2_CPARAMS_DEFAULTS, BLOSC2_DPARAMS_DEFAULTS);

    caterva_array_t *catarr = caterva_from_file(cat_ctx, store->id, load_in_mem);
    if (catarr == NULL) {
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
    }

    uint8_t *smeta;
    uint32_t smeta_len;
    if (blosc2_get_metalayer(catarr->sc, "iarray", &smeta, &smeta_len) < 0) {
        printf("Error in get_metalayer\n");
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
    }
    iarray_data_type_t dtype;
    if (deserialize_meta(smeta, smeta_len, &dtype) != 0) {
        printf("Error in deserialize_meta\n");
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
    }

    *container = (iarray_container_t*)ina_mem_alloc(sizeof(iarray_container_t));
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

    // Build the auxshape
    (*container)->auxshape = (iarray_auxshape_t*)ina_mem_alloc(sizeof(iarray_auxshape_t));
    iarray_auxshape_t* auxshape = (*container)->auxshape;
    for (int8_t i = 0; i < catarr->ndim; ++i) {
        auxshape->index[i] = i;
        auxshape->offset[i] = 0;
        auxshape->shape_wos[i] = catarr->shape[i];
        auxshape->pshape_wos[i] = catarr->pshape[i];
    }

    // Populate compression parameters
    blosc2_cparams *cparams;
    blosc2_schunk_get_cparams(catarr->sc, &cparams);
    blosc2_cparams *cparams2 = (blosc2_cparams*)ina_mem_alloc(sizeof(blosc2_cparams));
    memcpy(cparams2, cparams, sizeof(blosc2_cparams));
    free(cparams);
    (*container)->cparams = cparams2;  // we need an INA-allocated struct (to match INA_MEM_FREE_SAFE)
    blosc2_dparams *dparams;
    blosc2_schunk_get_dparams(catarr->sc, &dparams);
    blosc2_dparams *dparams2 = (blosc2_dparams*)ina_mem_alloc(sizeof(blosc2_dparams));
    memcpy(dparams2, dparams, sizeof(blosc2_dparams));
    free(dparams);
    (*container)->dparams = dparams2;  // we need an INA-allocated struct (to match INA_MEM_FREE_SAFE)

    (*container)->transposed = false;  // TODO: complete this
    (*container)->view = false;

    (*container)->store = ina_mem_alloc(sizeof(_iarray_container_store_t));
    if ((*container)->store == NULL) {
        INA_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }
    (*container)->store->id = ina_str_new_fromcstr(store->id);

    return INA_SUCCESS;

fail:
    caterva_free_ctx(cat_ctx);
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_to_buffer(iarray_context_t *ctx,
    iarray_container_t *container,
    void *buffer,
    size_t buflen)
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
            if (size * (int64_t)sizeof(double) > buflen)
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if (size * (int64_t)sizeof(float) > buflen)
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        default:
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    if (container->view) {
        int64_t start[IARRAY_DIMENSION_MAX];
        int64_t stop[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < container->dtshape->ndim; ++i) {
            start[i] = 0;
            stop[i] = container->dtshape->shape[i];
        }
        INA_FAIL_IF_ERROR(iarray_get_slice_buffer(ctx, container, start, stop, buffer, buflen));
    } else {
        if (caterva_to_buffer(container->catarr, buffer) != 0) {
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
        }
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
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }
    }

    return INA_SUCCESS;
    fail:
    return ina_err_get_rc();
}


INA_API(bool) iarray_is_empty(iarray_container_t *container) {
    INA_VERIFY_NOT_NULL(container);

    // TODO: Change this condition when an empty array would be of size 0
    if (container->catarr->empty)
    {
        return true;
    }
    return false;
}
