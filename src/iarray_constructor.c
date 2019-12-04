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

    ina_rc_t rc;

    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;

    IARRAY_FAIL_IF_ERROR(iarray_iter_write_new(ctx, &I, c, &val));

    while (INA_SUCCEED(iarray_iter_write_has_next(I))) {
        IARRAY_FAIL_IF_ERROR(iarray_iter_write_next(I));
        memcpy(val.elem_pointer, &value, sizeof(float));
    }
    IARRAY_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));
    iarray_iter_write_free(&I);

    rc = INA_SUCCESS;
    goto cleanup;

    fail:
        rc = ina_err_get_rc();
    cleanup:
        iarray_iter_write_free(&I);
        return rc;
}


static ina_rc_t _iarray_container_fill_double(iarray_context_t *ctx, iarray_container_t *c, double value)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(c);

    ina_rc_t rc;

    caterva_dims_t shape = caterva_new_dims(c->dtshape->shape, c->dtshape->ndim);

    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;

    IARRAY_FAIL_IF_ERROR(iarray_iter_write_new(ctx, &I, c, &val));

    while (INA_SUCCEED(iarray_iter_write_has_next(I))) {
        IARRAY_FAIL_IF_ERROR(iarray_iter_write_next(I));
        memcpy(val.elem_pointer, &value, sizeof(double));
    }
    IARRAY_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));
    iarray_iter_write_free(&I);

    rc = INA_SUCCESS;
    goto cleanup;

    fail:
    rc = ina_err_get_rc();
    cleanup:
    iarray_iter_write_free(&I);
    return rc;
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

    ina_rc_t rc;

    double contsize = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        contsize *= dtshape->shape[i];
    }

    double constant = (stop - start) / contsize;
    if (constant != step) {
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
    }

    IARRAY_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;

    IARRAY_FAIL_IF_ERROR(iarray_iter_write_new(ctx, &I, *container, &val));

    while (INA_SUCCEED(iarray_iter_write_has_next(I))) {
        IARRAY_FAIL_IF_ERROR(iarray_iter_write_next(I));

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
    IARRAY_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));
    iarray_iter_write_free(&I);

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
        iarray_container_free(ctx, container);
        rc = ina_err_get_rc();
    cleanup:
        iarray_iter_write_free(&I);
        return rc;
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

    ina_rc_t rc;

    double contsize = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        contsize *= dtshape->shape[i];
    }

    if (contsize != nelem) {
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
    }

    IARRAY_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    iarray_iter_write_t *I;
    iarray_iter_write_value_t val;

    IARRAY_FAIL_IF_ERROR(iarray_iter_write_new(ctx, &I, *container, &val));

    while (INA_SUCCEED(iarray_iter_write_has_next(I))) {
        IARRAY_FAIL_IF_ERROR(iarray_iter_write_next(I));

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
    IARRAY_FAIL_IF(ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
        iarray_container_free(ctx, container);
        rc = ina_err_get_rc();
    cleanup:
        iarray_iter_write_free(&I);
        return rc;
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

    ina_rc_t rc;

    IARRAY_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            IARRAY_FAIL_IF_ERROR(_iarray_container_fill_double(ctx, *container, 0.0));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            IARRAY_FAIL_IF_ERROR(_iarray_container_fill_float(ctx, *container, 0.0f));
            break;
        default:
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }
    rc = INA_SUCCESS;
    goto cleanup;
    fail:
        iarray_container_free(ctx, container);
        rc = ina_err_get_rc();
    cleanup:
        return rc;
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

    ina_rc_t rc;

    IARRAY_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    switch (dtshape->dtype) {
    case IARRAY_DATA_TYPE_DOUBLE:
        IARRAY_FAIL_IF_ERROR(_iarray_container_fill_double(ctx, *container, 1.0));
        break;
    case IARRAY_DATA_TYPE_FLOAT:
        IARRAY_FAIL_IF_ERROR(_iarray_container_fill_float(ctx, *container, 1.0f));
        break;
    default:
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_container_free(ctx, container);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
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

    ina_rc_t rc;

    IARRAY_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    IARRAY_FAIL_IF_ERROR(_iarray_container_fill_float(ctx, *container, value));

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_container_free(ctx, container);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
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

    ina_rc_t rc;

    IARRAY_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    IARRAY_FAIL_IF_ERROR(_iarray_container_fill_double(ctx, *container, value));

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_container_free(ctx, container);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
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

    ina_rc_t rc;
    IARRAY_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));

    switch ((*container)->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if ((* container)->catarr->size * (int64_t)sizeof(double) > buflen)
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if ((* container)->catarr->size * (int64_t)sizeof(float) > buflen)
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        default:
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    // TODO: would it be interesting to add a `buffer_len` parameter to `caterva_from_buffer()`?
    caterva_dims_t shape = caterva_new_dims((*container)->dtshape->shape, (*container)->dtshape->ndim);
    IARRAY_ERR_CATERVA(caterva_from_buffer((*container)->catarr, &shape, buffer));

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_container_free(ctx, container);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}


static ina_rc_t deserialize_meta(uint8_t *smeta, uint32_t smeta_len, iarray_data_type_t *dtype, bool *transposed) {
    INA_UNUSED(smeta_len);
    INA_VERIFY_NOT_NULL(smeta);
    INA_VERIFY_NOT_NULL(dtype);
    INA_VERIFY_NOT_NULL(transposed);
    ina_rc_t rc;

    uint8_t *pmeta = smeta;

    //version
    uint8_t version = *pmeta;
    pmeta +=1;

    // We only have an entry with the datatype (enumerated < 128)
    *dtype = *pmeta;
    pmeta += 1;

    *transposed = false;
    if ((*pmeta & 64ULL) != 0) {
        *transposed = true;
    }
    pmeta += 1;

    assert(pmeta - smeta == smeta_len);

    if (*dtype >= IARRAY_DATA_TYPE_MAX) {
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}

INA_API(ina_rc_t) iarray_from_file(iarray_context_t *ctx, iarray_store_properties_t *store,
                                   iarray_container_t **container, bool load_in_mem)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);

    ina_rc_t rc;
    caterva_ctx_t *cat_ctx = caterva_new_ctx(NULL, NULL, BLOSC2_CPARAMS_DEFAULTS, BLOSC2_DPARAMS_DEFAULTS);
    if (cat_ctx == NULL) {
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
    }

    caterva_array_t *catarr = caterva_from_file(cat_ctx, store->id, load_in_mem);
    if (catarr == NULL) {
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
    }

    uint8_t *smeta;
    uint32_t smeta_len;
    if (blosc2_get_metalayer(catarr->sc, "iarray", &smeta, &smeta_len) < 0) {
        fprintf(stderr, "Error in get_metalayer\n");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
    }
    iarray_data_type_t dtype;
    bool transposed;
    if (deserialize_meta(smeta, smeta_len, &dtype, &transposed) != 0) {
        fprintf(stderr, "Error in deserialize_meta\n");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
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
    if (blosc2_schunk_get_cparams(catarr->sc, &cparams) < 0) {
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
    }
    blosc2_cparams *cparams2 = (blosc2_cparams*)ina_mem_alloc(sizeof(blosc2_cparams));
    memcpy(cparams2, cparams, sizeof(blosc2_cparams));
    free(cparams);
    (*container)->cparams = cparams2;  // we need an INA-allocated struct (to match INA_MEM_FREE_SAFE)
    blosc2_dparams *dparams;
    if (blosc2_schunk_get_dparams(catarr->sc, &dparams) < 0) {
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
    }
    blosc2_dparams *dparams2 = (blosc2_dparams*)ina_mem_alloc(sizeof(blosc2_dparams));
    memcpy(dparams2, dparams, sizeof(blosc2_dparams));
    free(dparams);
    (*container)->dparams = dparams2;  // we need an INA-allocated struct (to match INA_MEM_FREE_SAFE)

    (*container)->transposed = transposed;  // TODO: complete this
    if (transposed) {
        int64_t aux[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < (*container)->dtshape->ndim; ++i) {
            aux[i] = (*container)->dtshape->shape[i];
        }
        for (int i = 0; i < (*container)->dtshape->ndim; ++i) {
            (*container)->dtshape->shape[i] = aux[(*container)->dtshape->ndim - 1 - i];
        }
        for (int i = 0; i < (*container)->dtshape->ndim; ++i) {
            aux[i] = (*container)->dtshape->pshape[i];
        }
        for (int i = 0; i < (*container)->dtshape->ndim; ++i) {
            (*container)->dtshape->pshape[i] = aux[(*container)->dtshape->ndim - 1 - i];
        }
    }
    (*container)->view = false;

    (*container)->store = ina_mem_alloc(sizeof(_iarray_container_store_t));
    if ((*container)->store == NULL) {
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_FAILED));
    }
    (*container)->store->id = ina_str_new_fromcstr(store->id);

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_container_free(ctx, container);
    rc = ina_err_get_rc();
    cleanup:
    caterva_free_ctx(cat_ctx);
    return rc;
}


INA_API(ina_rc_t) iarray_to_buffer(iarray_context_t *ctx,
    iarray_container_t *container,
    void *buffer,
    size_t buflen)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(buffer);
    INA_VERIFY_NOT_NULL(container);

    ina_rc_t rc;

    int64_t size = 1;
    for (int i = 0; i < container->dtshape->ndim; ++i) {
        size *= container->dtshape->shape[i];
    }

    switch (container->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if (size * (int64_t)sizeof(double) > buflen)
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if (size * (int64_t)sizeof(float) > buflen)
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            break;
        default:
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    if (container->view) {
        int64_t start[IARRAY_DIMENSION_MAX];
        int64_t stop[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < container->dtshape->ndim; ++i) {
            start[i] = 0;
            stop[i] = container->dtshape->shape[i];
        }
        IARRAY_FAIL_IF_ERROR(iarray_get_slice_buffer(ctx, container, start, stop, buffer, buflen));
    } else {
        IARRAY_ERR_CATERVA(caterva_to_buffer(container->catarr, buffer));
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
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}


INA_API(bool) iarray_is_empty(iarray_container_t *container) {
    INA_VERIFY_NOT_NULL(container);
    if (container->catarr->empty) {
        return true;
    }
    return false;
}


static void swap_store(void *dest, const void *pa, int size) {
    uint8_t* pa_ = (uint8_t*)pa;
    uint8_t* pa2_ = malloc((size_t)size);
    int i = 1;                    /* for big/little endian detection */
    char* p = (char*)&i;

    if (p[0] == 1) {
        /* little endian */
        switch (size) {
            case 8:
                pa2_[0] = pa_[7];
                pa2_[1] = pa_[6];
                pa2_[2] = pa_[5];
                pa2_[3] = pa_[4];
                pa2_[4] = pa_[3];
                pa2_[5] = pa_[2];
                pa2_[6] = pa_[1];
                pa2_[7] = pa_[0];
                break;
            case 4:
                pa2_[0] = pa_[3];
                pa2_[1] = pa_[2];
                pa2_[2] = pa_[1];
                pa2_[3] = pa_[0];
                break;
            case 2:
                pa2_[0] = pa_[1];
                pa2_[1] = pa_[0];
                break;
            case 1:
                pa2_[0] = pa_[0];
                break;
            default:
                fprintf(stderr, "Unhandled size: %d\n", size);
        }
    }
    memcpy(dest, pa2_, size);
    free(pa2_);
}

INA_API(ina_rc_t) iarray_to_sview(iarray_context_t *ctx, iarray_container_t *c, uint8_t **sview, int64_t *sview_len) {

    ina_rc_t rc;

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(sview);
    INA_VERIFY_NOT_NULL(sview_len);

    if (!c->view) {
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_INVALID_ARGUMENT));
    }
    *sview_len = 451;
    *sview = malloc(*sview_len);

    uint8_t *pview = *sview;

    // dtype
    *pview = (uint8_t) c->dtshape->dtype;
    pview += 1;

    // ndim
    *pview = 0xd0;
    pview += 1;
    *pview = (int8_t) c->dtshape->ndim;
    pview += 1;

    // shape
    *pview = 0x98;
    pview += 1;
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        *pview = 0xd3;
        pview += 1;
        swap_store(pview, &c->dtshape->shape[i], sizeof(int64_t));
        pview += sizeof(int64_t);
    }

    // pshape
    *pview = 0x98;
    pview += 1;
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        *pview = 0xd3;
        pview += 1;
        swap_store(pview, &c->dtshape->pshape[i], sizeof(int64_t));
        pview += sizeof(int64_t);
    }

    // offset
    *pview = 0x98;
    pview += 1;
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        *pview = 0xd3;
        pview += 1;
        swap_store(pview, &c->auxshape->offset[i], sizeof(int64_t));
        pview += sizeof(int64_t);
    }

    // shape_wos
    *pview = 0x98;
    pview += 1;
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        *pview = 0xd3;
        pview += 1;
        swap_store(pview, &c->auxshape->shape_wos[i], sizeof(int64_t));
        pview += sizeof(int64_t);
    }

    // pshape_wos
    *pview = 0x98;
    pview += 1;
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        *pview = 0xd3;
        pview += 1;
        swap_store(pview, &c->auxshape->pshape_wos[i], sizeof(int64_t));
        pview += sizeof(int64_t);
    }

    // index
    *pview = 0x98;
    pview += 1;
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        *pview = 0xd3;
        pview += 1;
        swap_store(pview, &c->auxshape->index[i], sizeof(int64_t));
        pview += sizeof(int64_t);
    }

    // catarr
    *pview = 0xcf;
    pview += 1;
    uint64_t address = (uint64_t) c->catarr;
    swap_store(pview, &address, sizeof(uint64_t));
    pview += sizeof(uint64_t);

    // transposed
    *pview = 0;
    if (c->transposed) {
        *pview = *pview | 64ULL;
    }
    pview += 1;

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}

INA_API(ina_rc_t) iarray_from_sview(iarray_context_t *ctx, uint8_t *sview, int64_t sview_len, iarray_container_t **c) {

    ina_rc_t rc;

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(sview);
    INA_VERIFY_NOT_NULL(c);

    *c = (iarray_container_t *) ina_mem_alloc(sizeof(iarray_container_t));
    (*c)->dtshape = (iarray_dtshape_t *) ina_mem_alloc(sizeof(iarray_dtshape_t));
    (*c)->auxshape = (iarray_auxshape_t *) ina_mem_alloc(sizeof(iarray_auxshape_t));
    (*c)->cparams = (blosc2_cparams *) ina_mem_alloc(sizeof(blosc2_cparams));
    (*c)->dparams = (blosc2_dparams *) ina_mem_alloc(sizeof(blosc2_dparams));

    //dtype
    uint8_t *pview = sview;
    (*c)->dtshape->dtype = (uint8_t) *pview;
    pview += 1;

    // ndim
    pview += 1;
    (*c)->dtshape->ndim = (int8_t) *pview;
    pview += 1;

    // shape
    pview += 1;
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        pview += 1;
        swap_store(&(*c)->dtshape->shape[i], pview, sizeof(int64_t));
        pview += sizeof(int64_t);
    }

    // pshape
    pview += 1;
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        pview += 1;
        swap_store(&(*c)->dtshape->pshape[i], pview, sizeof(int64_t));
        pview += sizeof(int64_t);
    }

    // offset
    pview += 1;
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        pview += 1;
        swap_store(&(*c)->auxshape->offset[i], pview, sizeof(int64_t));
        pview += sizeof(int64_t);
    }

    // shape_wos
    pview += 1;
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        pview += 1;
        swap_store(&(*c)->auxshape->shape_wos[i], pview, sizeof(int64_t));
        pview += sizeof(int64_t);

    }

    // pshape_wos
    pview += 1;
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        pview += 1;
        swap_store(&(*c)->auxshape->pshape_wos[i], pview, sizeof(int64_t));
        pview += sizeof(int64_t);
    }

    // index
    pview += 1;
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        pview += 1;
        swap_store(&(*c)->auxshape->index[i], pview, sizeof(int64_t));
        pview += sizeof(int64_t);
    }

    //catarr
    pview += 1;
    uint64_t address;
    swap_store(&address, pview, sizeof(int64_t));
    (*c)->catarr = (caterva_array_t *) address;
    pview += sizeof(uint64_t);

    // transposeD
    if ((*pview & 64ULL) != 0) {
        (*c)->transposed = true;
    } else {
        (*c)->transposed = false;
    }
    pview += 1;

    (*c)->view = true;
    memcpy((*c)->cparams, &(*c)->catarr->ctx->cparams, sizeof(blosc2_cparams));
    memcpy((*c)->dparams, &(*c)->catarr->ctx->dparams, sizeof(blosc2_dparams));

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}

INA_API(ina_rc_t) iarray_copy(iarray_context_t *ctx,
                              iarray_container_t *src,
                              bool view,
                              iarray_store_properties_t *store,
                              int flags,
                              iarray_container_t **dest) {

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(src);
    INA_VERIFY_NOT_NULL(dest);
    ina_rc_t rc;

    char* fname = NULL;
    if (flags & IARRAY_CONTAINER_PERSIST) {
        fname = (char*)store->id;
    }
    blosc2_frame *frame = blosc2_new_frame(fname);
    if (frame == NULL) {
        IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
    }
    (*dest) = (iarray_container_t *) ina_mem_alloc(sizeof(iarray_container_t));
    (*dest)->dtshape = (iarray_dtshape_t *) ina_mem_alloc(sizeof(iarray_dtshape_t));
    ina_mem_cpy((*dest)->dtshape, src->dtshape, sizeof(iarray_dtshape_t));
    (*dest)->view = view;
    (*dest)->transposed = src->transposed;
    (*dest)->cparams = (blosc2_cparams *) ina_mem_alloc(sizeof(blosc2_cparams));
    ina_mem_cpy((*dest)->cparams, src->cparams, sizeof(blosc2_cparams));
    (*dest)->dparams = (blosc2_dparams *) ina_mem_alloc(sizeof(blosc2_dparams));
    ina_mem_cpy((*dest)->dparams, src->dparams, sizeof(blosc2_dparams));

    if (src->view && !view) {
        (*dest)->auxshape = (iarray_auxshape_t *) ina_mem_alloc(sizeof(iarray_auxshape_t));
        for (int i = 0; i < (*dest)->dtshape->ndim; ++i) {
            (*dest)->auxshape->offset[i] = 0;
            (*dest)->auxshape->index[i] = i;
            (*dest)->auxshape->shape_wos[i] = src->dtshape->shape[i];
            (*dest)->auxshape->pshape_wos[i] = src->dtshape->pshape[i];
        }
    } else {
        (*dest)->auxshape = (iarray_auxshape_t *) ina_mem_alloc(sizeof(iarray_auxshape_t));
        ina_mem_cpy((*dest)->auxshape, src->auxshape, sizeof(iarray_auxshape_t));
    }
    if (view) {
        (*dest)->catarr = src->catarr;
    } else {
        caterva_ctx_t *cat_ctx = caterva_new_ctx(NULL, NULL, *(*dest)->cparams, *(*dest)->dparams);
        if (cat_ctx == NULL) {
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
        }

        if (src->catarr->storage == CATERVA_STORAGE_BLOSC) {
            int64_t pshape_[IARRAY_DIMENSION_MAX];
            for (int i = 0; i < src->catarr->ndim; ++i) {
                pshape_[i] = (int64_t) src->catarr->pshape[i];
            }
            caterva_dims_t pshape = caterva_new_dims(pshape_, src->catarr->ndim);
            (*dest)->catarr = caterva_empty_array(cat_ctx, frame, &pshape);
        } else {
            (*dest)->catarr = caterva_empty_array(cat_ctx, NULL, NULL);
        }
        if ((*dest)->catarr == NULL) {
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_CATERVA_FAILED));
        }

        if (src->view) {
            caterva_dims_t start = caterva_new_dims(src->auxshape->offset, src->catarr->ndim);
            int64_t stop_[IARRAY_DIMENSION_MAX];
            for (int i = 0; i < src->catarr->ndim; ++i) {
                stop_[i] = src->auxshape->offset[i] + src->auxshape->shape_wos[i];
            }
            caterva_dims_t stop = caterva_new_dims(stop_, src->catarr->ndim);
            IARRAY_ERR_CATERVA(caterva_get_slice((*dest)->catarr, src->catarr, &start, &stop));
            IARRAY_ERR_CATERVA(caterva_squeeze((*dest)->catarr));
        } else {
            IARRAY_ERR_CATERVA(caterva_copy((*dest)->catarr, src->catarr));
        }
    }

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_container_free(ctx, dest);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}
