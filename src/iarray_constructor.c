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
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    ina_rc_t rc;

    double contsize = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        contsize *= dtshape->shape[i];
    }

    double constant = (stop - start) / contsize;
    if (constant != step) {
        IARRAY_TRACE1(iarray.error, "The step parameter is invalid");
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
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    ina_rc_t rc;

    double contsize = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        contsize *= dtshape->shape[i];
    }

    if (contsize != nelem) {
        IARRAY_TRACE1(iarray.error, "The nelem parameter is invalid");
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
    INA_VERIFY_NOT_NULL(store);
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
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
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
    INA_VERIFY_NOT_NULL(store);
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
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
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
    INA_VERIFY_NOT_NULL(store);
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
    INA_VERIFY_NOT_NULL(store);
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
                                     int64_t buflen,
                                     iarray_store_properties_t *store,
                                     int flags,
                                     iarray_container_t **container)
{

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(buffer);
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(container);

    ina_rc_t rc;
    IARRAY_FAIL_IF_ERROR(iarray_container_new(ctx, dtshape, store, flags, container));
    (*container)->catarr->empty = false;

    switch ((*container)->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            if ((* container)->catarr->size * (int64_t) sizeof(double) > buflen) {
                IARRAY_TRACE1(iarray.error, "The size of the buffer is not enough");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if ((* container)->catarr->size * (int64_t) sizeof(float) > buflen) {
                IARRAY_TRACE1(iarray.error, "The size of the buffer is not enough");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            }
            break;
        default:
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }

    // TODO: would it be interesting to add a `buffer_len` parameter to `caterva_from_buffer()`?
    caterva_config_t cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
    caterva_params_t params = {0};
    iarray_create_caterva_params(dtshape, &params);
    caterva_storage_t storage = {0};
    iarray_create_caterva_storage(dtshape, store, &storage);
    caterva_context_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

    uint8_t *smeta = NULL;
    if (storage.backend == CATERVA_STORAGE_BLOSC) {
        storage.properties.blosc.nmetalayers = 1;
        storage.properties.blosc.metalayers[0].name = "iarray";
        uint32_t smeta_len;
        blosc2_get_metalayer((*container)->catarr->sc, "iarray", &smeta, &smeta_len);
        storage.properties.blosc.metalayers[0].sdata = smeta;
        storage.properties.blosc.metalayers[0].size = smeta_len;
    }
    IARRAY_ERR_CATERVA(caterva_array_free(cat_ctx, &(*container)->catarr));

    IARRAY_ERR_CATERVA(caterva_array_from_buffer(cat_ctx, buffer, buflen, &params, &storage, &(*container)->catarr));

    if (storage.backend == CATERVA_STORAGE_BLOSC) {
        free(smeta);
    }
    (*container)->catarr->empty = false;

    IARRAY_ERR_CATERVA(caterva_context_free(&cat_ctx));

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_container_free(ctx, container);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}


INA_API(ina_rc_t) iarray_to_buffer(iarray_context_t *ctx,
                                   iarray_container_t *container,
                                   void *buffer,
                                   int64_t buflen)
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
            if (size * (int64_t) sizeof(double) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            }
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            if (size * (int64_t) sizeof(float) > buflen) {
                IARRAY_TRACE1(iarray.error, "The buffer size is not enough");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_TOO_SMALL_BUFFER));
            }
            break;
        default:
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
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
//            case 4:
//                pa2_[0] = pa_[3];
//                pa2_[1] = pa_[2];
//                pa2_[2] = pa_[1];
//                pa2_[3] = pa_[0];
//                break;
//            case 2:
//                pa2_[0] = pa_[1];
//                pa2_[1] = pa_[0];
//                break;
//            case 1:
//                pa2_[0] = pa_[0];
//                break;
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
        IARRAY_TRACE1(iarray.error, "The container is not a view");
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

    INA_UNUSED(sview_len);

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

    blosc2_cparams cparams = {0};
    blosc2_dparams dparams = {0};
    int blosc_filter_idx = 0;
    cparams.compcode = ctx->cfg->compression_codec;
    cparams.use_dict = ctx->cfg->use_dict;
    cparams.clevel = (uint8_t)ctx->cfg->compression_level; /* Since its just a mapping, we know the cast is ok */
    cparams.blocksize = ctx->cfg->blocksize;
    cparams.nthreads = (uint16_t)ctx->cfg->max_num_threads; /* Since its just a mapping, we know the cast is ok */
    if ((ctx->cfg->filter_flags & IARRAY_COMP_TRUNC_PREC)) {
        cparams.filters[blosc_filter_idx] = BLOSC_TRUNC_PREC;
        cparams.filters_meta[blosc_filter_idx] = ctx->cfg->fp_mantissa_bits;
        blosc_filter_idx++;
    }
    if (ctx->cfg->filter_flags & IARRAY_COMP_BITSHUFFLE) {
        cparams.filters[blosc_filter_idx] = BLOSC_BITSHUFFLE;
        blosc_filter_idx++;
    }
    if (ctx->cfg->filter_flags & IARRAY_COMP_SHUFFLE) {
        cparams.filters[blosc_filter_idx] = BLOSC_SHUFFLE;
        blosc_filter_idx++;
    }
    if (ctx->cfg->filter_flags & IARRAY_COMP_DELTA) {
        cparams.filters[blosc_filter_idx] = BLOSC_DELTA;
        blosc_filter_idx++;
    }
    dparams.nthreads = (uint16_t)ctx->cfg->max_num_threads; /* Since its just a mapping, we know the cast is ok */

    memcpy((*c)->cparams, &cparams, sizeof(blosc2_cparams));
    memcpy((*c)->dparams, &dparams, sizeof(blosc2_dparams));

    rc = INA_SUCCESS;
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
    INA_VERIFY_NOT_NULL(store);
    INA_VERIFY_NOT_NULL(dest);

    INA_UNUSED(flags);
    ina_rc_t rc;

    caterva_config_t cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
    caterva_context_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_context_new(&cfg, &cat_ctx));

    (*dest) = (iarray_container_t *) ina_mem_alloc(sizeof(iarray_container_t));
    (*dest)->dtshape = (iarray_dtshape_t *) ina_mem_alloc(sizeof(iarray_dtshape_t));
    ina_mem_cpy((*dest)->dtshape, src->dtshape, sizeof(iarray_dtshape_t));
    (*dest)->view = view;
    (*dest)->transposed = src->transposed;
    if ((*dest)->view) {
        (*dest)->cparams = src->cparams;
        (*dest)->dparams = src->dparams;
        (*dest)->store = src->store;
    } else {
        (*dest)->cparams = (blosc2_cparams *) ina_mem_alloc(sizeof(blosc2_cparams));
        ina_mem_cpy((*dest)->cparams, src->cparams, sizeof(blosc2_cparams));
        (*dest)->dparams = (blosc2_dparams *) ina_mem_alloc(sizeof(blosc2_dparams));
        ina_mem_cpy((*dest)->dparams, src->dparams, sizeof(blosc2_dparams));
        (*dest)->store = (iarray_store_properties_t *) ina_mem_alloc(sizeof(iarray_store_properties_t));
        ina_mem_cpy((*dest)->store, store, sizeof(iarray_store_properties_t));
    }

    if (src->view && !view) {
        (*dest)->auxshape = (iarray_auxshape_t *) ina_mem_alloc(sizeof(iarray_auxshape_t));
        for (int i = 0; i < (*dest)->dtshape->ndim; ++i) {
            (*dest)->auxshape->offset[i] = 0;
            (*dest)->auxshape->index[i] = (int8_t) i;
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
        caterva_params_t params = {0};
        iarray_create_caterva_params(src->dtshape, &params);

        caterva_storage_t storage = {0};
        iarray_create_caterva_storage(src->dtshape, store, &storage);

        if (src->view) {
            int64_t *start = src->auxshape->offset;
            int64_t stop[IARRAY_DIMENSION_MAX];
            for (int i = 0; i < src->catarr->ndim; ++i) {
                stop[i] = src->auxshape->offset[i] + src->auxshape->shape_wos[i];
            }

            IARRAY_ERR_CATERVA(caterva_array_get_slice(cat_ctx, src->catarr, start, stop, &storage, &(*dest)->catarr));
            IARRAY_ERR_CATERVA(caterva_array_squeeze(cat_ctx, (*dest)->catarr));
        } else {
            IARRAY_ERR_CATERVA(caterva_array_copy(cat_ctx, src->catarr, &storage, &(*dest)->catarr));
        }
    }

    caterva_context_free(&cat_ctx);

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    iarray_container_free(ctx, dest);
    rc = ina_err_get_rc();
    cleanup:
    return rc;
}
