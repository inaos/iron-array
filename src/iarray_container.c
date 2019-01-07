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
        return -1;
    }
    if (a->ndim != b->ndim) {
        return -1;
    }
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return -1;
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

INA_API(ina_rc_t) iarray_slice(iarray_context_t *ctx,
    iarray_container_t *c,
    uint64_t *start,
    uint64_t *stop,
    iarray_dtshape_t *dtshape,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);

    iarray_container_new(ctx, dtshape, store, flags, container);

    caterva_dims_t start_ = caterva_new_dims(start, c->dtshape->ndim);
    caterva_dims_t stop_ = caterva_new_dims(stop, c->dtshape->ndim);

    INA_FAIL_IF(caterva_get_slice((*container)->catarr, c->catarr, start_, stop_) != 0);

    return INA_SUCCESS;

fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_slice_buffer(iarray_container_t *c,
                                        uint64_t *start,
                                       uint64_t *stop,
                                       void *buffer)
{
    INA_VERIFY_NOT_NULL(start);
    INA_VERIFY_NOT_NULL(stop);

    uint8_t ndim = c->dtshape->ndim;

    uint64_t pshape[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        pshape[i] = stop[i] - start[i];
    }

    caterva_dims_t start_ = caterva_new_dims(start, ndim);
    caterva_dims_t stop_ = caterva_new_dims(stop, ndim);
    caterva_dims_t pshape_ = caterva_new_dims(pshape, ndim);

    INA_FAIL_IF(caterva_get_slice_buffer(buffer, c->catarr, start_, stop_, pshape_) != 0);

    return INA_SUCCESS;

    fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_squeeze(iarray_context_t *ctx,
                                 iarray_container_t *container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(container);

    INA_FAIL_IF(caterva_squeeze(container->catarr) != 0);

    if (container->dtshape->ndim != container->catarr->ndim) {
        container->dtshape->ndim = (uint8_t) container->catarr->ndim;
        for (int i = 0; i < container->catarr->ndim; ++i) {
            container->dtshape->shape[i] = container->catarr->shape[i];
            container->dtshape->pshape[i] = container->catarr->pshape[i];
        }
    }

    return INA_SUCCESS;

fail:
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_container_info(iarray_container_t *c,
    uint64_t *nbytes,
    uint64_t *cbytes)
{
    INA_VERIFY_NOT_NULL(c);

    *nbytes = (uint64_t) c->catarr->sc->nbytes;
    *cbytes = (uint64_t) c->catarr->sc->cbytes;

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_container_almost_equal(iarray_container_t *a, iarray_container_t *b, double tol) {
    if(a->dtshape->dtype != b->dtshape->dtype){
        return false;
    }
    if(a->catarr->size != b->catarr->size) {
        return false;
    }
    size_t size = a->catarr->size;

    uint8_t *buf_a = malloc(a->catarr->size * a->catarr->sc->typesize);
    caterva_to_buffer(a->catarr, buf_a);
    uint8_t *buf_b = malloc(b->catarr->size * b->catarr->sc->typesize);
    caterva_to_buffer(b->catarr, buf_b);

    if(a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        double *b_a = (double *)buf_a;
        double *b_b = (double *)buf_b;

        for (size_t i = 0; i < size; ++i) {
            double vdiff = fabs((b_a[i] - b_b[i]) / b_a[i]);
            if (vdiff > tol) {
                printf("%f, %f\n", b_a[i], b_b[i]);
                printf("Values differ in (%lu nelem) (diff: %f)\n", i, vdiff);
                free(buf_a);
                free(buf_b);
                return false;
            }
        }
        free(buf_a);
        free(buf_b);
        return true;
    }
    else if(a->dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        float *b_a = (float *)buf_a;
        float *b_b = (float *)buf_b;

        for (size_t i = 0; i < size; ++i) {
            double vdiff = fabs((double)(b_a[i] - b_b[i]) / b_a[i]);
            if (vdiff > tol) {
                printf("%f, %f\n", b_a[i], b_b[i]);
                printf("Values differ in (%lu nelem) (diff: %f)\n", i, vdiff);
                free(buf_a);
                free(buf_b);
                return false;
            }
        }
        free(buf_a);
        free(buf_b);
        return true;
    }
    printf("Data type is not supported");
    free(buf_a);
    free(buf_b);
    return false;
}

INA_API(void) iarray_container_free(iarray_context_t *ctx, iarray_container_t **container)
{
    INA_VERIFY_FREE(container);
    if ((*container)->catarr != NULL) {
        caterva_free_array((*container)->catarr);
    }
    INA_MEM_FREE_SAFE((*container)->frame);
    INA_MEM_FREE_SAFE((*container)->cparams);
    INA_MEM_FREE_SAFE((*container)->dparams);
    INA_MEM_FREE_SAFE((*container)->dtshape);
    INA_MEM_FREE_SAFE(*container);
}

INA_API(ina_rc_t) iarray_container_gt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_container_lt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_container_gte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_container_lte(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_container_eq(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}
