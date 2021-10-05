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
#include "iarray_constructor.h"
#include <mkl_vsl.h>


INA_API(ina_rc_t) iarray_random_ctx_new(iarray_context_t *ctx,
    uint32_t seed,
    iarray_random_rng_t rng,
    iarray_random_ctx_t **rng_ctx)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(rng_ctx);
    *rng_ctx = (iarray_random_ctx_t*)ina_mem_alloc(sizeof(iarray_random_ctx_t));
    (*rng_ctx)->seed = seed;
    (*rng_ctx)->rng = rng;

    int mkl_rng;
    switch (rng) {
        case IARRAY_RANDOM_RNG_MRG32K3A:
            mkl_rng = VSL_BRNG_MRG32K3A;
            break;
        default:
            IARRAY_TRACE1(iarray.error, "Random generator not supported");
            return IARRAY_ERR_INVALID_RNG_METHOD;
    }

    vslNewStream(&(*rng_ctx)->stream, mkl_rng, seed);
    ina_mem_set((*rng_ctx)->dparams, 0, sizeof(double)*(IARRAY_RANDOM_DIST_PARAM_SENTINEL));
    ina_mem_set((*rng_ctx)->fparams, 0, sizeof(float)*(IARRAY_RANDOM_DIST_PARAM_SENTINEL));
    return INA_SUCCESS;
}

INA_API(void) iarray_random_ctx_free(iarray_context_t *ctx, iarray_random_ctx_t **rng_ctx)
{
    INA_VERIFY_FREE(rng_ctx);
    INA_UNUSED(ctx);
    vslDeleteStream(&((*rng_ctx)->stream));
    INA_MEM_FREE_SAFE(*rng_ctx);
}

INA_API(ina_rc_t) iarray_random_dist_set_param_float(iarray_random_ctx_t *ctx,
    iarray_random_dist_parameter_t key,
    float value)
{
    INA_VERIFY_NOT_NULL(ctx);
    ctx->fparams[key] = value;
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_random_dist_set_param_double(iarray_random_ctx_t *ctx,
    iarray_random_dist_parameter_t key,
    double value)
{
    INA_VERIFY_NOT_NULL(ctx);
    ctx->dparams[key] = value;
    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_random_kstest(iarray_context_t *ctx,
                                       iarray_container_t *container1,
                                       iarray_container_t *container2,
                                       bool *res)
{

    if (container1->catarr->nitems != container2->catarr->nitems) {
        return INA_ERROR(IARRAY_ERR_INVALID_SHAPE);
    }

    int64_t size = container1->catarr->nitems;

    int nbins = 100;
    double bins[100];
    double hist1[100];
    double hist2[100];

    double max = -INFINITY;
    double min = INFINITY;

    iarray_iter_read_t *iter;
    iarray_iter_read_value_t val;
    IARRAY_RETURN_IF_FAILED(iarray_iter_read_new(ctx, &iter, container1, &val));

    while (INA_SUCCEED(iarray_iter_read_has_next(iter))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_read_next(iter));

        double data = 0.0;
        switch(container1->dtshape->dtype){
            case IARRAY_DATA_TYPE_DOUBLE:
                data = ((double *) val.elem_pointer)[0];
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                data = ((float *) val.elem_pointer)[0];
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                return (INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }

        max = (data > max) ? data : max;
        min = (data < min) ? data : min;
    }
    IARRAY_ITER_FINISH();

    iarray_iter_read_free(&iter);

    IARRAY_RETURN_IF_FAILED(iarray_iter_read_new(ctx, &iter, container2, &val));
    while (INA_SUCCEED(iarray_iter_read_has_next(iter))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_read_next(iter));

        double data = 0.0;
        switch(container1->dtshape->dtype){
            case IARRAY_DATA_TYPE_DOUBLE:
                data = ((double *) val.elem_pointer)[0];
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                data = ((float *) val.elem_pointer)[0];
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                return (INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }

        max = (data > max) ? data : max;
        min = (data < min) ? data : min;
    }
    iarray_iter_read_free(&iter);
    IARRAY_ITER_FINISH();

    for (int i = 0; i < nbins; ++i) {
        bins[i] = min + (max-min)/nbins * (i+1);
        hist1[i] = 0;
        hist2[i] = 0;
    }

    IARRAY_RETURN_IF_FAILED(iarray_iter_read_new(ctx, &iter, container1, &val));

    while (INA_SUCCEED(iarray_iter_read_has_next(iter))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_read_next(iter));

        double data = 0;
        switch(container1->dtshape->dtype){
            case IARRAY_DATA_TYPE_DOUBLE:
                data = ((double *) val.elem_pointer)[0];
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                data = ((float *) val.elem_pointer)[0];
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                return (INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }

        for (int i = 0; i < nbins; ++i) {
            if (data <= bins[i]) {
                hist1[i] += 1;
                break;
            }
        }
    }
    iarray_iter_read_free(&iter);
    IARRAY_ITER_FINISH();

    IARRAY_RETURN_IF_FAILED(iarray_iter_read_new(ctx, &iter, container2, &val));

    while (INA_SUCCEED(iarray_iter_read_has_next(iter))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_read_next(iter));

        double data = 0;
        switch(container1->dtshape->dtype){
            case IARRAY_DATA_TYPE_DOUBLE:
                data = ((double *) val.elem_pointer)[0];
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                data = ((float *) val.elem_pointer)[0];
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                return (INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }
        for (int i = 0; i < nbins; ++i) {
            if (data <= bins[i]) {
                hist2[i] += 1;
                break;
            }
        }
    }
    iarray_iter_read_free(&iter);
    IARRAY_ITER_FINISH();

    for (int i = 1; i < nbins; ++i) {
        hist1[i] += hist1[i-1];
        hist2[i] += hist2[i-1];
    }

    double max_dif = -INFINITY;
    for (int i = 0; i < nbins; ++i) {
        max_dif = (fabs(hist1[i] - hist2[i]) / size > max_dif) ? fabs(hist1[i] - hist2[i]) / size : max_dif;
    }

    double a = 0.001;
    double threshold = sqrt(- log(a) / 2) * sqrt(2 * ((double) size) / (size * size));

    *res = (max_dif < threshold);
    return INA_SUCCESS;
}
