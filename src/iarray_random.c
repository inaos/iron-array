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


typedef enum _iarray_random_method_e {
    _IARRAY_RANDOM_METHOD_UNIFORM,
    _IARRAY_RANDOM_METHOD_GAUSSIAN,
    _IARRAY_RANDOM_METHOD_BETA,
    _IARRAY_RANDOM_METHOD_LOGNORMAL,
    _IARRAY_RANDOM_METHOD_EXPONENTIAL,
} _iarray_random_method_t;

struct iarray_random_ctx_s {
    iarray_random_rng_t rng;
    uint32_t seed;
    VSLStreamStatePtr stream;
    double dparams[IARRAY_RANDOM_DIST_PARAM_SENTINEL];
    float fparams[IARRAY_RANDOM_DIST_PARAM_SENTINEL];
};

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
        case IARRAY_RANDOM_RNG_MERSENNE_TWISTER:
            mkl_rng = VSL_BRNG_SFMT19937;
            break;
        case IARRAY_RANDOM_RNG_SOBOL:
            mkl_rng = VSL_BRNG_SOBOL;
            break;
        default:
            INA_FAIL_IF(1);
    }

    vslNewStream(&(*rng_ctx)->stream, mkl_rng, seed);

    ina_mem_set((*rng_ctx)->dparams, 0, sizeof(double)*(IARRAY_RANDOM_DIST_PARAM_SENTINEL));
    ina_mem_set((*rng_ctx)->fparams, 0, sizeof(float)*(IARRAY_RANDOM_DIST_PARAM_SENTINEL));

    return INA_SUCCESS;

fail:
    iarray_random_ctx_free(ctx, rng_ctx);
    return INA_ERR_ILLEGAL;
}

INA_API(void) iarray_random_ctx_free(iarray_context_t *ctx, iarray_random_ctx_t **rng_ctx)
{
    INA_ASSERT_NOT_NULL(ctx);
    INA_VERIFY_FREE(rng_ctx);
    INA_UNUSED(ctx);
    vslDeleteStream(&((*rng_ctx)->stream));
    INA_MEM_FREE_SAFE(*rng_ctx);
}

INA_API(ina_rc_t) iarray_random_dist_set_param_float(iarray_random_ctx_t *ctx,
    iarray_random_dist_parameter_t key,
    float value)
{
    INA_ASSERT_NOT_NULL(ctx);
    ctx->fparams[key] = value;
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_random_dist_set_param_double(iarray_random_ctx_t *ctx,
    iarray_random_dist_parameter_t key,
    double value)
{
    INA_ASSERT_NOT_NULL(ctx);
    ctx->dparams[key] = value;
    return INA_SUCCESS;
}

static ina_rc_t _iarray_rand_internal(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *random_ctx,
    iarray_container_t *container,
    _iarray_random_method_t method)
{
    int status = VSL_ERROR_OK;
    iarray_iter_write_part_t *iter;
    iarray_iter_write_part_new(ctx, container, &iter);

    int64_t max_part_size = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        max_part_size *= dtshape->pshape[i];
    }
    void *buffer_mem = ina_mem_alloc(max_part_size * sizeof(double));

    for (iarray_iter_write_part_init(iter);
        !iarray_iter_write_part_finished(iter);
        iarray_iter_write_part_next(iter)) {

        iarray_iter_write_part_value_t val;
        iarray_iter_write_part_value(iter, &val);

        int64_t part_size = 1;
        for (int i = 0; i < dtshape->ndim; ++i) {
            part_size *= val.part_shape[i];
        }

        if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
            float *r = (float*)buffer_mem;
            switch (method) {
                case _IARRAY_RANDOM_METHOD_UNIFORM: {
                    float a = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_A];
                    float b = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_B];
                    status = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, random_ctx->stream,
                                          (int) part_size, r, a, b);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_GAUSSIAN: {
                    float mu = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_MU];
                    float sigma = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_SIGMA];
                    status = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, random_ctx->stream,
                                           (int) part_size, r, mu, sigma);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_BETA: {
                    float alpha = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_ALPHA];
                    float beta = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_BETA];
                    status = vsRngBeta(VSL_RNG_METHOD_BETA_CJA, random_ctx->stream, (int) part_size, r, alpha, beta, 0, 1);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_LOGNORMAL: {
                    float mu = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_MU];
                    float sigma = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_SIGMA];
                    status = vsRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2, random_ctx->stream, (int) part_size, r, mu, sigma, 0, 1);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_EXPONENTIAL: {
                    float beta = random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_BETA];
                    status = vsRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF, random_ctx->stream, (int) part_size, r, 0, beta);
                    break;
                }
            }
            INA_FAIL_IF(status != VSL_ERROR_OK);

            for (int64_t i = 0; i < part_size; ++i) {
                ((float *)val.pointer)[i] = r[i];
            }
        }
        else {
            double *r = (double*)buffer_mem;
            switch (method) {
                case _IARRAY_RANDOM_METHOD_UNIFORM: {
                    double a = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_A];
                    double b = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_B];
                    status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, random_ctx->stream,
                                          (int) part_size, r, a, b);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_GAUSSIAN: {
                    double mu = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_MU];
                    double sigma = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_SIGMA];
                    status = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, random_ctx->stream,
                                           (int) part_size, r, mu, sigma);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_BETA: {
                    double alpha = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_ALPHA];
                    double beta = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_BETA];
                    status = vdRngBeta(VSL_RNG_METHOD_BETA_CJA, random_ctx->stream, (int) part_size, r, alpha, beta, 0, 1);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_LOGNORMAL: {
                    double mu = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_MU];
                    double sigma = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_SIGMA];
                    status = vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2, random_ctx->stream, (int) part_size, r, mu, sigma, 0, 1);
                    break;
                }
                case _IARRAY_RANDOM_METHOD_EXPONENTIAL: {
                    double beta = random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_BETA];
                    status = vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF, random_ctx->stream, (int) part_size, r, 0, beta);
                    break;
                }
            }
            INA_FAIL_IF(status != VSL_ERROR_OK);

            for (int64_t i = 0; i < part_size; ++i) {
                ((double *)val.pointer)[i] = r[i];
            }
        }
    }

    return INA_SUCCESS;

fail:
    return INA_ERR_ILLEGAL;
}

INA_API(ina_rc_t) iarray_random_rand(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *random_ctx,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        iarray_random_dist_set_param_float(random_ctx, IARRAY_RANDOM_DIST_PARAM_A, 0.0f);
        iarray_random_dist_set_param_float(random_ctx, IARRAY_RANDOM_DIST_PARAM_B, 1.0f);
    }
    else {
        iarray_random_dist_set_param_double(random_ctx, IARRAY_RANDOM_DIST_PARAM_A, 0.0);
        iarray_random_dist_set_param_double(random_ctx, IARRAY_RANDOM_DIST_PARAM_B, 1.0);
    }

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_UNIFORM);
}

INA_API(ina_rc_t) iarray_random_randn(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *random_ctx,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(container);

    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        iarray_random_dist_set_param_float(random_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 0.0f);
        iarray_random_dist_set_param_float(random_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 1.0f);
    }
    else {
        iarray_random_dist_set_param_double(random_ctx, IARRAY_RANDOM_DIST_PARAM_MU, 0.0);
        iarray_random_dist_set_param_double(random_ctx, IARRAY_RANDOM_DIST_PARAM_SIGMA, 1.0);
    }

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_GAUSSIAN);
}

INA_API(ina_rc_t) iarray_random_beta(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *random_ctx,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        INA_FAIL_IF(random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_ALPHA] <= 0);
        INA_FAIL_IF(random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_BETA] <= 0);
    }
    else {
        INA_FAIL_IF(random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_ALPHA] <= 0);
        INA_FAIL_IF(random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_BETA] <= 0);
    }

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_BETA);
fail:
    return INA_ERR_MISSING;
}

INA_API(ina_rc_t) iarray_random_lognormal(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_random_ctx_t *random_ctx,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        INA_FAIL_IF(random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_SIGMA] <= 0);
    }
    else {
        INA_FAIL_IF(random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_SIGMA] <= 0);
    }

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_LOGNORMAL);

    fail:
    return INA_ERR_MISSING;
}

INA_API(ina_rc_t) iarray_random_exponential(iarray_context_t *ctx,
                                           iarray_dtshape_t *dtshape,
                                           iarray_random_ctx_t *random_ctx,
                                           iarray_store_properties_t *store,
                                           int flags,
                                           iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        INA_FAIL_IF(random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_BETA] <= 0);
    }
    else {
        INA_FAIL_IF(random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_BETA] <= 0);
    }

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_EXPONENTIAL);

    fail:
    return INA_ERR_MISSING;
}

INA_API(ina_rc_t) iarray_random_uniform(iarray_context_t *ctx,
                                        iarray_dtshape_t *dtshape,
                                        iarray_random_ctx_t *random_ctx,
                                        iarray_store_properties_t *store,
                                        int flags,
                                        iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        INA_FAIL_IF(random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_A] >= random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_B]);
    }
    else {
        INA_FAIL_IF(random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_A] >= random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_B]);
    }

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_UNIFORM);

    fail:
    return INA_ERR_MISSING;
}

INA_API(ina_rc_t) iarray_random_normal(iarray_context_t *ctx,
                                       iarray_dtshape_t *dtshape,
                                       iarray_random_ctx_t *random_ctx,
                                       iarray_store_properties_t *store,
                                       int flags,
                                       iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(random_ctx);
    INA_VERIFY_NOT_NULL(container);

    /* validate distribution parameters */
    if (dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        INA_FAIL_IF(random_ctx->fparams[IARRAY_RANDOM_DIST_PARAM_SIGMA] <= 0);
    }
    else {
        INA_FAIL_IF(random_ctx->dparams[IARRAY_RANDOM_DIST_PARAM_SIGMA] <= 0);
    }

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    return _iarray_rand_internal(ctx, dtshape, random_ctx, *container, _IARRAY_RANDOM_METHOD_GAUSSIAN);

    fail:
    return INA_ERR_MISSING;
}


INA_API(ina_rc_t) iarray_random_kstest(iarray_context_t *ctx,
                                       iarray_container_t *c1,
                                       iarray_container_t *c2,
                                       bool *res)
{

    INA_ASSERT_SUCCEED(c1->catarr->size != c2->catarr->size);
    int64_t size = c1->catarr->size;

    int nbins = 100;
    double bins[100];
    double hist1[100];
    double hist2[100];

    double max = -INFINITY;
    double min = INFINITY;

    iarray_iter_read_t *iter;

    iarray_iter_read_new(ctx, c1, &iter);
    for (iarray_iter_read_init(iter);
         !iarray_iter_read_finished(iter);
         iarray_iter_read_next(iter)) {

        iarray_iter_read_value_t val;
        iarray_iter_read_value(iter, &val);

        double data;
        switch(c1->dtshape->dtype){
            case IARRAY_DATA_TYPE_DOUBLE:
                data = ((double *) val.pointer)[0];
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                data = ((float *) val.pointer)[0];
                break;
            default:
                return INA_ERR_MISSING;
        }

        max = (data > max) ? data : max;
        min = (data < min) ? data : min;
    }
    iarray_iter_read_free(iter);

    iarray_iter_read_new(ctx, c2, &iter);
    for (iarray_iter_read_init(iter);
         !iarray_iter_read_finished(iter);
         iarray_iter_read_next(iter)) {

        iarray_iter_read_value_t val;
        iarray_iter_read_value(iter, &val);

        double data;
        switch(c1->dtshape->dtype){
            case IARRAY_DATA_TYPE_DOUBLE:
                data = ((double *) val.pointer)[0];
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                data = ((float *) val.pointer)[0];
                break;
            default:
                return INA_ERR_MISSING;
        }

        max = (data > max) ? data : max;
        min = (data < min) ? data : min;
    }
    iarray_iter_read_free(iter);

    for (int i = 0; i < nbins; ++i) {
        bins[i] = min + (max-min)/nbins * (i+1);
        hist1[i] = 0;
        hist2[i] = 0;
    }

    iarray_iter_read_new(ctx, c1, &iter);
    for (iarray_iter_read_init(iter);
         !iarray_iter_read_finished(iter);
         iarray_iter_read_next(iter)) {

        iarray_iter_read_value_t val;
        iarray_iter_read_value(iter, &val);

        double data;
        switch(c1->dtshape->dtype){
            case IARRAY_DATA_TYPE_DOUBLE:
                data = ((double *) val.pointer)[0];
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                data = ((float *) val.pointer)[0];
                break;
            default:
                return INA_ERR_MISSING;
        }

        for (int i = 0; i < nbins; ++i) {
            if (data <= bins[i]) {
                hist1[i] += 1;
                break;
            }
        }
    }
    iarray_iter_read_free(iter);

    iarray_iter_read_new(ctx, c2, &iter);
    for (iarray_iter_read_init(iter);
         !iarray_iter_read_finished(iter);
         iarray_iter_read_next(iter)) {

        iarray_iter_read_value_t val;
        iarray_iter_read_value(iter, &val);

        double data;
        switch(c1->dtshape->dtype){
            case IARRAY_DATA_TYPE_DOUBLE:
                data = ((double *) val.pointer)[0];
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                data = ((float *) val.pointer)[0];
                break;
            default:
                return INA_ERR_MISSING;
        }
        for (int i = 0; i < nbins; ++i) {
            if (data <= bins[i]) {
                hist2[i] += 1;
                break;
            }
        }
    }
    iarray_iter_read_free(iter);

    for (int i = 1; i < nbins; ++i) {
        hist1[i] += hist1[i-1];
        hist2[i] += hist2[i-1];
    }

    double max_dif = -INFINITY;
    for (int i = 0; i < nbins; ++i) {
        max_dif = (fabs(hist1[i] - hist2[i]) / size > max_dif) ? fabs(hist1[i] - hist2[i]) / size : max_dif;
    }

    double a = 0.01;
    double threshold = sqrt(- log(a) / 2) * sqrt(2 * ((double) size) / (size * size));

    *res = (max_dif < threshold);
    return INA_SUCCESS;
}