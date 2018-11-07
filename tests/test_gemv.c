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

#include "test_common.h"

#define NTHREADS 1

#define KB  1024
#define MB  (1024*KB)
#define GB  (1024*MB)

int test_gemv(iarray_container_t *c_x, iarray_container_t *c_y, iarray_container_t *c_out, iarray_container_t *c_res) {
    iarray_gemv(c_x, c_y, c_out);
    if (iarray_equal_data(c_out, c_res) != 0) {
        return -1;
    }
    return 1;
}

INA_TEST_DATA(e_gemv) {
    int tests_run;

    blosc2_cparams cparams;
    blosc2_dparams dparams;

};

INA_TEST_SETUP(e_gemv) {

    blosc_init();

    data->cparams = BLOSC_CPARAMS_DEFAULTS;
    data->dparams = BLOSC_DPARAMS_DEFAULTS;

    data->cparams.compcode = BLOSC_LZ4;
    data->cparams.nthreads = NTHREADS;
    data->dparams.nthreads = NTHREADS;

}


INA_TEST_TEARDOWN(e_gemv)
{
    blosc_destroy();
}

INA_TEST_FIXTURE(e_gemv, double_data) {

    // Define fixture parameters
    size_t M = 163;
    size_t K = 135;
    size_t P = 24;
    data->cparams.typesize = sizeof(double);

    // Define 'x' caterva container
    caterva_pparams pparams_x;
    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        pparams_x.shape[i] = 1;
        pparams_x.cshape[i] = 1;
    }
    pparams_x.shape[CATERVA_MAXDIM - 1] = K;  // FIXME: 1's at the beginning should be removed
    pparams_x.shape[CATERVA_MAXDIM - 2] = M;  // FIXME: 1's at the beginning should be removed
    pparams_x.cshape[CATERVA_MAXDIM - 1] = P;  // FIXME: 1's at the beginning should be removed
    pparams_x.cshape[CATERVA_MAXDIM - 2] = P;  // FIXME: 1's at the beginning should be removed
    pparams_x.ndims = 2;
    blosc2_frame fr_x = BLOSC_EMPTY_FRAME;
    caterva_array *cta_x = caterva_new_array(data->cparams, data->dparams, &fr_x, pparams_x);
    double *buf_x = (double *) malloc(sizeof(double) * M * K);
    dfill_buf(buf_x, M * K);
    caterva_from_buffer(cta_x, buf_x);

    // Create 'x' iarray container
    iarray_context_t *iactx_x;
    iarray_config_t cfg_x = {.max_num_threads = 1, .flags = IARRAY_EXPR_EVAL_CHUNK,
            .cparams = &data->cparams, .dparams = &data->dparams, .pparams = &pparams_x};
    iarray_ctx_new(&cfg_x, &iactx_x);
    iarray_container_t *c_x;
    iarray_from_ctarray(iactx_x, cta_x, IARRAY_DATA_TYPE_DOUBLE, &c_x);


    // Define 'y' caterva container
    caterva_pparams pparams_y;
    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        pparams_y.shape[i] = 1;
        pparams_y.cshape[i] = 1;
    }
    pparams_y.shape[CATERVA_MAXDIM - 1] = K;  // FIXME: 1's at the beginning should be removed
    pparams_y.cshape[CATERVA_MAXDIM - 1] = P;  // FIXME: 1's at the beginning should be removed
    pparams_y.ndims = 1;
    blosc2_frame fr_y = BLOSC_EMPTY_FRAME;
    caterva_array *cta_y = caterva_new_array(data->cparams, data->dparams, &fr_y, pparams_y);
    double *buf_y = (double *) malloc(sizeof(double) * K);
    dfill_buf(buf_y, K);
    caterva_from_buffer(cta_y, buf_y);

    // Create 'y' iarray container
    iarray_context_t *iactx_y;
    iarray_config_t cfg_y = {.max_num_threads = 1, .flags = IARRAY_EXPR_EVAL_CHUNK,
            .cparams = &data->cparams, .dparams = &data->dparams, .pparams = &pparams_y};
    iarray_ctx_new(&cfg_y, &iactx_y);
    iarray_container_t *c_y;
    iarray_from_ctarray(iactx_y, cta_y, IARRAY_DATA_TYPE_DOUBLE, &c_y);

    // Define 'out' caterva container
    caterva_pparams pparams_out;
    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        pparams_out.shape[i] = 1;
        pparams_out.cshape[i] = 1;
    }
    pparams_out.shape[CATERVA_MAXDIM - 1] = M;  // FIXME: 1's at the beginning should be removed
    pparams_out.cshape[CATERVA_MAXDIM - 1] = P;  // FIXME: 1's at the beginning should be removed
    pparams_out.ndims = 1;
    blosc2_frame fr_out = BLOSC_EMPTY_FRAME;
    caterva_array *cta_out = caterva_new_array(data->cparams, data->dparams, &fr_out, pparams_out);

    // Create 'out' iarray container
    iarray_context_t *iactx_out;
    iarray_config_t cfg_out = {.max_num_threads = 1, .flags = IARRAY_EXPR_EVAL_CHUNK,
            .cparams = &data->cparams, .dparams = &data->dparams, .pparams = &pparams_out};
    iarray_ctx_new(&cfg_out, &iactx_out);
    iarray_container_t *c_out;
    iarray_from_ctarray(iactx_out, cta_out, IARRAY_DATA_TYPE_DOUBLE, &c_out);

    // Define 'res' caterva container
    caterva_pparams pparams_res;
    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        pparams_res.shape[i] = 1;
        pparams_res.cshape[i] = 1;
    }
    pparams_res.shape[CATERVA_MAXDIM - 1] = M;  // FIXME: 1's at the beginning should be removed
    pparams_res.cshape[CATERVA_MAXDIM - 1] = P;  // FIXME: 1's at the beginning should be removed
    pparams_res.ndims = 1;
    blosc2_frame fr_res = BLOSC_EMPTY_FRAME;
    caterva_array *cta_res = caterva_new_array(data->cparams, data->dparams, &fr_res, pparams_res);


    // Obtain values of 'res' buffer
    double *buf_res = (double *) calloc(cta_res->size, (size_t)cta_res->sc->typesize);
    dmv_mul(M, K, buf_x, buf_y, buf_res);
    caterva_from_buffer(cta_res, buf_res);


    // Create 'res' iarray container
    iarray_context_t *iactx_res;
    iarray_config_t cfg_res = {.max_num_threads = 1, .flags = IARRAY_EXPR_EVAL_CHUNK,
            .cparams = &data->cparams, .dparams = &data->dparams, .pparams = &pparams_res};
    iarray_ctx_new(&cfg_res, &iactx_res);
    iarray_container_t *c_res;
    iarray_from_ctarray(iactx_res, cta_res, IARRAY_DATA_TYPE_DOUBLE, &c_res);

    INA_TEST_ASSERT_TRUE(test_gemv(c_x, c_y, c_out, c_res));

    // Free memory
    free(buf_x);
    free(buf_y);
    free(buf_res);

    caterva_free_array(cta_x);
    caterva_free_array(cta_y);
    caterva_free_array(cta_out);
    caterva_free_array(cta_res);

    iarray_ctx_free(&iactx_x);
    iarray_ctx_free(&iactx_y);
    iarray_ctx_free(&iactx_out);
    iarray_ctx_free(&iactx_res);
}

INA_TEST_FIXTURE(e_gemv, float_data) {

    // Define fixture parameters
    size_t M = 345;
    size_t K = 65;
    size_t P = 15;
    data->cparams.typesize = sizeof(float);

    // Define 'x' caterva container
    caterva_pparams pparams_x;
    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        pparams_x.shape[i] = 1;
        pparams_x.cshape[i] = 1;
    }
    pparams_x.shape[CATERVA_MAXDIM - 1] = K;  // FIXME: 1's at the beginning should be removed
    pparams_x.shape[CATERVA_MAXDIM - 2] = M;  // FIXME: 1's at the beginning should be removed
    pparams_x.cshape[CATERVA_MAXDIM - 1] = P;  // FIXME: 1's at the beginning should be removed
    pparams_x.cshape[CATERVA_MAXDIM - 2] = P;  // FIXME: 1's at the beginning should be removed
    pparams_x.ndims = 2;
    blosc2_frame fr_x = BLOSC_EMPTY_FRAME;
    caterva_array *cta_x = caterva_new_array(data->cparams, data->dparams, &fr_x, pparams_x);
    float *buf_x = (float *) malloc(sizeof(float) * M * K);
    ffill_buf(buf_x, M * K);
    caterva_from_buffer(cta_x, buf_x);

    // Create 'x' iarray container
    iarray_context_t *iactx_x;
    iarray_config_t cfg_x = {.max_num_threads = 1, .flags = IARRAY_EXPR_EVAL_CHUNK,
            .cparams = &data->cparams, .dparams = &data->dparams, .pparams = &pparams_x};
    iarray_ctx_new(&cfg_x, &iactx_x);
    iarray_container_t *c_x;
    iarray_from_ctarray(iactx_x, cta_x, IARRAY_DATA_TYPE_FLOAT, &c_x);


    // Define 'y' caterva container
    caterva_pparams pparams_y;
    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        pparams_y.shape[i] = 1;
        pparams_y.cshape[i] = 1;
    }
    pparams_y.shape[CATERVA_MAXDIM - 1] = K;  // FIXME: 1's at the beginning should be removed
    pparams_y.cshape[CATERVA_MAXDIM - 1] = P;  // FIXME: 1's at the beginning should be removed
    pparams_y.ndims = 1;
    blosc2_frame fr_y = BLOSC_EMPTY_FRAME;
    caterva_array *cta_y = caterva_new_array(data->cparams, data->dparams, &fr_y, pparams_y);
    float *buf_y = (float *) malloc(sizeof(float) * K);
    ffill_buf(buf_y, K);
    caterva_from_buffer(cta_y, buf_y);

    // Create 'y' iarray container
    iarray_context_t *iactx_y;
    iarray_config_t cfg_y = {.max_num_threads = 1, .flags = IARRAY_EXPR_EVAL_CHUNK,
            .cparams = &data->cparams, .dparams = &data->dparams, .pparams = &pparams_y};
    iarray_ctx_new(&cfg_y, &iactx_y);
    iarray_container_t *c_y;
    iarray_from_ctarray(iactx_y, cta_y, IARRAY_DATA_TYPE_FLOAT, &c_y);

    // Define 'out' caterva container
    caterva_pparams pparams_out;
    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        pparams_out.shape[i] = 1;
        pparams_out.cshape[i] = 1;
    }
    pparams_out.shape[CATERVA_MAXDIM - 1] = M;  // FIXME: 1's at the beginning should be removed
    pparams_out.cshape[CATERVA_MAXDIM - 1] = P;  // FIXME: 1's at the beginning should be removed
    pparams_out.ndims = 1;
    blosc2_frame fr_out = BLOSC_EMPTY_FRAME;
    caterva_array *cta_out = caterva_new_array(data->cparams, data->dparams, &fr_out, pparams_out);

    // Create 'out' iarray container
    iarray_context_t *iactx_out;
    iarray_config_t cfg_out = {.max_num_threads = 1, .flags = IARRAY_EXPR_EVAL_CHUNK,
            .cparams = &data->cparams, .dparams = &data->dparams, .pparams = &pparams_out};
    iarray_ctx_new(&cfg_out, &iactx_out);
    iarray_container_t *c_out;
    iarray_from_ctarray(iactx_out, cta_out, IARRAY_DATA_TYPE_FLOAT, &c_out);

    // Define 'res' caterva container
    caterva_pparams pparams_res;
    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        pparams_res.shape[i] = 1;
        pparams_res.cshape[i] = 1;
    }
    pparams_res.shape[CATERVA_MAXDIM - 1] = M;  // FIXME: 1's at the beginning should be removed
    pparams_res.cshape[CATERVA_MAXDIM - 1] = P;  // FIXME: 1's at the beginning should be removed
    pparams_res.ndims = 1;
    blosc2_frame fr_res = BLOSC_EMPTY_FRAME;
    caterva_array *cta_res = caterva_new_array(data->cparams, data->dparams, &fr_res, pparams_res);


    // Obtain values of 'res' buffer
    float *buf_res = (float *) calloc(cta_res->size, (size_t) cta_res->sc->typesize);
    fmv_mul(M, K, buf_x, buf_y, buf_res);
    caterva_from_buffer(cta_res, buf_res);


    // Create 'res' iarray container
    iarray_context_t *iactx_res;
    iarray_config_t cfg_res = {.max_num_threads = 1, .flags = IARRAY_EXPR_EVAL_CHUNK,
            .cparams = &data->cparams, .dparams = &data->dparams, .pparams = &pparams_res};
    iarray_ctx_new(&cfg_res, &iactx_res);
    iarray_container_t *c_res;
    iarray_from_ctarray(iactx_res, cta_res, IARRAY_DATA_TYPE_FLOAT, &c_res);

    INA_TEST_ASSERT_TRUE(test_gemv(c_x, c_y, c_out, c_res));

    // Free memory
    free(buf_x);
    free(buf_y);
    free(buf_res);

    caterva_free_array(cta_x);
    caterva_free_array(cta_y);
    caterva_free_array(cta_out);
    caterva_free_array(cta_res);

    iarray_ctx_free(&iactx_x);
    iarray_ctx_free(&iactx_y);
    iarray_ctx_free(&iactx_out);
    iarray_ctx_free(&iactx_res);
}
