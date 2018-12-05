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

static ina_rc_t _iarray_gemm(iarray_container_t *a, iarray_container_t *b, iarray_container_t *c) {

    caterva_update_shape(c->catarr, *c->shape);

    const int32_t P = (int32_t) a->catarr->pshape[0];
    uint64_t M = a->catarr->eshape[0];
    uint64_t K = a->catarr->eshape[1];
    uint64_t N = b->catarr->eshape[1];

    uint64_t p_size = (uint64_t) P * P * a->catarr->sc->typesize;
    int dtype = a->dtshape->dtype;

    uint8_t *a_block = malloc(p_size);
    uint8_t *b_block = malloc(p_size);
    uint8_t *c_block = malloc(p_size);

    for (size_t m = 0; m < M / P; m++)
    {
        for (size_t n = 0; n < N / P; n++)
        {
            memset(c_block, 0, p_size);
            for (size_t k = 0; k < K / P; k++)
            {
                size_t a_i = (m * K / P + k);
                size_t b_i = (k * N / P + n);

                int a_tam = blosc2_schunk_decompress_chunk(a->catarr->sc, (int)a_i, a_block, p_size);
                int b_tam = blosc2_schunk_decompress_chunk(b->catarr->sc, (int)b_i, b_block, p_size);

                if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, P, P, P, 1.0, (double *)a_block, P, (double *)b_block, P, 1.0, (double *)c_block, P);
                }
                else if (dtype == IARRAY_DATA_TYPE_FLOAT) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, P, P, P, 1.0, (float *)a_block, P, (float *)b_block, P, 1.0, (float *)c_block, P);
                }
            }
            blosc2_schunk_append_buffer(c->catarr->sc, &c_block[0], p_size);
        }
    }
    free(a_block);
    free(b_block);
    free(c_block);

    return INA_SUCCESS;
}

static ina_rc_t _iarray_gemv(iarray_container_t *a, iarray_container_t *b, iarray_container_t *c) {

    caterva_update_shape(c->catarr, *c->shape);

    int32_t P = (int32_t) a->catarr->pshape[0];

    uint64_t M = a->catarr->eshape[0];
    uint64_t K = a->catarr->eshape[1];

    uint64_t p_size = (uint64_t) P * P * a->catarr->sc->typesize;
    uint64_t p_vsize = (uint64_t) P * a->catarr->sc->typesize;

    int dtype = a->dtshape->dtype;

    uint8_t *a_block = malloc(p_size);
    uint8_t *b_block = malloc(p_vsize);
    uint8_t *c_block = malloc(p_vsize);

    size_t a_i, b_i;

    for (size_t m = 0; m < M / P; m++)
    {
        memset(c_block, 0, p_vsize);
        for (size_t k = 0; k < K / P; k++)
        {
            a_i = (m * K / P + k);
            b_i = (k);

            int a_tam = blosc2_schunk_decompress_chunk(a->catarr->sc, (int)a_i, a_block, p_size);
            int b_tam = blosc2_schunk_decompress_chunk(b->catarr->sc, (int)b_i, b_block, p_vsize);

            if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
                cblas_dgemv(CblasRowMajor, CblasNoTrans, P, P, 1.0, (double *) a_block, P, (double *) b_block, 1, 1.0, (double *) c_block, 1);
            }
            else if (dtype == IARRAY_DATA_TYPE_FLOAT) {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, P, P, 1.0, (float *) a_block, P, (float *) b_block, 1, 1.0, (float *) c_block, 1);
            }
        }
        blosc2_schunk_append_buffer(c->catarr->sc, &c_block[0], p_vsize);
    }
    free(a_block);
    free(b_block);
    free(c_block);

    return INA_SUCCESS;;
}

INA_API(ina_rc_t) iarray_operation_transpose(iarray_container_t *a)
{
    if (a->transposed == 0) {
        a->transposed = 1;
    }
    else {
        a->transposed = 0;
    }
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_linalg_matmul(iarray_container_t *a, iarray_container_t *b, iarray_container_t *c, int flag)
{
    /* FIXME: handle special shapes */
    if (a->dtshape->ndim != 2) {
        return INA_ERR_INVALID_ARGUMENT;
    }
    if (b->dtshape->ndim == 1) {
        return _iarray_gemv(a, b, c);
    }
    else if (b->dtshape->ndim == 2) {
        return _iarray_gemm(a, b, c);
    }
    else {
        return INA_ERR_INVALID_ARGUMENT;
    }
}
