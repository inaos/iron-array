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

typedef void (*_iarray_mkl_fun_d)(const MKL_INT n, const double a[], const double b[], double r[]);
typedef void (*_iarray_mkl_fun_f)(const MKL_INT n, const float a[], const float b[], float r[]);

static ina_rc_t _iarray_gemm(iarray_container_t *a, iarray_container_t *b, iarray_container_t *c)
{
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

static ina_rc_t _iarray_gemv(iarray_container_t *a, iarray_container_t *b, iarray_container_t *c)
{
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

static ina_rc_t _iarray_operator_elem_wise(
        iarray_context_t *ctx,
        iarray_container_t *a,
        iarray_container_t *b,
        iarray_container_t *result,
        _iarray_mkl_fun_d mkl_fun_d,
        _iarray_mkl_fun_f mkl_fun_f)
{
    if (!INA_SUCCEED(iarray_container_dtshape_equal(a->dtshape, b->dtshape))) {
        return INA_ERR_INVALID_ARGUMENT;
    }

    caterva_update_shape(result->catarr, *result->shape);

    size_t psize = (size_t)a->catarr->sc->typesize;
    for (int i = 0; i < a->catarr->ndim; ++i) {
        if (a->catarr->pshape[i] != b->catarr->pshape[i]) {
            return INA_ERR_ILLEGAL;
        }
        psize *= a->catarr->pshape[i];
    }

    int8_t *a_chunk = (int8_t*)ina_mempool_dalloc(ctx->mp_op, psize);
    int8_t *b_chunk = (int8_t*)ina_mempool_dalloc(ctx->mp_op, psize);
    int8_t *c_chunk = (int8_t*)ina_mempool_dalloc(ctx->mp_op, psize);

    for (int i = 0; i < a->catarr->sc->nchunks; ++i) {
        INA_FAIL_IF(blosc2_schunk_decompress_chunk(a->catarr->sc, i, a_chunk, psize) < 0);
        INA_FAIL_IF(blosc2_schunk_decompress_chunk(b->catarr->sc, i, b_chunk, psize) < 0);
        switch (a->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_fun_d((const int)(psize/sizeof(double)), (const double*)a_chunk, (const double*)b_chunk, (double*)c_chunk);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_fun_f((const int)psize/sizeof(float), (const float*)a_chunk, (const float*)b_chunk, (float*)c_chunk);
                break;
        }
        blosc2_schunk_append_buffer(result->catarr->sc, c_chunk, psize);
    }

    ina_mempool_reset(ctx->mp_op);

    return INA_SUCCESS;

fail:
    ina_mempool_reset(ctx->mp_op);
    /* FIXME: error handling */
    return INA_ERR_ILLEGAL;
}

INA_API(ina_rc_t) iarray_operator_transpose(iarray_context_t *ctx, iarray_container_t *a)
{
    if (a->transposed == 0) {
        a->transposed = 1;
    }
    else {
        a->transposed = 0;
    }
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_linalg_matmul(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *c, int flag)
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

INA_API(ina_rc_t) iarray_operator_add(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return _iarray_operator_elem_wise(ctx, a, b, result, vdAdd, vsAdd);
}
