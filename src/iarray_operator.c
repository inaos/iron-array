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



static ina_rc_t _iarray_gemm(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *c) {

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

    iarray_itr_matmul_t *I;
    _iarray_itr_matmul_new(ctx, a, b, &I);

    memset(c_block, 0, p_size);
    for (_iarray_itr_matmul_init(I); !_iarray_itr_matmul_finished(I); _iarray_itr_matmul_next(I)) {

        int a_tam = blosc2_schunk_decompress_chunk(a->catarr->sc, (int)I->nchunk1, a_block, p_size);
        int b_tam = blosc2_schunk_decompress_chunk(b->catarr->sc, (int)I->nchunk2, b_block, p_size);

        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, P, P, P, 1.0, (double *)a_block, P, (double *)b_block, P, 1.0, (double *)c_block, P);
        }
        else if (dtype == IARRAY_DATA_TYPE_FLOAT) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, P, P, P, 1.0, (float *)a_block, P, (float *)b_block, P, 1.0, (float *)c_block, P);
        }
        if((I->cont + 1) % (K / P) == 0) {
            blosc2_schunk_append_buffer(c->catarr->sc, &c_block[0], p_size);
            memset(c_block, 0, p_size);
        }
    }

    free(a_block);
    free(b_block);
    free(c_block);

    return INA_SUCCESS;
}

static ina_rc_t _iarray_gemv(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *c) {

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

    iarray_itr_matmul_t *I;
    _iarray_itr_matmul_new(ctx, a, b, &I);

    memset(c_block, 0, p_vsize);
    for (_iarray_itr_matmul_init(I); !_iarray_itr_matmul_finished(I); _iarray_itr_matmul_next(I)) {

        int a_tam = blosc2_schunk_decompress_chunk(a->catarr->sc, (int)I->nchunk1, a_block, p_size);
        int b_tam = blosc2_schunk_decompress_chunk(b->catarr->sc, (int)I->nchunk2, b_block, p_vsize);

        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            cblas_dgemv(CblasRowMajor, CblasNoTrans, P, P, 1.0, (double *) a_block, P, (double *) b_block, 1, 1.0, (double *) c_block, 1);
        }
        else if (dtype == IARRAY_DATA_TYPE_FLOAT) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, P, P, 1.0, (float *) a_block, P, (float *) b_block, 1, 1.0, (float *) c_block, 1);
        }

        if((I->cont + 1) % (K / P) == 0) {
            blosc2_schunk_append_buffer(c->catarr->sc, &c_block[0], p_vsize);
            memset(c_block, 0, p_vsize);
        }
    }
    _iarray_itr_matmul_free(ctx, I);
    free(a_block);
    free(b_block);
    free(c_block);

    return INA_SUCCESS;;
}

static ina_rc_t _iarray_operator_elwise_a(
    iarray_context_t *ctx,
    iarray_container_t *a,
    iarray_container_t *result,
    _iarray_vml_fun_d_a mkl_fun_d,
    _iarray_vml_fun_s_a mkl_fun_s)
{
    INA_ASSERT_NOT_NULL(ctx);
    INA_ASSERT_NOT_NULL(a);
    INA_ASSERT_NOT_NULL(result);
    INA_ASSERT_NOT_NULL(mkl_fun_d);
    INA_ASSERT_NOT_NULL(mkl_fun_s);

    caterva_update_shape(result->catarr, *result->shape);

    size_t psize = (size_t)a->catarr->sc->typesize;
    for (int i = 0; i < a->catarr->ndim; ++i) {
        psize *= a->catarr->pshape[i];
    }

    int8_t *a_chunk = (int8_t*)ina_mempool_dalloc(ctx->mp_op, psize);
    int8_t *c_chunk = (int8_t*)ina_mempool_dalloc(ctx->mp_op, psize);

    for (int i = 0; i < a->catarr->sc->nchunks; ++i) {
        INA_FAIL_IF(blosc2_schunk_decompress_chunk(a->catarr->sc, i, a_chunk, psize) < 0);
        switch (a->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            mkl_fun_d((const int)(psize / sizeof(double)), (const double*)a_chunk, (double*)c_chunk);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            mkl_fun_s((const int)psize / sizeof(float), (const float*)a_chunk, (float*)c_chunk);
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

static ina_rc_t _iarray_operator_elwise_ab(
        iarray_context_t *ctx,
        iarray_container_t *a,
        iarray_container_t *b,
        iarray_container_t *result,
        _iarray_vml_fun_d_ab mkl_fun_d,
        _iarray_vml_fun_s_ab mkl_fun_s)
{
    INA_ASSERT_NOT_NULL(ctx);
    INA_ASSERT_NOT_NULL(a);
    INA_ASSERT_NOT_NULL(b);
    INA_ASSERT_NOT_NULL(result);
    INA_ASSERT_NOT_NULL(mkl_fun_d);
    INA_ASSERT_NOT_NULL(mkl_fun_s);

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
                mkl_fun_s((const int)psize/sizeof(float), (const float*)a_chunk, (const float*)b_chunk, (float*)c_chunk);
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

INA_API(ina_rc_t) iarray_linalg_matmul(iarray_context_t *ctx,
                                       iarray_container_t *a,
                                       iarray_container_t *b,
                                       iarray_container_t *c,
                                       iarray_operator_hint_t hint)
{
    /* FIXME: handle special shapes */
    if (a->dtshape->ndim != 2) {
        return INA_ERR_INVALID_ARGUMENT;
    }
    if (b->dtshape->ndim == 1) {
        return _iarray_gemv(ctx, a, b, c);
    }
    else if (b->dtshape->ndim == 2) {
        return _iarray_gemm(ctx, a, b, c);
    }
    else {
        return INA_ERR_INVALID_ARGUMENT;
    }
}

INA_API(ina_rc_t) iarray_operator_and(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_operator_or(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_operator_xor(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_operator_nand(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_operator_not(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_operator_add(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return _iarray_operator_elwise_ab(ctx, a, b, result, vdAdd, vsAdd);
}

INA_API(ina_rc_t) iarray_operator_sub(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return _iarray_operator_elwise_ab(ctx, a, b, result, vdSub, vsSub);
}

INA_API(ina_rc_t) iarray_operator_mul(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return _iarray_operator_elwise_ab(ctx, a, b, result, vdMul, vsMul);
}

INA_API(ina_rc_t) iarray_operator_div(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return _iarray_operator_elwise_ab(ctx, a, b, result, vdDiv, vsDiv);
}

INA_API(ina_rc_t) iarray_operator_abs(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdAbs, vsAbs);
}

INA_API(ina_rc_t) iarray_operator_acos(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdAcos, vsAcos);
}

INA_API(ina_rc_t) iarray_operator_asin(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdAsin, vsAsin);
}

INA_API(ina_rc_t) iarray_operator_atanc(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_operator_atan2(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_operator_ceil(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdCeil, vsCeil);
}

INA_API(ina_rc_t) iarray_operator_cos(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdCos, vsCos);
}

INA_API(ina_rc_t) iarray_operator_cosh(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdCosh, vsCosh);
}

INA_API(ina_rc_t) iarray_operator_exp(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdExp, vsExp);
}

INA_API(ina_rc_t) iarray_operator_floor(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdFloor, vsFloor);
}

INA_API(ina_rc_t) iarray_operator_log(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_operator_log10(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdLog10, vsLog10);
}

INA_API(ina_rc_t) iarray_operator_pow(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    return _iarray_operator_elwise_ab(ctx, a, b, result, vdPow, vsPow);
}

INA_API(ina_rc_t) iarray_operator_sin(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdSin, vsSin);
}

INA_API(ina_rc_t) iarray_operator_sinh(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdSinh, vsSinh);
}

INA_API(ina_rc_t) iarray_operator_sqrt(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdSqrt, vsSqrt);
}

INA_API(ina_rc_t) iarray_operator_tan(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdTan, vsTan);
}

INA_API(ina_rc_t) iarray_operator_tanh(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdTanh, vsTanh);
}

INA_API(ina_rc_t) iarray_operator_erf(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdErf, vsErf);
}

INA_API(ina_rc_t) iarray_operator_erfc(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdErfc, vsErfc);
}

INA_API(ina_rc_t) iarray_operator_cdfnorm(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdCdfNorm, vsCdfNorm);
}

INA_API(ina_rc_t) iarray_operator_erfinv(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdErfInv, vsErfInv);
}

INA_API(ina_rc_t) iarray_operator_erfcinv(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdErfcInv, vsErfcInv);
}

INA_API(ina_rc_t) iarray_operator_cdfnorminv(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdCdfNormInv, vsCdfNormInv);
}

INA_API(ina_rc_t) iarray_operator_lgamma(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdLGamma, vsLGamma);
}

INA_API(ina_rc_t) iarray_operator_tgamma(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdTGamma, vsTGamma);
}

INA_API(ina_rc_t) iarray_operator_expint1(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    return _iarray_operator_elwise_a(ctx, a, result, vdExpInt1, vsExpInt1);
}
