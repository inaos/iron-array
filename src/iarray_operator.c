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


static ina_rc_t _iarray_gemm(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *c,
                             uint64_t *bshape_a, uint64_t *bshape_b) {

    caterva_dims_t shape = caterva_new_dims(c->dtshape->shape, c->dtshape->ndim);
    caterva_update_shape(c->catarr, shape);

    uint64_t B0 = bshape_a[0];
    uint64_t B1 = bshape_a[1];
    uint64_t B2 = bshape_b[1];

    // the extended shape is recalculated from the block shape
    uint64_t eshape_a[IARRAY_DIMENSION_MAX];
    uint64_t eshape_b[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        if (a->dtshape->shape[i] % bshape_a[i] == 0) {
            eshape_a[i] = a->dtshape->shape[i];
        } else {
            eshape_a[i] = (a->dtshape->shape[i] / bshape_a[i] + 1) * bshape_a[i];
        }
        if (b->dtshape->shape[i] % bshape_b[i] == 0) {
            eshape_b[i] = b->dtshape->shape[i];
        } else {
            eshape_b[i] = (b->dtshape->shape[i] / bshape_b[i] + 1) * bshape_b[i];
        }
    }

    // block sizes are claculated
    uint64_t a_size = (uint64_t) B0 * B1 * a->catarr->sc->typesize;
    uint64_t b_size = (uint64_t) B1 * B2 * b->catarr->sc->typesize;
    uint64_t c_size = (uint64_t) B0 * B2 * c->catarr->sc->typesize;
    int dtype = a->dtshape->dtype;

    uint8_t *a_block = malloc(a_size);
    uint8_t *b_block = malloc(b_size);
    uint8_t *c_block = malloc(c_size);

    // Start a iterator that returns the index matrix blocks
    iarray_iter_matmul_t *iter;
    _iarray_iter_matmul_new(ctx, a, b, bshape_a, bshape_b, &iter);

    memset(c_block, 0, c_size);

    for (_iarray_iter_matmul_init(iter); !_iarray_iter_matmul_finished(iter); _iarray_iter_matmul_next(iter)) {
        uint64_t start_a[IARRAY_DIMENSION_MAX];
        uint64_t stop_a[IARRAY_DIMENSION_MAX];
        uint64_t start_b[IARRAY_DIMENSION_MAX];
        uint64_t stop_b[IARRAY_DIMENSION_MAX];

        uint64_t inc_a = 1;
        uint64_t inc_b = 1;

        // the block coords are calculated from the index
        uint64_t part_ind_a[IARRAY_DIMENSION_MAX];
        uint64_t part_ind_b[IARRAY_DIMENSION_MAX];
        for (int i = a->dtshape->ndim - 1; i >= 0; --i) {
            part_ind_a[i] = iter->npart1 % (inc_a * (eshape_a[i] / bshape_a[i])) / inc_a;
            inc_a *= (eshape_a[i] / bshape_a[i]);
            part_ind_b[i] = iter->npart2 % (inc_b * (eshape_b[i] / bshape_b[i])) / inc_b;
            inc_b *= (eshape_b[i] / bshape_b[i]);
        }

        // a start and a stop are calculated from the block coords
        for (int i = 0; i < a->dtshape->ndim; ++i) {
            start_a[i] = part_ind_a[i] * bshape_a[i];
            start_b[i] = part_ind_b[i] * bshape_b[i];
            if (start_a[i] + bshape_a[i] > a->dtshape->shape[i]) {
                stop_a[i] = a->dtshape->shape[i];
            } else {
                stop_a[i] = start_a[i] + bshape_a[i];
            }
            if (start_b[i] + bshape_b[i] > b->dtshape->shape[i]) {
                stop_b[i] = b->dtshape->shape[i];
            } else {
                stop_b[i] = start_b[i] + bshape_b[i];
            }
        }

        // Obtain desired blocks from iarray containers
        memset(a_block, 0, a_size);
        memset(b_block, 0, b_size);
        _iarray_slice_buffer(ctx, a, start_a, stop_a, bshape_a, a_block, a_size);
        _iarray_slice_buffer(ctx, b, start_b, stop_b, bshape_b, b_block, b_size);

        // Make blocks multiplication
        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B0, B2, B1, 1.0, (double *)a_block, B1, (double *)b_block, B2, 1.0, (double *)c_block, B2);
        }
        else if (dtype == IARRAY_DATA_TYPE_FLOAT) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B0, B2, B1, 1.0, (float *)a_block, B1, (float *)b_block, B2, 1.0, (float *)c_block, B2);
        }

        // Append it to a new iarray contianer
        if((iter->cont + 1) % (eshape_a[1] / B1) == 0) {
            blosc2_schunk_append_buffer(c->catarr->sc, &c_block[0], c_size);
            memset(c_block, 0, c_size);
        }
    }

    _iarray_iter_matmul_free(iter);
    free(a_block);
    free(b_block);
    free(c_block);

    return INA_SUCCESS;
}

static ina_rc_t _iarray_gemv(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *c,
                             uint64_t *bshape_a, uint64_t *bshape_b) {

    caterva_dims_t shape = caterva_new_dims(c->dtshape->shape, c->dtshape->ndim);
    caterva_update_shape(c->catarr, shape);

    uint64_t B0 = bshape_a[0];
    uint64_t B1 = bshape_a[1];

    uint64_t eshape_a[2];
    uint64_t eshape_b[1];

    // the extended shape is recalculated from the block shape
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        if (a->dtshape->shape[i] % bshape_a[i] == 0) {
            eshape_a[i] = a->dtshape->shape[i];
        } else {
            eshape_a[i] = (a->dtshape->shape[i] / bshape_a[i] + 1) * bshape_a[i];
        }
    }
    if (b->dtshape->shape[0] % bshape_b[0] == 0) {
        eshape_b[0] = b->dtshape->shape[0];
    } else {
        eshape_b[0] = (b->dtshape->shape[0] / bshape_b[0] + 1) * bshape_b[0];
    }

    // block sizes are claculated
    uint64_t a_size = (uint64_t) B0 * B1 * a->catarr->sc->typesize;
    uint64_t b_size = (uint64_t) B1 * a->catarr->sc->typesize;
    uint64_t c_size = (uint64_t) B0 * a->catarr->sc->typesize;

    int dtype = a->dtshape->dtype;

    uint8_t *a_block = malloc(a_size);
    uint8_t *b_block = malloc(b_size);
    uint8_t *c_block = malloc(c_size);

    // Start a iterator that returns the index matrix blocks
    iarray_iter_matmul_t *iter;
    _iarray_iter_matmul_new(ctx, a, b, bshape_a, bshape_b, &iter);

    memset(c_block, 0, c_size);
    for (_iarray_iter_matmul_init(iter); !_iarray_iter_matmul_finished(iter); _iarray_iter_matmul_next(iter)) {

        uint64_t start_a[IARRAY_DIMENSION_MAX];
        uint64_t stop_a[IARRAY_DIMENSION_MAX];
        uint64_t start_b[IARRAY_DIMENSION_MAX];
        uint64_t stop_b[IARRAY_DIMENSION_MAX];

        uint64_t inc_a = 1;

        uint64_t part_ind_a[IARRAY_DIMENSION_MAX];
        uint64_t part_ind_b[IARRAY_DIMENSION_MAX];

        // the block coords are calculated from the index
        for (int i = a->dtshape->ndim - 1; i >= 0; --i) {
            part_ind_a[i] = iter->npart1 % (inc_a * (eshape_a[i] / bshape_a[i])) / inc_a;
            inc_a *= (eshape_a[i] / bshape_a[i]);
        }
        part_ind_b[0] = iter->npart2 % ( (eshape_b[0] / bshape_b[0]));


        // a start and a stop are calculated from the block coords
        for (int i = 0; i < a->dtshape->ndim; ++i) {
            start_a[i] = part_ind_a[i] * bshape_a[i];
            if (start_a[i] + bshape_a[i] > a->dtshape->shape[i]) {
                stop_a[i] = a->dtshape->shape[i];
            } else {
                stop_a[i] = start_a[i] + bshape_a[i];
            }

        }
        start_b[0] = part_ind_b[0] * bshape_b[0];
        if (start_b[0] + bshape_b[0] > b->dtshape->shape[0]) {
            stop_b[0] = b->dtshape->shape[0];
        } else {
            stop_b[0] = start_b[0] + bshape_b[0];
        }

        // Obtain desired blocks from iarray containers
        memset(a_block, 0, a_size);
        memset(b_block, 0, b_size);
        _iarray_slice_buffer(ctx, a, start_a, stop_a, bshape_a, a_block, a_size);
        _iarray_slice_buffer(ctx, b, start_b, stop_b, bshape_b, b_block, b_size);

        // Make blocks multiplication
        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            cblas_dgemv(CblasRowMajor, CblasNoTrans, B0, B1, 1.0, (double *) a_block, B1, (double *) b_block, 1, 1.0, (double *) c_block, 1);
        }
        else if (dtype == IARRAY_DATA_TYPE_FLOAT) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, B0, B1, 1.0, (float *) a_block, B1, (float *) b_block, 1, 1.0, (float *) c_block, 1);
        }

        // Append it to a new iarray contianer
        if((iter->cont + 1) % (eshape_a[1] / B1) == 0) {
            blosc2_schunk_append_buffer(c->catarr->sc, &c_block[0], c_size);
            memset(c_block, 0, c_size);
        }
    }

    _iarray_iter_matmul_free(iter);
    free(a_block);
    free(b_block);
    free(c_block);

    return INA_SUCCESS;
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

    caterva_dims_t shape = caterva_new_dims(result->dtshape->shape, result->dtshape->ndim);
    caterva_update_shape(result->catarr, shape);

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

    caterva_dims_t shape = caterva_new_dims(result->dtshape->shape, result->dtshape->ndim);
    caterva_update_shape(result->catarr, shape);

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
                                       uint64_t *bshape_a,
                                       uint64_t *bshape_b,
                                       iarray_operator_hint_t hint)
{
    INA_ASSERT_NOT_NULL(ctx);
    INA_ASSERT_NOT_NULL(a);
    INA_ASSERT_NOT_NULL(b);
    INA_ASSERT_NOT_NULL(c);
    INA_ASSERT_NOT_NULL(bshape_a);
    INA_ASSERT_NOT_NULL(bshape_b);

    if (bshape_a[0] != c->dtshape->pshape[0]){
        return INA_ERR_INVALID_ARGUMENT;
    }

    if (a->dtshape->ndim != 2) {
        return INA_ERR_INVALID_ARGUMENT;
    }
    if (b->dtshape->ndim == 1) {
        return _iarray_gemv(ctx, a, b, c, bshape_a, bshape_b);
    }
    else if (b->dtshape->ndim == 2) {
        if (bshape_b[1] != c->dtshape->pshape[1]) {
            return INA_ERR_INVALID_ARGUMENT;
        }
        return _iarray_gemm(ctx, a, b, c, bshape_a, bshape_b);
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
