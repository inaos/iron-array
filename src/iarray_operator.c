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



static ina_rc_t _iarray_operator_elwise_a(
    iarray_context_t *ctx,
    iarray_container_t *a,
    iarray_container_t *result,
    _iarray_vml_fun_d_a mkl_fun_d,
    _iarray_vml_fun_s_a mkl_fun_s)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(result);
    INA_VERIFY_NOT_NULL(mkl_fun_d);
    INA_VERIFY_NOT_NULL(mkl_fun_s);


    size_t chunksize = (size_t)a->catarr->sc->typesize;
    for (int i = 0; i < a->catarr->ndim; ++i) {
        chunksize *= a->catarr->chunkshape[i];
    }

    iarray_iter_read_block_t *iter_read;
    iarray_iter_read_block_value_t val_read;
    IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_new(ctx, &iter_read, a, result->storage->chunkshape, &val_read, false));

    iarray_iter_write_block_t *iter_write;
    iarray_iter_write_block_value_t val_write;
    IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_new(ctx, &iter_write, result, result->storage->chunkshape, &val_write, false));


    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter_write)) && INA_SUCCEED(iarray_iter_read_block_has_next(iter_read))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_next(iter_write, NULL, 0));
        IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_next(iter_read, NULL, 0));
        switch (a->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_fun_d((const int)(iter_read->cur_block_size), (const double *) *iter_read->block_pointer, (double *) *iter_write->block_pointer);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_fun_s((const int)(iter_read->cur_block_size), (const float *) *iter_read->block_pointer, (float *) *iter_write->block_pointer);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                return (INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }
    }
    iarray_iter_read_block_free(&iter_read);
    iarray_iter_write_block_free(&iter_write);

    IARRAY_ITER_FINISH();

    return INA_SUCCESS;
}


static ina_rc_t _iarray_operator_elwise_ab(
        iarray_context_t *ctx,
        iarray_container_t *a,
        iarray_container_t *b,
        iarray_container_t *result,
        _iarray_vml_fun_d_ab mkl_fun_d,
        _iarray_vml_fun_s_ab mkl_fun_s)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(b);
    INA_VERIFY_NOT_NULL(result);
    INA_VERIFY_NOT_NULL(mkl_fun_d);
    INA_VERIFY_NOT_NULL(mkl_fun_s);

    IARRAY_RETURN_IF_FAILED(iarray_container_dtshape_equal(a->dtshape, b->dtshape));

    size_t chunksize = (size_t)a->catarr->sc->typesize;
    for (int i = 0; i < a->catarr->ndim; ++i) {
        if (a->catarr->chunkshape[i] != b->catarr->chunkshape[i]) {
            IARRAY_TRACE1(iarray.error, "The chunkshapes must be equals");
            return (INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE));
        }
        chunksize *= a->catarr->chunkshape[i];
    }

    iarray_iter_read_block_t *iter_read;
    iarray_iter_read_block_value_t val_read;
    IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_new(ctx, &iter_read, a, result->storage->chunkshape, &val_read, false));

    iarray_iter_read_block_t *iter_read2;
    iarray_iter_read_block_value_t val_read2;
    IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_new(ctx, &iter_read2, b, result->storage->chunkshape, &val_read2, false));

    iarray_iter_write_block_t *iter_write;
    iarray_iter_write_block_value_t val_write;
    IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_new(ctx, &iter_write, result, result->storage->chunkshape, &val_write, false));

    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter_write)) &&
           INA_SUCCEED(iarray_iter_read_block_has_next(iter_read)) &&
            INA_SUCCEED(iarray_iter_read_block_has_next(iter_read2))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_next(iter_write, NULL, 0));
        IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_next(iter_read, NULL, 0));
        IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_next(iter_read2, NULL, 0));
        switch (a->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_fun_d((const int)(iter_read->cur_block_size), (const double *) *iter_read->block_pointer,
                          (double *) *iter_read2->block_pointer, (double *) *iter_write->block_pointer);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_fun_s((const int)(iter_read->cur_block_size), (const float *) *iter_read->block_pointer,
                          (float *) *iter_read2->block_pointer, (float *) *iter_write->block_pointer);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                return (INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }
    }
    iarray_iter_read_block_free(&iter_read);
    iarray_iter_read_block_free(&iter_read2);
    iarray_iter_write_block_free(&iter_write);

    IARRAY_ITER_FINISH();

    return INA_SUCCESS;
}

/**
 * This function performs a matrix multiplication between iarray containers `a` and `b`and stores it
 * in `c` iarray container (a * b = c).
 *
 * The parameter `ctx` is an iarray context that allow users define the compression ratio, the
 * threads number, ...
 *
 * The `a` iarray container must be a dataset of 2 dimensions. If not, an error will be returned.
 *
 * In the same way, `b` container must be a dataset of 1 or 2 dimensions. If it have 1 dimension a
 * matrix-vector multiplication is performed. If it have 2 dimensions a matrix-matrix multiplication
 * is done.
 *
 * The `c` container must be an iarray container whose dimensions are equal to the `b` container.
 *
 * `blockshape_a` indicates indicates the block size with which the container `a` will be iterated when
 *  performing block multiplication. The chunkshape[0] of `c` must be equal to blockshape_a[0].
 *
 * `blockshape_b` indicates indicates the block size with which the container `b` will be iterated when
 *  performing block multiplication. The chunkshape[1] of `c` must be equal to blockshape_a[1].
 *
 *  In addition, in order to perform the multiplication correctly blockshape_a[1] = blockshape_b[0].
 *
 *  It is also supported the multiplication between containers with different structures
 *
 *  This function returns an error code ina_rc_t.
 */

INA_API(ina_rc_t) iarray_linalg_matmul(iarray_context_t *ctx,
                                       iarray_container_t *a,
                                       iarray_container_t *b,
                                       iarray_container_t *c,
                                       int64_t *blockshape_a,
                                       int64_t *blockshape_b,
                                       iarray_operator_hint_t hint)
{
    INA_UNUSED(hint);
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(b);
    INA_VERIFY_NOT_NULL(c);

    if (c->catarr->filled) {
        IARRAY_TRACE1(iarray.error, "The output container must be empty");
        return INA_ERROR(IARRAY_ERR_FULL_CONTAINER);
    }

    if (a->dtshape->dtype != b->dtshape->dtype) {
        IARRAY_TRACE1(iarray.error, "The data types must be equal");
        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }

    if (a->dtshape->ndim != 2) {
        IARRAY_TRACE1(iarray.error, "The dimensions of the first container must be 2");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }

    if (a->dtshape->shape[1] != b->dtshape->shape[0]) {
        IARRAY_TRACE1(iarray.error, "The second dimension of the first container shape must be"
                                    "equal to the first dimension of the second container shape");
        return INA_ERROR(IARRAY_ERR_INVALID_SHAPE);
    }

    if (blockshape_a == NULL) {
        blockshape_a = a->dtshape->shape;
    }
    if (blockshape_b == NULL) {
        blockshape_b = b->dtshape->shape;
    }

    if (blockshape_a[1] != blockshape_b[0]) {
        IARRAY_TRACE1(iarray.error, "The second dimension of the first bshape must be"
                                    "equal to the first dimension of the second bshape");
        return INA_ERROR(IARRAY_ERR_INVALID_BLOCKSHAPE);
    }

    if (blockshape_a[0] != c->storage->chunkshape[0]){
        IARRAY_TRACE1(iarray.error, "The first dimension of the first bshape must be"
                                    "equal to the first dimension of the output container chunkshape");
        return INA_ERROR(IARRAY_ERR_INVALID_BLOCKSHAPE);
    }

    if (b->dtshape->ndim == 1) {
        return _iarray_gemv(ctx, a, b, c, blockshape_a, blockshape_b);
    }
    else if (b->dtshape->ndim == 2) {
        if (blockshape_b[1] != c->storage->chunkshape[1]) {
            IARRAY_TRACE1(iarray.error, "The second dimension of the second bshape must be"
                                        "equal to the second dimension of the output container chunkshape");
            return INA_ERROR(IARRAY_ERR_INVALID_BLOCKSHAPE);
        }
        return _iarray_gemm(ctx, a, b, c, blockshape_a, blockshape_b);
    }
    else {
        return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
    }
}


INA_API(ina_rc_t) iarray_operator_and(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_operator_or(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_operator_xor(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_operator_nand(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_operator_not(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(b);
    INA_UNUSED(result);
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
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(result);
    return INA_ERR_NOT_IMPLEMENTED;
}

INA_API(ina_rc_t) iarray_operator_atan2(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(result);
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
    return _iarray_operator_elwise_a(ctx, a, result, vdLn, vsLn);
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

INA_API(ina_rc_t) iarray_operator_cumsum(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *result)
{
    INA_UNUSED(ctx);
    INA_UNUSED(a);
    INA_UNUSED(result);
    return INA_ERR_NOT_IMPLEMENTED;
}
