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

static int mult_c(const double *a, const double *b, double *c, const int I, const int J, const int K) {

    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            double sum = 0;
            for (int k = 0; k < K; ++k) {
                sum = sum + a[i * K + k] * b[k * J + j];
            }
            c[i * J + j] += sum;
        }
    }

    return 0;
}

static ina_rc_t _iarray_gemm(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *c,
                             int64_t *bshape_a, int64_t *bshape_b) {

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(b);
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(bshape_a);
    INA_VERIFY_NOT_NULL(bshape_b);

    ina_rc_t rc;

    caterva_dims_t shape = caterva_new_dims(c->dtshape->shape, c->dtshape->ndim);
    IARRAY_ERR_CATERVA(caterva_update_shape(c->catarr, &shape));

    int64_t typesize = a->catarr->ctx->cparams.typesize;

    /* Check if the block is equal to the shape */
    bool a_copy = a->store->backend == IARRAY_STORAGE_PLAINBUFFER ? false : true;
    if (!a_copy) {
        for (int i = 0; i < a->dtshape->ndim; ++i) {
            if (bshape_a[i] != a->dtshape->shape[i]) {
                a_copy = true;
                break;
            }
        }
    }

    bool b_copy = b->store->backend == IARRAY_STORAGE_PLAINBUFFER ? false : true;
    if (!b_copy) {
        for (int i = 0; i < b->dtshape->ndim; ++i) {
            if (bshape_b[i] != b->dtshape->shape[i]) {
                b_copy = true;
                break;
            }
        }
    }

    // define mkl parameters
    int64_t B0 = bshape_a[0];
    int64_t B1 = bshape_a[1];
    int64_t B2 = bshape_b[1];

    int flag_a = CblasNoTrans;
    int ld_a = (int) B1;
    if (a->transposed == 1) {
        flag_a = CblasTrans;
        ld_a = (int) B0;
    }

    int flag_b = CblasNoTrans;
    int ld_b = (int) B2;
    if (b->transposed == 1) {
        flag_b = CblasTrans;
        ld_b = (int) B1;
    }

    int ld_c = (int) B2;

    // the extended shape is recalculated from the block shape
    int64_t eshape_a[IARRAY_DIMENSION_MAX];
    int64_t eshape_b[IARRAY_DIMENSION_MAX];

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

    // block sizes are calculated
    size_t a_size = (size_t) B0 * B1 * typesize;
    size_t b_size = (size_t) B1 * B2 * typesize;
    size_t c_size = (size_t) B0 * B2 * typesize;
    int dtype = a->dtshape->dtype;

    uint8_t *a_block = NULL;
    uint8_t *b_block = NULL;

    uint8_t *c_block;

    if (c->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        c_block = c->catarr->ctx->alloc(c_size);
    } else {
        c_block = ina_mem_alloc(c_size);
    }

    if (a_copy) {
        a_block = ina_mem_alloc(a_size);
    }
    if (b_copy) {
        b_block = ina_mem_alloc(b_size);
    }

    memset(c_block, 0, c_size);

    // Start a iterator that returns the index matrix blocks
    iarray_iter_matmul_t *iter;
    IARRAY_FAIL_IF_ERROR(_iarray_iter_matmul_new(ctx, a, b, bshape_a, bshape_b, &iter));
    for (_iarray_iter_matmul_init(iter); !_iarray_iter_matmul_finished(iter); _iarray_iter_matmul_next(iter)) {
        int64_t start_a[IARRAY_DIMENSION_MAX];
        int64_t stop_a[IARRAY_DIMENSION_MAX];
        int64_t start_b[IARRAY_DIMENSION_MAX];
        int64_t stop_b[IARRAY_DIMENSION_MAX];

        int64_t inc_a = 1;
        int64_t inc_b = 1;

        // the block coords are calculated from the index
        int64_t part_ind_a[IARRAY_DIMENSION_MAX];
        int64_t part_ind_b[IARRAY_DIMENSION_MAX];

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
        if (!a_copy) {
            IARRAY_FAIL_IF_ERROR(_iarray_get_slice_buffer_no_copy(ctx, a, start_a, stop_a, (void **) &a_block, a_size));
        } else {
            IARRAY_FAIL_IF_ERROR(_iarray_get_slice_buffer(ctx, a, start_a, stop_a, bshape_a, a_block, a_size));
        }
        if (!b_copy) {
            IARRAY_FAIL_IF_ERROR(_iarray_get_slice_buffer_no_copy(ctx, b, start_b, stop_b, (void **) &b_block, b_size));
        } else {
            IARRAY_FAIL_IF_ERROR(_iarray_get_slice_buffer(ctx, b, start_b, stop_b, bshape_b, b_block, b_size));
        }

        // Make blocks multiplication

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                cblas_dgemm(CblasRowMajor, flag_a, flag_b, (int) B0, (int) B2, (int) B1,
                    1.0, (double *)a_block, ld_a, (double *)b_block, ld_b, 1.0, (double *)c_block, ld_c);

                break;
            case IARRAY_DATA_TYPE_FLOAT:
                cblas_sgemm(CblasRowMajor, flag_a, flag_b, (const int)B0, (const int)B2, (const int)B1,
                    1.0f, (float *)a_block, ld_a, (float *)b_block, ld_b, 1.0f, (float *)c_block, ld_c);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }


        if (c->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
            if((iter->cont + 1) % (eshape_a[1] / B1) == 0) {
                c->catarr->buf = c_block;
            }
        } else {
            // Append it to a new iarray container
            if ((iter->cont + 1) % (eshape_a[1] / B1) == 0) {
                int blosc_rc = blosc2_schunk_append_buffer(c->catarr->sc, &c_block[0], c_size);
                if (blosc_rc < 0) {
                    IARRAY_TRACE1(iarray.error, "Error appending a buffer to a blosc schunk");
                    IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
                }
                memset(c_block, 0, c_size);
            }
        }
    }

    c->catarr->filled = true;
    rc = INA_SUCCESS;
    goto cleanup;

    fail:
    rc = ina_err_get_rc();
    cleanup:
    _iarray_iter_matmul_free(&iter);

    if (a_copy) {
        INA_MEM_FREE_SAFE(a_block);
    }
    if (b_copy) {
        INA_MEM_FREE_SAFE(b_block);
    }

    if (c->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
        INA_MEM_FREE_SAFE(c_block);
    }

    return rc;
}

static ina_rc_t _iarray_gemv(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *c,
                             int64_t *bshape_a, int64_t *bshape_b) {

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(b);
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(bshape_a);
    INA_VERIFY_NOT_NULL(bshape_b);

    ina_rc_t rc;

    caterva_dims_t shape = caterva_new_dims(c->dtshape->shape, c->dtshape->ndim);
    IARRAY_ERR_CATERVA(caterva_update_shape(c->catarr, &shape));
    int64_t typesize = a->catarr->ctx->cparams.typesize;

    /* Check if the block is equal to the shape */
    bool a_copy = a->store->backend == IARRAY_STORAGE_PLAINBUFFER ? false : true;
    if (!a_copy) {
        for (int i = 0; i < a->dtshape->ndim; ++i) {
            if (bshape_a[i] != a->dtshape->shape[i]) {
                a_copy = true;
                break;
            }
        }
    }

    bool b_copy = b->store->backend == IARRAY_STORAGE_PLAINBUFFER ? false : true;
    if (!b_copy) {
        for (int i = 0; i < b->dtshape->ndim; ++i) {
            if (bshape_b[i] != b->dtshape->shape[i]) {
                b_copy = true;
                break;
            }
        }
    }

    // Define parameters needed in mkl multiplication
    int64_t B0 = bshape_a[0];
    int64_t B1 = bshape_a[1];

    int M = (int) bshape_a[0];
    int K = (int) bshape_a[1];
    int ld_a = K;
    int flag_a = CblasNoTrans;
    if (a->transposed == 1) {
        flag_a = CblasTrans;
        ld_a = M;
        M = (int) bshape_a[1];
        K = (int) bshape_a[0];
    }

    int64_t eshape_a[2];
    int64_t eshape_b[1];

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
    size_t a_size = (size_t) B0 * B1 * typesize;
    size_t b_size = (size_t) B1 * typesize;
    size_t c_size = (size_t) B0 * typesize;

    int dtype = a->dtshape->dtype;

    uint8_t *a_block = NULL;
    uint8_t *b_block = NULL;

    uint8_t *c_block;

    if (c->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        c_block = c->catarr->ctx->alloc(c_size);
    } else {
        c_block = ina_mem_alloc(c_size);
    }

    if (a_copy) {
        a_block = ina_mem_alloc(a_size);
    }
    if (b_copy) {
        b_block = ina_mem_alloc(b_size);
    }

    memset(c_block, 0, c_size);

    // Start a iterator that returns the index matrix blocks
    iarray_iter_matmul_t *iter;
    IARRAY_FAIL_IF_ERROR(_iarray_iter_matmul_new(ctx, a, b, bshape_a, bshape_b, &iter));

    for (_iarray_iter_matmul_init(iter); !_iarray_iter_matmul_finished(iter); _iarray_iter_matmul_next(iter)) {

        int64_t start_a[IARRAY_DIMENSION_MAX];
        int64_t stop_a[IARRAY_DIMENSION_MAX];
        int64_t start_b[IARRAY_DIMENSION_MAX];
        int64_t stop_b[IARRAY_DIMENSION_MAX];

        int64_t inc_a = 1;

        int64_t part_ind_a[IARRAY_DIMENSION_MAX];
        int64_t part_ind_b[IARRAY_DIMENSION_MAX];

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

        if (!a_copy) {
            IARRAY_FAIL_IF_ERROR(_iarray_get_slice_buffer_no_copy(ctx, a, start_a, stop_a, (void **) &a_block, a_size));
        } else {
            IARRAY_FAIL_IF_ERROR(_iarray_get_slice_buffer(ctx, a, start_a, stop_a, bshape_a, a_block, a_size));
        }
        if (!b_copy) {
            IARRAY_FAIL_IF_ERROR(_iarray_get_slice_buffer_no_copy(ctx, b, start_b, stop_b, (void **) &b_block, b_size));
        } else {
            IARRAY_FAIL_IF_ERROR(_iarray_get_slice_buffer(ctx, b, start_b, stop_b, bshape_b, b_block, b_size));
        }

        // Make blocks multiplication
        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                cblas_dgemv(CblasRowMajor, flag_a, M, K, 1.0, (double *) a_block, ld_a, (double *) b_block, 1, 1.0, (double *) c_block, 1);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                cblas_sgemv(CblasRowMajor, flag_a, M, K, 1.0f, (float *) a_block, ld_a, (float *) b_block, 1, 1.0f, (float *) c_block, 1);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }

        if (c->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
            if((iter->cont + 1) % (eshape_a[1] / B1) == 0) {
                c->catarr->buf = c_block;
            }
        } else {
            // Append it to a new iarray container
            if ((iter->cont + 1) % (eshape_a[1] / B1) == 0) {
                blosc2_schunk_append_buffer(c->catarr->sc, &c_block[0], c_size);
                memset(c_block, 0, c_size);
            }
        }
    }

    c->catarr->filled = true;
    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    _iarray_iter_matmul_free(&iter);

    if (a_copy) {
        INA_MEM_FREE_SAFE(a_block);
    }
    if (b_copy) {
        INA_MEM_FREE_SAFE(b_block);
    }

    if (c->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
        INA_MEM_FREE_SAFE(c_block);
    }
    return rc;
}

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

    caterva_dims_t shape = caterva_new_dims(result->dtshape->shape, result->dtshape->ndim);
    IARRAY_ERR_CATERVA(caterva_update_shape(result->catarr, &shape));

    size_t psize = (size_t)a->catarr->sc->typesize;
    for (int i = 0; i < a->catarr->ndim; ++i) {
        psize *= a->catarr->pshape[i];
    }

    int8_t *a_chunk = (int8_t*)ina_mempool_dalloc(ctx->mp_op, psize);
    int8_t *c_chunk = (int8_t*)ina_mempool_dalloc(ctx->mp_op, psize);

    for (int i = 0; i < a->catarr->sc->nchunks; ++i) {
        if (blosc2_schunk_decompress_chunk(a->catarr->sc, i, a_chunk, psize) < 0) {
            IARRAY_TRACE1(iarray.error, "Error decompressing a chunk from a schunk");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
        }

        switch (a->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            mkl_fun_d((const int)(psize / sizeof(double)), (const double*)a_chunk, (double*)c_chunk);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            mkl_fun_s((const int)(psize / sizeof(float)), (const float*)a_chunk, (float*)c_chunk);
            break;
        default:
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }
        if (blosc2_schunk_append_buffer(result->catarr->sc, c_chunk, psize) < 0) {
            IARRAY_TRACE1(iarray.error, "Error appending a buffer to a blosc schunk");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
        }
    }

    result->catarr->filled = true;
    ina_mempool_reset(ctx->mp_op);

    return INA_SUCCESS;

fail:
    ina_mempool_reset(ctx->mp_op);
    /* FIXME: error handling */
    return ina_err_get_rc();
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

    IARRAY_FAIL_IF_ERROR(iarray_container_dtshape_equal(a->dtshape, b->dtshape));

    caterva_dims_t shape = caterva_new_dims(result->dtshape->shape, result->dtshape->ndim);
    IARRAY_ERR_CATERVA(caterva_update_shape(result->catarr, &shape));

    size_t psize = (size_t)a->catarr->sc->typesize;
    for (int i = 0; i < a->catarr->ndim; ++i) {
        if (a->catarr->pshape[i] != b->catarr->pshape[i]) {
            IARRAY_TRACE1(iarray.error, "The pshapes must be equals");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_PSHAPE));
        }
        psize *= a->catarr->pshape[i];
    }

    int8_t *a_chunk = (int8_t*)ina_mempool_dalloc(ctx->mp_op, psize);
    int8_t *b_chunk = (int8_t*)ina_mempool_dalloc(ctx->mp_op, psize);
    int8_t *c_chunk = (int8_t*)ina_mempool_dalloc(ctx->mp_op, psize);

    for (int i = 0; i < a->catarr->sc->nchunks; ++i) {
        if (blosc2_schunk_decompress_chunk(a->catarr->sc, i, a_chunk, psize) < 0) {
            IARRAY_TRACE1(iarray.error, "Error decompressing a chunk from a blosc schunk");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
        }
        if (blosc2_schunk_decompress_chunk(b->catarr->sc, i, b_chunk, psize) < 0) {
            IARRAY_TRACE1(iarray.error, "Error decompressing a chunk from a blosc schunk");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
        }
        switch (a->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_fun_d((const int) (psize/sizeof(double)), (const double*) a_chunk, (const double*) b_chunk, (double*) c_chunk);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_fun_s((const int) (psize / sizeof(float)), (const float*) a_chunk, (const float*) b_chunk, (float*) c_chunk);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }
        if (blosc2_schunk_append_buffer(result->catarr->sc, c_chunk, psize) < 0) {
            IARRAY_TRACE1(iarray.error, "Error appending a buffer to a blosc schunk");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
        }
    }

    result->catarr->filled = true;

    ina_mempool_reset(ctx->mp_op);

    return INA_SUCCESS;

fail:
    ina_mempool_reset(ctx->mp_op);
    /* FIXME: error handling */
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_linalg_transpose(iarray_context_t *ctx, iarray_container_t *a)
{
    INA_VERIFY_NOT_NULL(ctx);
    if (a->dtshape->ndim != 2) {
        IARRAY_TRACE1(iarray.error, "The container dimension is not 2");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }

    if (a->transposed == 0) {
        a->transposed = 1;

    }
    else {
        a->transposed = 0;
    }

    if (a->catarr->storage == CATERVA_STORAGE_BLOSC && blosc2_has_metalayer(a->catarr->sc, "iarray") > 0) {
        uint8_t *content;
        uint32_t content_len;
        blosc2_get_metalayer(a->catarr->sc, "iarray", &content, &content_len);
        *(content + 2) = *(content + 2) ^ 64ULL;
        blosc2_update_metalayer(a->catarr->sc, "iarray", content, content_len);
    }

    int64_t aux[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        aux[i] = a->dtshape->shape[i];
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        a->dtshape->shape[i] = aux[a->dtshape->ndim - 1 - i];
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        aux[i] = a->dtshape->pshape[i];
    }
    for (int i = 0; i < a->dtshape->ndim; ++i) {
        a->dtshape->pshape[i] = aux[a->dtshape->ndim - 1 - i];
    }
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
 * `bshape_a` indicates indicates the block size with which the container `a` will be iterated when
 *  performing block multiplication. The pshape[0] of `c` must be equal to bshape_a[0].
 *
 * `bshape_b` indicates indicates the block size with which the container `b` will be iterated when
 *  performing block multiplication. The pshape[1] of `c` must be equal to bshape_a[1].
 *
 *  In addition, in order to perform the multiplication correctly bshape_a[1] = bshape_b[0].
 *
 *  It is also supported the multiplication between containers with different structures
 *
 *  This function returns an error code ina_rc_t.
 */

INA_API(ina_rc_t) iarray_linalg_matmul(iarray_context_t *ctx,
                                       iarray_container_t *a,
                                       iarray_container_t *b,
                                       iarray_container_t *c,
                                       int64_t *bshape_a,
                                       int64_t *bshape_b,
                                       iarray_operator_hint_t hint)
{
    INA_UNUSED(hint);
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(b);
    INA_VERIFY_NOT_NULL(c);

    if (c->catarr->filled) {
        IARRAY_TRACE1(iarray.error, "The output container must be empty");
        INA_ERROR(IARRAY_ERR_FULL_CONTAINER);
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

    if (bshape_a == NULL) {
        bshape_a = a->dtshape->shape;
    }
    if (bshape_b == NULL) {
        bshape_b = b->dtshape->shape;
    }

    if (bshape_a[1] != bshape_b[0]) {
        IARRAY_TRACE1(iarray.error, "The second dimension of the first bshape must be"
                                    "equal to the first dimension of the second bshape");
        return INA_ERROR(IARRAY_ERR_INVALID_BSHAPE);
    }

    if (bshape_a[0] != c->dtshape->pshape[0]){
        IARRAY_TRACE1(iarray.error, "The first dimension of the first bshape must be"
                                    "equal to the first dimension of the output container pshape");
        return INA_ERROR(IARRAY_ERR_INVALID_BSHAPE);
    }

    if (b->dtshape->ndim == 1) {
        return _iarray_gemv(ctx, a, b, c, bshape_a, bshape_b);
    }
    else if (b->dtshape->ndim == 2) {
        if (bshape_b[1] != c->dtshape->pshape[1]) {
            IARRAY_TRACE1(iarray.error, "The second dimension of the second bshape must be"
                                        "equal to the second dimension of the output container pshape");
            return INA_ERROR(IARRAY_ERR_INVALID_BSHAPE);
        }
        return _iarray_gemm(ctx, a, b, c, bshape_a, bshape_b);
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
