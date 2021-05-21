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

#include "iarray_private.h"
#include <libiarray/iarray.h>


static ina_rc_t _iarray_gemm(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *c,
                             int64_t *bshape_a, int64_t *bshape_b) {

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(b);
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(bshape_a);
    INA_VERIFY_NOT_NULL(bshape_b);

    int64_t typesize = a->catarr->itemsize;

    /* Check if the block is equal to the shape */
    bool a_copy = a->storage->backend == IARRAY_STORAGE_PLAINBUFFER ? false : true;
    if (!a_copy) {
        a_copy = a->view ? true : false;
    }
    if (!a_copy) {
        for (int i = 0; i < a->dtshape->ndim; ++i) {
            if (bshape_a[i] != a->dtshape->shape[i]) {
                a_copy = true;
                break;
            }
        }
    }

    bool b_copy = b->storage->backend == IARRAY_STORAGE_PLAINBUFFER ? false : true;
    if (!b_copy) {
        b_copy = b->view ? true : false;
    }
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
    int flag_b = CblasNoTrans;

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

    uint8_t *c_block = NULL;

    caterva_config_t cfg = {0};
    IARRAY_ERR_CATERVA(iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg));
    caterva_ctx_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_ctx_new(&cfg, &cat_ctx));

    if (c->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        c_block = cat_ctx->cfg->alloc(c_size);
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
    IARRAY_RETURN_IF_FAILED(_iarray_iter_matmul_new(ctx, a, b, bshape_a, bshape_b, &iter));
    for (_iarray_iter_matmul_init(iter); !_iarray_iter_matmul_finished(iter); _iarray_iter_matmul_next(iter)) {
        int64_t start_a[IARRAY_DIMENSION_MAX];
        int64_t stop_a[IARRAY_DIMENSION_MAX];
        int64_t cbshape_a[IARRAY_DIMENSION_MAX];
        int64_t csize_a;
        int64_t start_b[IARRAY_DIMENSION_MAX];
        int64_t stop_b[IARRAY_DIMENSION_MAX];
        int64_t cbshape_b[IARRAY_DIMENSION_MAX];
        int64_t csize_b;

        int64_t inc_a = 1;
        int64_t inc_b = 1;

        // the block coords are calculated from the index
        int64_t chunk_ind_a[IARRAY_DIMENSION_MAX];
        int64_t chunk_ind_b[IARRAY_DIMENSION_MAX];

        for (int i = a->dtshape->ndim - 1; i >= 0; --i) {
            chunk_ind_a[i] = iter->nchunk1 % (inc_a * (eshape_a[i] / bshape_a[i])) / inc_a;
            inc_a *= (eshape_a[i] / bshape_a[i]);

            chunk_ind_b[i] = iter->nchunk2 % (inc_b * (eshape_b[i] / bshape_b[i])) / inc_b;
            inc_b *= (eshape_b[i] / bshape_b[i]);
        }

        // a start and a stop are calculated from the block coords
        csize_a = typesize;
        csize_b = typesize;
        for (int i = 0; i < a->dtshape->ndim; ++i) {
            start_a[i] = chunk_ind_a[i] * bshape_a[i];
            start_b[i] = chunk_ind_b[i] * bshape_b[i];
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
            cbshape_a[i] = stop_a[i] - start_a[i];
            cbshape_b[i] = stop_b[i] - start_b[i];
            csize_a *= cbshape_a[i];
            csize_b *= cbshape_b[i];

        }

        // Obtain desired blocks from iarray containers
        if (!a_copy) {
            IARRAY_RETURN_IF_FAILED(_iarray_get_slice_buffer_no_copy(ctx, a, start_a, stop_a, (void **) &a_block, a_size));
        } else {
            IARRAY_RETURN_IF_FAILED(_iarray_get_slice_buffer(ctx, a, start_a, stop_a, cbshape_a, a_block, csize_a));
        }
        if (!b_copy) {
            IARRAY_RETURN_IF_FAILED(_iarray_get_slice_buffer_no_copy(ctx, b, start_b, stop_b, (void **) &b_block, b_size));
        } else {
            IARRAY_RETURN_IF_FAILED(_iarray_get_slice_buffer(ctx, b, start_b, stop_b, cbshape_b, b_block, csize_b));
        }

        int64_t cB0 = cbshape_a[0];
        int64_t cB1 = cbshape_a[1];
        int64_t cB2 = cbshape_b[1];

        int ld_a = (int) cB1;
        int ld_b = (int) cB2;
        int ld_c = (int) cB2;

        // Make blocks multiplication
        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                cblas_dgemm(CblasRowMajor, flag_a, flag_b, (int) cB0, (int) cB2, (int) cB1,
                    1.0, (double *)a_block, ld_a, (double *)b_block, ld_b, 1.0, (double *)c_block, ld_c);

                break;
            case IARRAY_DATA_TYPE_FLOAT:
                cblas_sgemm(CblasRowMajor, flag_a, flag_b, (const int)cB0, (const int)cB2, (const int)cB1,
                    1.0f, (float *)a_block, ld_a, (float *)b_block, ld_b, 1.0f, (float *)c_block, ld_c);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
        }


        if (c->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
            if((iter->cont + 1) % (eshape_a[1] / B1) == 0) {
                c->catarr->buf = c_block;
            }
        } else {
            // Append it to a new iarray container
            if ((iter->cont + 1) % (eshape_a[1] / B1) == 0) {
                IARRAY_ERR_CATERVA(caterva_append(cat_ctx, c->catarr, &c_block[0], cB0 * cB2 * typesize));
                memset(c_block, 0, c_size);
            }
        }
    }
    _iarray_iter_matmul_free(&iter);

    IARRAY_ERR_CATERVA(caterva_ctx_free(&cat_ctx));

    if (a_copy) {
        INA_MEM_FREE_SAFE(a_block);
    }
    if (b_copy) {
        INA_MEM_FREE_SAFE(b_block);
    }

    if (c->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
        INA_MEM_FREE_SAFE(c_block);
    }

    return INA_SUCCESS;
}

static ina_rc_t _iarray_gemv(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *c,
                             int64_t *bshape_a, int64_t *bshape_b) {

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(b);
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(bshape_a);
    INA_VERIFY_NOT_NULL(bshape_b);

    int64_t typesize = a->catarr->itemsize;

    /* Check if the block is equal to the shape */
    bool a_copy = a->storage->backend == IARRAY_STORAGE_PLAINBUFFER ? false : true;
    if (!a_copy) {
        a_copy = a->view ? true : false;
    }
    if (!a_copy) {
        for (int i = 0; i < a->dtshape->ndim; ++i) {
            if (bshape_a[i] != a->dtshape->shape[i]) {
                a_copy = true;
                break;
            }
        }
    }

    bool b_copy = b->storage->backend == IARRAY_STORAGE_PLAINBUFFER ? false : true;
    if (!b_copy) {
        b_copy = b->view ? true : false;
    }
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

    // block sizes are claculated
    size_t a_size = (size_t) B0 * B1 * typesize;
    size_t b_size = (size_t) B1 * typesize;
    size_t c_size = (size_t) B0 * typesize;

    int flag_a = CblasNoTrans;

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

    int dtype = a->dtshape->dtype;

    uint8_t *a_block = NULL;
    uint8_t *b_block = NULL;

    uint8_t *c_block = NULL;

    caterva_config_t cfg = {0};
    IARRAY_ERR_CATERVA(iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg));
    caterva_ctx_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_ctx_new(&cfg, &cat_ctx));

    if (c->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        c_block = cat_ctx->cfg->alloc(c_size);
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
    IARRAY_RETURN_IF_FAILED(_iarray_iter_matmul_new(ctx, a, b, bshape_a, bshape_b, &iter));

    for (_iarray_iter_matmul_init(iter); !_iarray_iter_matmul_finished(iter); _iarray_iter_matmul_next(iter)) {

        int64_t start_a[IARRAY_DIMENSION_MAX];
        int64_t stop_a[IARRAY_DIMENSION_MAX];
        int64_t cbshape_a[IARRAY_DIMENSION_MAX];
        int64_t csize_a;
        int64_t start_b[IARRAY_DIMENSION_MAX];
        int64_t stop_b[IARRAY_DIMENSION_MAX];
        int64_t cbshape_b[IARRAY_DIMENSION_MAX];
        int64_t csize_b;

        int64_t inc_a = 1;

        int64_t chunk_ind_a[IARRAY_DIMENSION_MAX];
        int64_t chunk_ind_b[IARRAY_DIMENSION_MAX];

        // the block coords are calculated from the index
        for (int i = a->dtshape->ndim - 1; i >= 0; --i) {
            chunk_ind_a[i] = iter->nchunk1 % (inc_a * (eshape_a[i] / bshape_a[i])) / inc_a;
            inc_a *= (eshape_a[i] / bshape_a[i]);
        }
        chunk_ind_b[0] = iter->nchunk2 % ( (eshape_b[0] / bshape_b[0]));


        // a start and a stop are calculated from the block coords
        csize_a = typesize;
        for (int i = 0; i < a->dtshape->ndim; ++i) {
            start_a[i] = chunk_ind_a[i] * bshape_a[i];
            if (start_a[i] + bshape_a[i] > a->dtshape->shape[i]) {
                stop_a[i] = a->dtshape->shape[i];
            } else {
                stop_a[i] = start_a[i] + bshape_a[i];
            }
            cbshape_a[i] = stop_a[i] - start_a[i];
            csize_a *= cbshape_a[i];
        }

        csize_b = typesize;
        start_b[0] = chunk_ind_b[0] * bshape_b[0];
        if (start_b[0] + bshape_b[0] > b->dtshape->shape[0]) {
            stop_b[0] = b->dtshape->shape[0];
        } else {
            stop_b[0] = start_b[0] + bshape_b[0];
        }
        cbshape_b[0] = stop_b[0] - start_b[0];
        csize_b *= cbshape_b[0];

        int64_t cB0 = cbshape_a[0];
        int64_t cB1 = cbshape_a[1];

        int ld_a = (int) cB1;

        if (!a_copy) {
            IARRAY_RETURN_IF_FAILED(_iarray_get_slice_buffer_no_copy(ctx, a, start_a, stop_a, (void **) &a_block, a_size));
        } else {
            IARRAY_RETURN_IF_FAILED(_iarray_get_slice_buffer(ctx, a, start_a, stop_a, cbshape_a, a_block, csize_a));
        }
        if (!b_copy) {
            IARRAY_RETURN_IF_FAILED(_iarray_get_slice_buffer_no_copy(ctx, b, start_b, stop_b, (void **) &b_block, b_size));
        } else {
            IARRAY_RETURN_IF_FAILED(_iarray_get_slice_buffer(ctx, b, start_b, stop_b, cbshape_b, b_block, csize_b));
        }

        // Make blocks multiplication
        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                cblas_dgemv(CblasRowMajor, flag_a, (int) cB0, (int) cB1, 1.0, (double *) a_block,
                            ld_a, (double *) b_block, 1, 1.0, (double *) c_block, 1);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                cblas_sgemv(CblasRowMajor, flag_a, (int) cB0, (int) cB1, 1.0f, (float *) a_block,
                            ld_a, (float *) b_block, 1, 1.0f, (float *) c_block, 1);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "The data type is invalid");
                return (INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
        }

        if (c->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
            if((iter->cont + 1) % (eshape_a[1] / B1) == 0) {
                c->catarr->buf = c_block;
            }
        } else {
            // Append it to a new iarray container
            if ((iter->cont + 1) % (eshape_a[1] / B1) == 0) {
                IARRAY_ERR_CATERVA(caterva_append(cat_ctx, c->catarr, &c_block[0], cbshape_a[0] * typesize));
                memset(c_block, 0, c_size);
            }
        }
    }

    _iarray_iter_matmul_free(&iter);

    IARRAY_ERR_CATERVA(caterva_ctx_free(&cat_ctx));

    if (a_copy) {
        INA_MEM_FREE_SAFE(a_block);
    }
    if (b_copy) {
        INA_MEM_FREE_SAFE(b_block);
    }

    if (c->catarr->storage != CATERVA_STORAGE_PLAINBUFFER) {
        INA_MEM_FREE_SAFE(c_block);
    }
    return INA_SUCCESS;
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
