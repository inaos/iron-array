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


void _iarray_prefilter_block_info(iarray_container_t *c,
                                  int64_t *chunk_index,
                                  int32_t offset,
                                  int32_t size,
                                  int32_t *start,
                                  int32_t *shape,
                                  int32_t *strides
                                  ) {

    int8_t ndim = c->dtshape->ndim;
    
    // Element strides (in elements)
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0 ; --i) {
        strides[i] = strides[i+1] * c->catarr->blockshape[i+1];
    }

    // Block strides (in blocks)
    int32_t strides_block[IARRAY_DIMENSION_MAX];
    strides_block[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0 ; --i) {
        strides_block[i] = strides_block[i+1] * (int32_t) (c->catarr->extchunkshape[i+1] / 
                c->catarr->blockshape[i+1]);
    }

    // Flattened block number
    int32_t nblock = offset / size;

    // Multidimensional block number
    int32_t nblock_ndim[IARRAY_DIMENSION_MAX];
    for (int i = ndim - 1; i >= 0; --i) {
        if (i != 0) {
            nblock_ndim[i] = (nblock % strides_block[i-1]) / strides_block[i];
        } else {
            nblock_ndim[i] = (nblock % (c->catarr->extchunknitems / c->catarr->blocknitems)) / strides_block[i];
        }
    }

    // Position of the first element of the block (inside current chunk)
    int64_t start_in_chunk[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ndim; ++i) {
        start_in_chunk[i] = nblock_ndim[i] * c->catarr->blockshape[i];
    }

    // Position of the first element of the block (inside container)
    for (int i = 0; i < ndim; ++i) {
        start[i] = start_in_chunk[i] + chunk_index[i] * c->catarr->chunkshape[i];
    }

    // Check if the block is out of bounds
    bool out_of_bounds = false;
    for (int i = 0; i < ndim; ++i) {
        if (start[i] > c->catarr->shape[i]) {
            out_of_bounds = true;
            break;
        }
    }

    // Shape of the current block
    for (int i = 0; i < ndim; ++i) {
        if (out_of_bounds) {
            shape[i] = 0;
        } else if (start[i] + c->catarr->blockshape[i] > c->catarr->shape[i]) {
            shape[i] = (int32_t) (c->catarr->shape[i] - start[i]);
        } else if (start_in_chunk[i] + c->catarr->blockshape[i] > c->catarr->chunkshape[i]) {
            shape[i] = (int32_t) (c->catarr->chunkshape[i] - start_in_chunk[i]);
        } else {
            shape[i] = c->catarr->blockshape[i];
        }
    }
}

typedef struct iarray_parallel_matmul_params_s {
    iarray_container_t *a;
    iarray_container_t *b;
    iarray_container_t *c;
    int64_t *chunk_index;
} iarray_parallel_matmul_params_t;


int _iarray_matmul_prefilter(blosc2_prefilter_params *pparams) {

    iarray_parallel_matmul_params_t *matmul_params = (iarray_parallel_matmul_params_t *) pparams->user_data;
    iarray_container_t *a = matmul_params->a;
    iarray_container_t *b = matmul_params->b;
    iarray_container_t *c = matmul_params->c;

    // Compute block info
    int32_t start[2] = {0};
    int32_t shape[2] = {0};
    int32_t strides[2] = {0};
    _iarray_prefilter_block_info(matmul_params->c, matmul_params->chunk_index, pparams->out_offset, pparams->out_size,
                                 start,
                                 shape,
                                 strides);

    if (shape[0] == 0 || shape[1] == 0) {
        // All block elements are padding
        memset(pparams->out, 0, pparams->out_size);
        return 0;
    }

    // Create single-thread context
    iarray_config_t st_cfg = IARRAY_CONFIG_DEFAULTS;
    st_cfg.max_num_threads = 1;
    iarray_context_t *st_ctx;
    iarray_context_new(&st_cfg, &st_ctx);

    // Extract desired slide from a
    int64_t start_a[2] = {0};
    start_a[0] = start[0];
    start_a[1] = 0;

    int64_t stop_a[2] = {0};
    stop_a[0] = start[0] + shape[0];
    stop_a[1] = a->dtshape->shape[1];

    int64_t shape_a[2] = {0};
    shape_a[0] = c->storage->blockshape[0];
    shape_a[1] = a->dtshape->shape[1];

    int64_t buffer_a_size = shape_a[0] * shape_a[1] * a->catarr->itemsize;
    void* buffer_a = ina_mem_alloc(buffer_a_size);

    if (INA_FAILED(_iarray_get_slice_buffer(st_ctx, a, start_a, stop_a, shape_a, buffer_a, buffer_a_size))) {
        printf("Error getting slice\n");
        return -1;
    }

    // Extract desired slide from b
    int64_t start_b[2] = {0};
    start_b[0] = 0;
    start_b[1] = start[1];

    int64_t stop_b[2] = {0};
    stop_b[0] = b->dtshape->shape[0];
    stop_b[1] = start[1] + shape[1];

    int64_t shape_b[2] = {0};
    shape_b[0] = b->dtshape->shape[0];
    shape_b[1] = c->storage->blockshape[1];

    int64_t buffer_b_size = shape_b[0] * shape_b[1] * b->catarr->itemsize;
    void* buffer_b = ina_mem_alloc(buffer_b_size);

    if (INA_FAILED(_iarray_get_slice_buffer(st_ctx, b, start_b, stop_b, shape_b, buffer_b, buffer_b_size))) {
        printf("Error getting slice\n");
        return -1;
    }

    int trans_a = a->transposed ? CblasTrans : CblasNoTrans;
    int trans_b = b->transposed ? CblasTrans : CblasNoTrans;

    int m = shape_a[0];
    int k = shape_a[1];
    int n = shape_b[1];

    int ld_a = a->transposed ? m : k;
    int ld_b = b->transposed ? k : n;
    int ld_c = n;

    if (c->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        cblas_dgemm(CblasRowMajor, trans_a, trans_b, (int) m, (int) n, (int) k,
                    1.0, (double *) buffer_a, ld_a, (double *) buffer_b, ld_b, 0.0, (double *) pparams->out, ld_c);
    } else {
        cblas_sgemm(CblasRowMajor, trans_a, trans_b, (int) m, (int) n, (int) k,
                    1.0f, (float *) buffer_a, ld_a, (float *) buffer_b, ld_b, 0.0f, (float *) pparams->out, ld_c);
    }

    INA_MEM_FREE_SAFE(buffer_a);
    INA_MEM_FREE_SAFE(buffer_b);
    iarray_context_free(&st_ctx);

    return 0;
}


INA_API(ina_rc_t) iarray_linalg_matmul_blosc(iarray_context_t *ctx,
                                             iarray_container_t *a,
                                             iarray_container_t *b,
                                             iarray_container_t *c) {
    iarray_container_t *out = c;

    // Set up prefilter
    iarray_context_t *prefilter_ctx = ina_mem_alloc(sizeof(iarray_context_t));
    memcpy(prefilter_ctx, ctx, sizeof(iarray_context_t));
    prefilter_ctx->prefilter_fn = (blosc2_prefilter_fn) _iarray_matmul_prefilter;
    iarray_parallel_matmul_params_t matmul_params = {0};
    matmul_params.a = a;
    matmul_params.b = b;
    matmul_params.c = out;
    blosc2_prefilter_params pparams = {0};
    pparams.user_data = &matmul_params;
    prefilter_ctx->prefilter_params = &pparams;

    // Write array using an iterator
    iarray_iter_write_block_t *iter;
    iarray_iter_write_block_value_t iter_value;
    IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_new(ctx, &iter, out, out->storage->chunkshape, &iter_value,
                                                        true));

    size_t external_buffer_size = out->catarr->extchunknitems * out->catarr->itemsize + BLOSC_MAX_OVERHEAD;
    void *external_buffer = NULL;

    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter))) {
        external_buffer = malloc(external_buffer_size);
        IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_next(iter, external_buffer, external_buffer_size));

        blosc2_cparams cparams = {0};
        IARRAY_RETURN_IF_FAILED(iarray_create_blosc_cparams(&cparams, prefilter_ctx, out->catarr->itemsize,
                                                               out->catarr->blocknitems * out->catarr->itemsize));

        matmul_params.chunk_index = iter_value.block_index;

        blosc2_context *cctx = blosc2_create_cctx(cparams);
        int csize = blosc2_compress_ctx(cctx, out->catarr->extchunknitems * out->catarr->itemsize,
                                        NULL, iter_value.block_pointer,
                                        out->catarr->extchunknitems * out->catarr->itemsize +
                                        BLOSC_MAX_OVERHEAD);
        if (csize <= 0) {
            IARRAY_TRACE1(iarray.error, "Error compressing a blosc chunk");
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }
        blosc2_free_ctx(cctx);
        
        iter->compressed_chunk_buffer = true;
    }
    IARRAY_ITER_FINISH();
    iarray_iter_write_block_free(&iter);

    INA_MEM_FREE_SAFE(prefilter_ctx);

    return INA_SUCCESS;
}

ina_rc_t iarray_linalg_matmul_plainbuffer(iarray_context_t *ctx,
                                          iarray_container_t *a,
                                          iarray_container_t *b,
                                          iarray_container_t *c) {

    // Write array using an iterator
    iarray_iter_write_block_t *iter;
    iarray_iter_write_block_value_t iter_value;
    IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_new(ctx, &iter, c, c->dtshape->shape, &iter_value, false));

    // Only one iteration is done
    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_next(iter, NULL, 0));

        size_t size_a = a->catarr->nitems * a->catarr->itemsize;
        uint8_t *buffer_a = malloc(size_a);
        IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, a, buffer_a, size_a));
        size_t size_b = b->catarr->nitems * b->catarr->itemsize;
        uint8_t *buffer_b = malloc(size_b);
        IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, b, buffer_b, size_b));

        int m = a->dtshape->shape[0];
        int k = a->dtshape->shape[1];
        int n = b->dtshape->shape[1];

        int ld_a = k;
        int ld_b = n;
        int ld_c = n;

        if (c->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) m, (int) n, (int) k,
                        1.0, (double *) buffer_a, ld_a, (double *) buffer_b, ld_b, 0.0, (double *) iter_value.block_pointer,
                        ld_c);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) m, (int) n, (int) k,
                        1.0f, (float *) buffer_a, ld_a, (float *) buffer_b, ld_b, 0.0f, (float *) iter_value.block_pointer,
                        ld_c);
        }

    }
    IARRAY_ITER_FINISH();
    iarray_iter_write_block_free(&iter);

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_linalg_parallel_matmul(iarray_context_t *ctx,
                                                iarray_container_t *a,
                                                iarray_container_t *b,
                                                iarray_storage_t *storage,
                                                iarray_container_t **c) {
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(b);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(c);
    
    // Inputs checking
    if (a->dtshape->shape[1] != b->dtshape->shape[0]) {
        return INA_ERROR(IARRAY_ERR_INVALID_SHAPE);
    }
    if (a->dtshape->dtype != b->dtshape->dtype) {
        return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }

    iarray_dtshape_t dtshape = {0};
    dtshape.dtype = a->dtshape->dtype;
    dtshape.ndim = 2;
    dtshape.shape[0] = a->dtshape->shape[0];
    dtshape.shape[1] = b->dtshape->shape[1];

    for (int i = 0; i < dtshape.ndim; ++i) {
        if (dtshape.shape[i] < storage->chunkshape[i]) {
            return INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE);
        }
    }

    // Create output array
    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, &dtshape, storage, 0, c));

    if ((*c)->storage->backend == IARRAY_STORAGE_PLAINBUFFER) {
        IARRAY_RETURN_IF_FAILED(iarray_linalg_matmul_plainbuffer(ctx, a, b, *c));
    } else {
        IARRAY_RETURN_IF_FAILED(iarray_linalg_matmul_blosc(ctx, a, b, *c));
    }
    return INA_SUCCESS;
}
