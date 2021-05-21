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
#include "gemm.h"

typedef struct iarray_parallel_matmul_params_s {
    iarray_container_t *a;
    iarray_container_t *b;
    iarray_container_t *c;
    uint8_t *cache_a;
    uint8_t *cache_b;
} iarray_parallel_matmul_params_t;

static void _gemm_prefilter_block_info(iarray_container_t *c,
                                       int32_t offset,
                                       int32_t size,
                                       int32_t *start) {

    int8_t ndim = c->dtshape->ndim;

    int64_t strides[2] = {0};
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

    for (int i = 0; i < ndim; ++i) {
        start[i] = start_in_chunk[i];
    }
}

static int _gemm_prefilter(blosc2_prefilter_params *pparams) {

    iarray_parallel_matmul_params_t *matmul_params = (iarray_parallel_matmul_params_t *) pparams->user_data;
    iarray_container_t *a = matmul_params->a;
    iarray_container_t *b = matmul_params->b;
    iarray_container_t *c = matmul_params->c;

    // Compute block info
    int32_t start[2] = {0};
    _gemm_prefilter_block_info(matmul_params->c, pparams->out_offset, pparams->out_size, start);

    uint8_t* buffer_a = &matmul_params->cache_a[start[0] * a->dtshape->shape[1] * a->catarr->itemsize];
    uint8_t* buffer_b = &matmul_params->cache_b[start[1] * b->dtshape->shape[0] * b->catarr->itemsize];

    int trans_a = CblasNoTrans;
    int trans_b = CblasNoTrans;

    int m = c->storage->blockshape[0];
    int k = a->dtshape->shape[1];
    int n = c->storage->blockshape[1];

    int ld_a = k;
    int ld_b = n;
    int ld_c = n;

    if (c->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        cblas_dgemm(CblasRowMajor, trans_a, trans_b, (int) m, (int) n, (int) k,
                    1.0, (double *) buffer_a, ld_a, (double *) buffer_b, ld_b, 0.0, (double *) pparams->out, ld_c);
    } else {
        cblas_sgemm(CblasRowMajor, trans_a, trans_b, (int) m, (int) n, (int) k,
                    1.0f, (float *) buffer_a, ld_a, (float *) buffer_b, ld_b, 0.0f, (float *) pparams->out, ld_c);
    }

    return 0;
}


static ina_rc_t _gemm_repart_caches(iarray_context_t *ctx,
                                    int64_t m,
                                    int64_t n,
                                    int64_t k,
                                    int8_t itemsize,
                                    uint8_t *cache_aux,
                                    uint8_t *cache) {

    int64_t cache_aux_pointer = 0;
    int64_t inc = k;
    for (int nblock = 0; nblock < n / k; ++nblock) {
        int64_t start = nblock * inc;
        for (int i = 0; i < m; ++i) {
            memcpy(&cache[cache_aux_pointer * itemsize], &cache_aux[(i * n + start) * itemsize], inc * itemsize);
            cache_aux_pointer += inc;
        }
    }
    return INA_SUCCESS;
}


static ina_rc_t gemm_blosc(iarray_context_t *ctx,
                           iarray_container_t *a,
                           iarray_container_t *b,
                           iarray_container_t *c) {

    // Set up prefilter
    iarray_context_t *prefilter_ctx = ina_mem_alloc(sizeof(iarray_context_t));
    memcpy(prefilter_ctx, ctx, sizeof(iarray_context_t));
    prefilter_ctx->prefilter_fn = (blosc2_prefilter_fn) _gemm_prefilter;
    iarray_parallel_matmul_params_t matmul_params = {0};
    matmul_params.a = a;
    matmul_params.b = b;
    matmul_params.c = c;
    blosc2_prefilter_params pparams = {0};
    pparams.user_data = &matmul_params;
    prefilter_ctx->prefilter_params = &pparams;

    // Init caches

    int64_t cache_size_b = b->dtshape->shape[0] * c->catarr->extchunkshape[1] * c->catarr->itemsize;
    uint8_t *cache_b = ina_mem_alloc(cache_size_b);

    int64_t cache_size_a = c->catarr->extchunkshape[0] * a->dtshape->shape[1] * c->catarr->itemsize;
    int64_t cache_alloc_size = cache_size_a > cache_size_b ? cache_size_a : cache_size_b;

    uint8_t *cache_a = ina_mem_alloc(cache_alloc_size);
    uint8_t *cache_aux_b = cache_a;

    matmul_params.cache_a = cache_a;
    matmul_params.cache_b = cache_b;

    // Start iterator

    int64_t chunk_index[2] = {0};
    int64_t nchunk = 0;
    int64_t chunk_row = -1;
    int *chunk_order = (int *) malloc(c->catarr->extnitems / c->catarr->chunknitems * sizeof(int));

    while (nchunk < c->catarr->extnitems / c->catarr->chunknitems) {
        int64_t elem_index[2] = {0};
        for (int i = 0; i < 2; ++i) {
            elem_index[i] = chunk_index[i] * c->catarr->chunkshape[i];
        }
        int64_t chunk_shape[2] = {0};
        for (int i = 0; i < 2; ++i) {
            if (elem_index[i] + c->catarr->chunkshape[i] <= c->catarr->shape[i]) {
                chunk_shape[i] = c->catarr->chunkshape[i];
            } else {
                chunk_shape[i] = c->catarr->shape[i] - elem_index[i];
            }
        }

        // Compute starts and stops
        if (chunk_row != chunk_index[1]) {
            chunk_row = chunk_index[1];
            int64_t start_b[2] = {0};
            start_b[0] = 0;
            start_b[1] = elem_index[1];
            int64_t stop_b[2] = {0};
            stop_b[0] = b->dtshape->shape[0];
            stop_b[1] = elem_index[1] + chunk_shape[1];

            int64_t shape_b[2] = {0};
            shape_b[0] = b->dtshape->shape[0];
            shape_b[1] = c->catarr->extchunkshape[1];

            _iarray_get_slice_buffer(ctx, b, start_b, stop_b, shape_b, cache_aux_b, cache_size_b);
            _gemm_repart_caches(ctx, shape_b[0], shape_b[1], c->catarr->blockshape[1], c->catarr->itemsize,
                                cache_aux_b, cache_b);

        }

        int64_t start_a[2] = {0};
        start_a[0] = elem_index[0];
        start_a[1] = 0;
        int64_t stop_a[2] = {0};
        stop_a[0] = elem_index[0] + chunk_shape[0];
        stop_a[1] = a->dtshape->shape[1];

        int64_t shape_a[2] = {0};
        shape_a[0] = c->catarr->extchunkshape[0];
        shape_a[1] = a->dtshape->shape[1];

        IARRAY_RETURN_IF_FAILED(_iarray_get_slice_buffer(ctx, a, start_a, stop_a, shape_a, cache_a, cache_size_a));

        // Compress data
        blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
        IARRAY_RETURN_IF_FAILED(iarray_create_blosc_cparams(&cparams, prefilter_ctx, c->catarr->itemsize,
                                                            c->catarr->blocknitems * c->catarr->itemsize));
        blosc2_context *cctx = blosc2_create_cctx(cparams);
        uint8_t *chunk = malloc(c->catarr->extchunknitems * c->catarr->itemsize +
                                BLOSC_MAX_OVERHEAD);
        int csize = blosc2_compress_ctx(cctx, NULL, c->catarr->extchunknitems * c->catarr->itemsize,
                                        chunk,
                                        c->catarr->extchunknitems * c->catarr->itemsize +
                                        BLOSC_MAX_OVERHEAD);
        if (csize <= 0) {
            IARRAY_TRACE1(iarray.error, "Error compressing a blosc chunk");
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }
        blosc2_free_ctx(cctx);

        // Append to schunk
        blosc2_schunk_update_chunk(c->catarr->sc, nchunk, chunk, false);

        int new_position = chunk_index[1] + chunk_index[0] * (c->catarr->extshape[1] / c->catarr->chunkshape[1]);
        chunk_order[new_position] = nchunk;
        nchunk++;

        chunk_index[0] = nchunk % (c->catarr->extshape[0] / c->catarr->chunkshape[0]);
        chunk_index[1] = nchunk / (c->catarr->extshape[0] / c->catarr->chunkshape[0]);
    }

    blosc2_schunk_reorder_offsets(c->catarr->sc, chunk_order);

    INA_MEM_FREE_SAFE(cache_a);
    INA_MEM_FREE_SAFE(cache_b);

    return INA_SUCCESS;
}

static ina_rc_t gemm_plainbuffer(iarray_context_t *ctx,
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


INA_API(ina_rc_t) iarray_gemm(iarray_context_t *ctx,
                              iarray_container_t *a,
                              iarray_container_t *b,
                              iarray_container_t *c) {
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(b);
    INA_VERIFY_NOT_NULL(c);


    if (c->storage->backend == IARRAY_STORAGE_PLAINBUFFER) {
        IARRAY_RETURN_IF_FAILED(gemm_plainbuffer(ctx, a, b, c));
    } else {
        int nthreads = mkl_get_max_threads();
        mkl_set_num_threads(1);
        IARRAY_RETURN_IF_FAILED(gemm_blosc(ctx, a, b, c));
        mkl_set_num_threads(nthreads);
    }
    return INA_SUCCESS;
}
