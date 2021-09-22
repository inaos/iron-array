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
#include "iarray_reduce_operations.h"
#include <libiarray/iarray.h>


static bool chunk_is_zeros(uint8_t *chunk) {
    uint8_t blosc2_flags = *(chunk + BLOSC2_CHUNK_BLOSC2_FLAGS);
    uint8_t special_value = (blosc2_flags >> 4) & BLOSC2_SPECIAL_MASK;
    if (special_value == BLOSC2_SPECIAL_ZERO) {
        return true;
    } else {
        return false;
    }
}


static bool block_is_zeros(uint8_t *chunk, int64_t nblock) {
    uint8_t blosc_flags = *(chunk + BLOSC2_CHUNK_FLAGS);
    bool memcpyed = blosc_flags & 0x02u;
    bool split = blosc_flags & 0x4u;
    uint8_t itemsize = *(chunk + BLOSC2_CHUNK_TYPESIZE);

    uint8_t blosc2_flags = *(chunk + BLOSC2_CHUNK_BLOSC2_FLAGS);
    uint8_t special_value = (blosc2_flags >> 4) & BLOSC2_SPECIAL_MASK;
    bool lazy = blosc2_flags & 0x08u;

    if (memcpyed || special_value) {
        return false;
    }

    if (lazy) {
        uint8_t *bstarts = chunk + BLOSC_EXTENDED_HEADER_LENGTH;

        int32_t nbytes = *(int32_t *)(chunk + BLOSC2_CHUNK_NBYTES);  // TODO: Fix endian
        int32_t blocksize = *(int32_t *)(chunk + BLOSC2_CHUNK_BLOCKSIZE);  // TODO: Fix endian
        int32_t nblocks = nbytes % blocksize == 0 ? nbytes / blocksize : nbytes / blocksize + 1;

        uint8_t *trailer = bstarts + sizeof(int32_t) * nblocks;
        uint8_t *csizes = trailer + sizeof(int32_t) + sizeof(int64_t);
        int32_t csize = ((int32_t *) csizes)[nblock];  // TODO: Fix endian
        if (csize != 0) {
            return false;
        }
    } else {
        uint8_t *bstarts = chunk + BLOSC_EXTENDED_HEADER_LENGTH;
        int32_t bstart = ((int32_t *) bstarts)[nblock];  // TODO: Fix endian

        uint8_t *cdata = chunk + bstart;
        if (split) {
            for (int i = 0; i < itemsize; ++i) {
                int32_t csize = ((int32_t *) cdata)[0];  // TODO: Fix endian
                if (csize != 0) {
                    return false;
                }
                cdata += sizeof(int32_t) + csize;
            }
        } else {
            int32_t csize = ((int32_t *) cdata)[0];  // TODO: Fix endian
            if (csize != 0) {
                return false;
            }
        }
    }

    return true;
}

typedef struct iarray_gemm2_params_s {
    iarray_container_t *a;
    iarray_container_t *b;
    uint8_t *b_blocks;
    bool *b_block_zeros;
} iarray_gemm2_params_t;


static int _gemm2_prefilter(blosc2_prefilter_params *pparams) {
    iarray_gemm2_params_t *gparams = (iarray_gemm2_params_t *) pparams->user_data;
    iarray_container_t *a = gparams->a;
    iarray_container_t *b = gparams->b;

    uint8_t *b_blocks = gparams->b_blocks;
    bool *b_block_zeros = gparams->b_block_zeros;

    // printf("C_nchunk: %lld, %lld\n", c_chunk[0], c_chunk[1]);

    blosc2_dparams a_dparams = {.nthreads = 1, .schunk = a->catarr->sc};
    blosc2_context *a_dctx = blosc2_create_dctx(a_dparams);

    uint8_t *a_block = ina_mem_alloc_aligned(64, a->catarr->blocknitems * a->catarr->itemsize);

    for (int i = 0; i < a->catarr->blockshape[0] * b->catarr->blockshape[1]; ++i) {
        switch(a->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                ((double *) pparams->out)[i] = 0;
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                ((float *) pparams->out)[i] = 0;
                break;
            default:
                IARRAY_TRACE1(iarray.tracing, "dtype not supported");
                return -1;
        }
    }

    int64_t c_nblock = pparams->out_offset / pparams->out_size;
    int64_t a_nchunk = c_nblock;

    uint8_t *a_chunk;
    bool a_needs_free;
    int a_csize = blosc2_schunk_get_lazychunk(a->catarr->sc, (int) a_nchunk, &a_chunk, &a_needs_free);
    if (a_csize < 0) {
        IARRAY_TRACE1(iarray.tracing, "Error getting lazy a_chunk");
        return -1;
    }

    if (chunk_is_zeros(a_chunk)) {
        if (a_needs_free) {
            free(a_chunk);
            INA_MEM_FREE_SAFE(a_block);
        }
        return 0;
    }

    int32_t a_nblocks_in_chunk = (int32_t) a->catarr->extchunkshape[1] / a->catarr->blockshape[1];

    for (int a_nblock = 0; a_nblock < a_nblocks_in_chunk; ++a_nblock) {
        int b_nblock = a_nblock;
        if (b_block_zeros[b_nblock]) {
            continue;
        }
        if (block_is_zeros(a_chunk, a_nblock)) {
            continue;
        }

        int a_start = (int) a_nblock * a->catarr->blocknitems;

        int a_bsize = blosc2_getitem_ctx(a_dctx, a_chunk, a_csize, a_start,
                                         a->catarr->blocknitems, a_block,
                                         a->catarr->blocknitems * a->catarr->itemsize);
        if (a_bsize < 0) {
            IARRAY_TRACE1(iarray.tracing, "Error getting block");
            return -1;
        }

        int b_start = (int) b_nblock * b->catarr->blocknitems;
        uint8_t *b_block = &b_blocks[b_start * b->catarr->itemsize];

        if (true) {
            switch (a->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                (int) a->catarr->blockshape[0],
                                (int) b->catarr->blockshape[1],
                                (int) a->catarr->blockshape[1],
                                1.0, (double *) a_block, (int) a->catarr->blockshape[1],
                                (double *) b_block, (int) b->catarr->blockshape[1],
                                1.0, (double *) pparams->out, (int) b->catarr->blockshape[1]);
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                (int) a->catarr->blockshape[0],
                                (int) b->catarr->blockshape[1],
                                (int) a->catarr->blockshape[1],
                                1.0f, (float *) a_block, (int) a->catarr->blockshape[1],
                                (float *) b_block, (int) b->catarr->blockshape[1],
                                1.0f, (float *) pparams->out, (int) b->catarr->blockshape[1]);
                    break;
                default:
                    IARRAY_TRACE1(iarray.tracing, "dtype not supported");
                    return -1;
            }
        }
    }

    INA_MEM_FREE_SAFE(a_block);
    blosc2_free_ctx(a_dctx);

    return 0;
}


INA_API(ina_rc_t) iarray_opt_gemm2(iarray_context_t *ctx,
                                  iarray_container_t *a,
                                  iarray_container_t *b,
                                  iarray_storage_t *storage,
                                  iarray_container_t **c) {

    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(b);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(c);

    if (a->storage->backend == IARRAY_STORAGE_PLAINBUFFER) {
        IARRAY_TRACE1(iarray.error, "gemm2 can not be performed over a plainbuffer "
                                    "container");
        return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
    }
    if (b->storage->backend == IARRAY_STORAGE_PLAINBUFFER) {
        IARRAY_TRACE1(iarray.error, "gemm2 can not be performed over a plainbuffer "
                                    "container");
        return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
    }

    if (a->catarr->ndim != 2) {
        IARRAY_TRACE1(iarray.error, "The a dimension must be 2");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }
    if (b->catarr->ndim != 2) {
        IARRAY_TRACE1(iarray.error, "The b dimension must be 2");
        return INA_ERROR(IARRAY_ERR_INVALID_NDIM);
    }

    if (a->catarr->chunkshape[1] != a->catarr->shape[1]) {
        IARRAY_TRACE1(iarray.error, "a->chunkshape[1]  != a->shape[1]");
        return INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE);
    }

    if (b->catarr->chunkshape[0] != b->catarr->shape[0]) {
        IARRAY_TRACE1(iarray.error, "b->chunkshape[0] != c->chunkshape[0]");
        return INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE);
    }


    if (a->catarr->shape[0] != storage->chunkshape[0]) {
        IARRAY_TRACE1(iarray.error, "a->shape[0] != c->chunkshape[0]");
        return INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE);
    }
    if (a->catarr->chunkshape[1] != b->catarr->chunkshape[0]) {
        IARRAY_TRACE1(iarray.error, "a->chunkshape[1] != b->chunkshape[0]");
        return INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE);
    }
    if (b->catarr->chunkshape[1] != storage->chunkshape[1]) {
        IARRAY_TRACE1(iarray.error, "b->chunkshape[1] != c->chunkshape[1]");
        return INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE);
    }

    if (a->catarr->chunkshape[0] != storage->blockshape[0]) {
        IARRAY_TRACE1(iarray.error, "a->chunkshape[0] != c->blockshape[0]");
        return INA_ERROR(IARRAY_ERR_INVALID_BLOCKSHAPE);
    }

    if (b->catarr->chunkshape[1] != storage->blockshape[1]) {
        IARRAY_TRACE1(iarray.error, "b->chunkshape[1] != c->blockshape[1]");
        return INA_ERROR(IARRAY_ERR_INVALID_BLOCKSHAPE);
    }
    if (a->catarr->blockshape[0] != b->catarr->blockshape[1]) {
        IARRAY_TRACE1(iarray.error, "a->blockshape[0] != b->blockshape[1]");
        return INA_ERROR(IARRAY_ERR_INVALID_BLOCKSHAPE);
    }

    int nthreads = mkl_get_max_threads();
    mkl_set_num_threads(1);

    iarray_dtshape_t dtshape;
    dtshape.dtype = a->dtshape->dtype;
    dtshape.ndim = 2;
    dtshape.shape[0] = a->dtshape->shape[0];
    dtshape.shape[1] = b->dtshape->shape[1];

    IARRAY_RETURN_IF_FAILED(iarray_empty(ctx, &dtshape, storage, 0, c));

    iarray_container_t *cc = *c;

    // Set up prefilter
    iarray_context_t *prefilter_ctx;
    iarray_context_new(ctx->cfg, &prefilter_ctx);
    prefilter_ctx->prefilter_fn = (blosc2_prefilter_fn) _gemm2_prefilter;
    iarray_gemm2_params_t gemm2_params = {0};
    blosc2_prefilter_params pparams = {0};
    pparams.user_data = &gemm2_params;
    prefilter_ctx->prefilter_params = &pparams;

    // Fill prefilter params
    gemm2_params.a = a;
    gemm2_params.b = b;

    int32_t b_nblocks_in_chunk = (int32_t) b->catarr->extchunkshape[0] / b->catarr->blockshape[0];
    int32_t b_nbytes = b->catarr->sc->chunksize;
    uint8_t *b_blocks = ina_mem_alloc(b_nbytes);
    bool *b_block_zeros = ina_mem_alloc(b_nblocks_in_chunk);
    blosc2_dparams b_dparams = {.nthreads = 1, .schunk = b->catarr->sc};
    blosc2_context *b_dctx = blosc2_create_dctx(b_dparams);

    gemm2_params.b_blocks = b_blocks;
    gemm2_params.b_block_zeros = b_block_zeros;


    // Iterate over chunks
    int64_t c_nchunks = cc->catarr->nchunks;
    int64_t c_nchunk = 0;
    while (c_nchunk < c_nchunks) {
        uint8_t *b_chunk;
        bool needs_free;
        int b_csize = blosc2_schunk_get_lazychunk(b->catarr->sc, (int) c_nchunk, &b_chunk, &needs_free);
        for (int b_nblock = 0; b_nblock < b_nblocks_in_chunk; ++b_nblock) {
            b_block_zeros[b_nblock] = block_is_zeros(b_chunk, b_nblock);
            if(!b_block_zeros[b_nblock]) {
                int b_start = (int) b_nblock * b->catarr->blocknitems;
                int b_bsize = blosc2_getitem_ctx(b_dctx, b_chunk, b_csize,
                                                 b_start, b->catarr->blocknitems,
                                                 &b_blocks[b_start * b->catarr->itemsize],
                                                 b->catarr->blocknitems * b->catarr->itemsize);
                if (b_bsize < 0) {
                    IARRAY_TRACE1(iarray.tracing, "Error getting block");
                    return -1;
                }

            }
        }
        if (needs_free) {
            free(b_chunk);
        }

        // Compress data
        blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
        IARRAY_RETURN_IF_FAILED(iarray_create_blosc_cparams(&cparams, prefilter_ctx, cc->catarr->itemsize,
                                                            cc->catarr->blocknitems * cc->catarr->itemsize));
        cparams.schunk = a->catarr->sc;
        blosc2_context *cctx = blosc2_create_cctx(cparams);
        uint8_t *chunk = malloc(cc->catarr->extchunknitems * cc->catarr->itemsize +
                                BLOSC_MAX_OVERHEAD);
        int csize = blosc2_compress_ctx(cctx, NULL, (int32_t) cc->catarr->extchunknitems * cc->catarr->itemsize,
                                        chunk,
                                        (int32_t) cc->catarr->extchunknitems * cc->catarr->itemsize +
                                        BLOSC_MAX_OVERHEAD);
        if (csize <= 0) {
            IARRAY_TRACE1(iarray.error, "Error compressing a blosc chunk");
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }
        blosc2_free_ctx(cctx);

        blosc2_schunk_update_chunk(cc->catarr->sc, (int) c_nchunk, chunk, false);

        c_nchunk++;
    }

    blosc2_free_ctx(b_dctx);

    INA_MEM_FREE_SAFE(b_blocks);
    INA_MEM_FREE_SAFE(b_block_zeros);

    mkl_set_num_threads(nthreads);

    iarray_context_free(&prefilter_ctx);

    return INA_SUCCESS;
}
