/*
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <iarray_private.h>
#include <libiarray/iarray.h>


typedef struct {
    iarray_constructor_array_info_t *array_info;
    iarray_constructor_chunk_info_t *chunk_info;
    void *custom_info;
    void *custom_chunk_info;
    iarray_constructor_generator_fn generator_fn;
    iarray_constructor_block_init_fn block_init_fn;
    iarray_constructor_block_destroy_fn block_destroy_fn;
    iarray_constructor_item_fn item_fn;
} iarray_constructor_params;


static int iarray_constructor_prefilter(blosc2_prefilter_params *pparams) {
    iarray_constructor_params *constructor_params = pparams->user_data;
    iarray_constructor_chunk_info_t *chunk_info = constructor_params->chunk_info;
    iarray_constructor_array_info_t *array_params = constructor_params->array_info;
    void *custom_info = constructor_params->custom_info;
    void *custom_chunk_info = constructor_params->custom_chunk_info;

    iarray_container_t *a = array_params->a;
    uint8_t ndim = array_params->ndim;

    // Compute block information
    iarray_constructor_block_info_t block_info;

    block_info.tid = pparams->tid;
    // index_in_chunk params
    block_info.index_in_chunk_flat = pparams->out_offset / pparams->out_size;
    iarray_index_unidim_to_multidim(ndim, chunk_info->chunk_strides_block,
                                    block_info.index_in_chunk_flat, block_info.index_in_chunk);

    // shape
    int64_t block_start_inside_chunk[IARRAY_DIMENSION_MAX];
    block_info.size = 1;
    for (int i = 0; i < ndim; ++i) {
        block_start_inside_chunk[i] = block_info.index_in_chunk[i] * a->catarr->blockshape[i];
        block_info.start[i] = chunk_info->start[i] + block_start_inside_chunk[i];
        block_info.stop[i] = block_info.start[i] + a->catarr->blockshape[i];
        if (block_info.start[i] > chunk_info->stop[i]) {
            return 0;
        } else if (block_info.stop[i] > chunk_info->stop[i]) {
            block_info.stop[i] = chunk_info->stop[i];
        }
        block_info.shape[i] = block_info.stop[i] - block_info.start[i];
        block_info.size *= block_info.shape[i];
    }
    compute_strides(ndim, block_info.shape, block_info.block_strides);

    // Execute the custom block algorithm
    void *custom_block_info = NULL;
    if (constructor_params->block_init_fn) {
        constructor_params->block_init_fn(array_params, chunk_info, &block_info, custom_info, custom_chunk_info, &custom_block_info);
    }

    // Constructor formula
    constructor_params->generator_fn(pparams->out,
                                     array_params,
                                     chunk_info,
                                     &block_info,
                                     custom_info,
                                     custom_chunk_info,
                                     custom_block_info,
                                     constructor_params->item_fn);

    if (constructor_params->block_destroy_fn) {
        constructor_params->block_destroy_fn(array_params, chunk_info, &block_info, custom_info, custom_chunk_info, &custom_block_info);
    }

    return BLOSC2_ERROR_SUCCESS;
}


ina_rc_t iarray_constructor_block(iarray_context_t *ctx,
                                  iarray_dtshape_t *dtshape,
                                  iarray_constructor_block_params_t *const_params,
                                  iarray_storage_t *storage,
                                  iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    INA_VERIFY_NOT_NULL(container);

    // Create an empty array
    IARRAY_RETURN_IF_FAILED(iarray_empty(ctx, dtshape, storage, container));

    iarray_container_t *c = *container;
    uint8_t ndim = c->catarr->ndim;
    uint8_t itemsize = c->catarr->itemsize;

    // Set up the prefilter
    iarray_context_t *prefilter_ctx;
    iarray_context_new(ctx->cfg, &prefilter_ctx);
    prefilter_ctx->prefilter_fn = (blosc2_prefilter_fn) iarray_constructor_prefilter;
    iarray_constructor_params constructor_params = {0};
    blosc2_prefilter_params prefilter_params = {0};
    prefilter_params.user_data = &constructor_params;
    prefilter_ctx->prefilter_params = &prefilter_params;

    // Fill constructor_params
    constructor_params.custom_info = const_params->constructor_info;
    constructor_params.generator_fn = const_params->generator_fn;
    constructor_params.block_init_fn = const_params->block_init_fn;
    constructor_params.block_destroy_fn = const_params->block_destroy_fn;
    constructor_params.item_fn = const_params->item_fn;

    iarray_constructor_array_info_t array_info;
    constructor_params.array_info = &array_info;
    array_info.ndim = ndim;
    array_info.itemsize = itemsize;
    array_info.a = c;
    compute_strides(ndim, c->dtshape->shape, array_info.strides);
    compute_strides(ndim, c->storage->chunkshape, array_info.chunk_strides);
    compute_strides(ndim, c->storage->blockshape, array_info.block_strides);

    iarray_constructor_chunk_info_t chunk_info;
    constructor_params.chunk_info = &chunk_info;
    int64_t block_in_chunk_shape[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < c->catarr->ndim; ++i) {
        block_in_chunk_shape[i] = c->catarr->extchunkshape[i] / c->catarr->blockshape[i];
    }
    compute_strides(ndim, block_in_chunk_shape, chunk_info.chunk_strides_block);

    int64_t chunk_in_shape_shape[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < c->catarr->ndim; ++i) {
        chunk_in_shape_shape[i] = c->catarr->extshape[i] / c->catarr->chunkshape[i];
    }
    int64_t chunk_in_shape_strides[IARRAY_DIMENSION_MAX];
    compute_strides(c->dtshape->ndim, chunk_in_shape_shape, chunk_in_shape_strides);


    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    IARRAY_RETURN_IF_FAILED(iarray_create_blosc_cparams(&cparams, prefilter_ctx, c->catarr->itemsize,
                                                        c->catarr->blocknitems * c->catarr->itemsize));
    cparams.schunk = c->catarr->sc;
    blosc2_context *cctx = blosc2_create_cctx(cparams);

    // Iterate over chunks
    int64_t c_nchunk = 0;
    while (c_nchunk < c->catarr->nchunks) {
        // Compute index information
        iarray_index_unidim_to_multidim((int8_t) c->catarr->ndim, chunk_in_shape_strides, c_nchunk,
                                        chunk_info.index);
        chunk_info.index_flat = c_nchunk;

        for (int i = 0; i < c->catarr->ndim; ++i) {
            chunk_info.start[i] = chunk_info.index[i] * c->catarr->chunkshape[i];
            chunk_info.stop[i] = chunk_info.start[i] + c->catarr->chunkshape[i];
            if (chunk_info.stop[i] > c->catarr->shape[i]) {
                chunk_info.stop[i] = c->catarr->shape[i];
            }
            chunk_info.shape[i] = chunk_info.stop[i] - chunk_info.shape[i];
        }

        void *custom_chunk_info;
        if (const_params->chunk_init_fn) {
            const_params->chunk_init_fn(&array_info, &chunk_info, const_params->constructor_info, &custom_chunk_info);
        }

        // Compress data using the prefilter
        uint8_t *chunk = malloc(c->catarr->extchunknitems * c->catarr->itemsize +
                                BLOSC2_MAX_OVERHEAD);
        int csize = blosc2_compress_ctx(cctx, NULL, (int32_t) c->catarr->extchunknitems * c->catarr->itemsize,
                                        chunk,
                                        (int32_t) c->catarr->extchunknitems * c->catarr->itemsize +
                                        BLOSC2_MAX_OVERHEAD);
        if (csize <= 0) {
            IARRAY_TRACE1(iarray.error, "Error compressing a blosc index");
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }

        if (const_params->chunk_destroy_fn) {
            const_params->chunk_destroy_fn(&array_info, &chunk_info, const_params->constructor_info, &custom_chunk_info);
        }

        // Update data
        blosc2_schunk_update_chunk(c->catarr->sc, (int) c_nchunk, chunk, false);

        c_nchunk++;
    }

    blosc2_free_ctx(cctx);

    iarray_context_free(&prefilter_ctx);

    return INA_SUCCESS;
}
