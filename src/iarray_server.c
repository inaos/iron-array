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

#include "iarray_private.h"
#include <libiarray/iarray.h>
#include <stdio.h>


typedef struct {
    char *server_urlpath;
    //!< urlpath to server with arrays.
    char *urlpath;
    //!< urlpath to array with data.
    int32_t blocksize;
    //!< schunk's blocksize
    rhandler_ptr request_handler;
    //!< Function pointer to request handler
} request_postparams_udata;


ina_rc_t request_postfilter(blosc2_postfilter_params *postparams)
{
    request_postparams_udata *udata = postparams->user_data;

    int block_nitems = udata->blocksize / postparams->typesize;
    uint8_t* cblock = malloc(postparams->size + BLOSC_MAX_OVERHEAD);
    int32_t csize = udata->request_handler(udata->server_urlpath, udata->urlpath, postparams->nchunk,
                                               postparams->nblock * block_nitems,
                                               block_nitems, postparams->size, cblock);
    if (csize < 0) {
        IARRAY_TRACE1(iarray.tracing, "could not get block from server");
        return IARRAY_ERR_BLOSC_FAILED;
    }

    blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
    blosc2_context *dctx = blosc2_create_dctx(dparams);
    int rc = blosc2_decompress_ctx(dctx, cblock, csize, postparams->out, postparams->size);
    blosc2_free_ctx(dctx);
    free(cblock);
    if (rc != postparams->size) {
        IARRAY_TRACE1(iarray.tracing, "Error decompressing block");
        return IARRAY_ERR_BLOSC_FAILED;
    }

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_server_job(iarray_context_t *ctx, iarray_container_t *a, int64_t nchunk,
                  int32_t start, int32_t nitems, int32_t size, uint8_t *dest, int32_t *block_size)
{
    INA_UNUSED(ctx);
    uint8_t *chunk;
    bool needs_free;
    int csize = blosc2_schunk_get_lazychunk(a->catarr->sc, nchunk, &chunk, &needs_free);
    if (csize < 0) {
        IARRAY_TRACE1(iarray.tracing, "Error getting lazy chunk");
        return IARRAY_ERR_BLOSC_FAILED;
    }

    uint8_t *dc_block = ina_mem_alloc(size);
    int bsize = blosc2_getitem_ctx(a->catarr->sc->dctx, chunk, csize, start, nitems, dc_block, size);
    if (needs_free) {
        free(chunk);
    }
    if (bsize < 0) {
        IARRAY_TRACE1(iarray.tracing, "Error getting block");
        return IARRAY_ERR_BLOSC_FAILED;
    }

    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    cparams.compcode = BLOSC_ZSTD;
    cparams.clevel = 1;
    blosc2_context *blosc_ctx = blosc2_create_cctx(cparams);

    csize = blosc2_compress_ctx(blosc_ctx, dc_block, bsize, dest, size);
    ina_mem_free(dc_block);
    blosc2_free_ctx(blosc_ctx);
    if (csize <= 0) {
        IARRAY_TRACE1(iarray.tracing, "Error compressing the block");
        return IARRAY_ERR_BLOSC_FAILED;
    }
    *block_size = csize;

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_add_request_postfilter(iarray_container_t *src, char *server_urlpath, char *urlpath,
                                                rhandler_ptr request_handler)
{
    INA_VERIFY_NOT_NULL(src);

    // Create params
    blosc2_dparams *dparams;
    blosc2_schunk_get_dparams(src->catarr->sc, &dparams);

    dparams->postfilter = (blosc2_postfilter_fn)request_postfilter;

    blosc2_postfilter_params *postparams = malloc(sizeof(blosc2_postfilter_params));
    request_postparams_udata *rpostparams = malloc(sizeof(request_postparams_udata));

    // Fill the user_data
    rpostparams->server_urlpath = malloc(strlen(server_urlpath) + 1);
    strcpy(rpostparams->server_urlpath, server_urlpath);

    rpostparams->urlpath = malloc(strlen(urlpath) + 1);
    strcpy(rpostparams->urlpath, urlpath);

    rpostparams->blocksize = src->catarr->sc->blocksize;
    rpostparams->request_handler = request_handler;

    postparams->user_data = (void*)rpostparams;
    dparams->postparams = postparams;

    // Create new context since postparams is empty in the old one
    blosc2_free_ctx(src->catarr->sc->dctx);
    src->catarr->sc->dctx = blosc2_create_dctx(*dparams);
    free(dparams);

    return INA_SUCCESS;
}
