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

#include <contribs/tinyexpr/tinyexpr.h>

#include <iarray_private.h>

#define _IARRAY_MEMPOOL_EVAL_SIZE (8*1024*1024)
#define _IARRAY_EXPR_VAR_MAX      (128)

struct iarray_context_s {
    iarray_config_t *cfg;
    ina_mempool_t *mp;
    /* FIXME: track expressions -> list */
};

typedef struct _iarray_tinyexpr_var_s {
    const char *var;
    iarray_container_t *c;
} _iarray_tinyexpr_var_t;

struct iarray_expression_s {
    iarray_context_t *ctx;
    ina_str_t expr;
    size_t nchunks;
    size_t blocksize;
    size_t typesize;
    size_t chunksize;
    int nvars;
    te_expr *texpr;
    iarray_temporary_t **temp_vars;
    iarray_container_t *out;
    _iarray_tinyexpr_var_t vars[_IARRAY_EXPR_VAR_MAX];
};

typedef struct _iarray_container_store_s {
    ina_str_t id;
} _iarray_container_store_t;

struct iarray_container_s {
    iarray_dtshape_t *dtshape;
    blosc2_cparams *cparams;
    blosc2_dparams *dparams;
    caterva_dims_t *pshape;
    caterva_dims_t *shape;
    blosc2_frame *frame;
    caterva_array_t *catarr;
    _iarray_container_store_t *store;
    union {
        float f;
        double d;
    } scalar_value;
};

static int _ina_inited = 0;
static int _blosc_inited = 0;

static ina_rc_t _iarray_container_new(iarray_context_t *ctx, iarray_dtshape_t *dtshape,
                                      iarray_store_properties_t *store,
                                      int flags,
                                      iarray_container_t **c)
{
    blosc2_cparams cparams = BLOSC_CPARAMS_DEFAULTS;
    blosc2_dparams dparams = BLOSC_DPARAMS_DEFAULTS;
    caterva_dims_t pshape;
    caterva_dims_t shape;
    int blosc_filter_idx = 0;

    /* validation */
    if (dtshape->ndim > CATERVA_MAXDIM) {
        return INA_ERROR(INA_ERR_EXCEEDED);
    }
    if (flags & IARRAY_CONTAINER_PERSIST && store == NULL) {
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }
    for (int i = 0; i < dtshape->ndim; ++i) {
        if (dtshape->shape[i] < dtshape->partshape[i]) {
            return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
        }
    }

    *c = (iarray_container_t*)ina_mem_alloc(sizeof(iarray_container_t));
    INA_RETURN_IF_NULL(c);

    (*c)->dtshape = (iarray_dtshape_t*)ina_mem_alloc(sizeof(iarray_dtshape_t));
    INA_FAIL_IF((*c)->dtshape == NULL);
    ina_mem_cpy((*c)->dtshape, dtshape, sizeof(iarray_dtshape_t));

    (*c)->frame = (blosc2_frame*)ina_mem_alloc(sizeof(blosc2_frame));
    INA_FAIL_IF((*c)->frame == NULL);
    ina_mem_cpy((*c)->frame, &BLOSC_EMPTY_FRAME, sizeof(blosc2_frame));

    (*c)->cparams = (blosc2_cparams*)ina_mem_alloc(sizeof(blosc2_cparams));
    INA_FAIL_IF((*c)->cparams == NULL);

    (*c)->dparams = (blosc2_dparams*)ina_mem_alloc(sizeof(blosc2_dparams));
    INA_FAIL_IF((*c)->dparams == NULL);

    (*c)->shape = (caterva_dims_t*)ina_mem_alloc(sizeof(caterva_dims_t));
    INA_FAIL_IF((*c)->shape == NULL);

    (*c)->pshape = (caterva_dims_t*)ina_mem_alloc(sizeof(caterva_dims_t));
    INA_FAIL_IF((*c)->pshape == NULL);

    if (flags & IARRAY_CONTAINER_PERSIST) {
        (*c)->store = ina_mem_alloc(sizeof(_iarray_container_store_t));
        INA_FAIL_IF((*c)->store == NULL);
        (*c)->store->id = ina_str_new_fromcstr(store->id);
        (*c)->frame->fname = (char*)ina_str_cstr((*c)->store->id); /* FIXME: shouldn't fname be a const char? */
    }

    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            cparams.typesize = sizeof(double);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            cparams.typesize = sizeof(float);
            break;
    }
    cparams.compcode = ctx->cfg->compression_codec;
    cparams.clevel = (uint8_t)ctx->cfg->compression_level; /* Since its just a mapping, we know the cast is ok */
    cparams.blocksize = ctx->cfg->blocksize;
    cparams.nthreads = (uint16_t)ctx->cfg->max_num_threads; /* Since its just a mapping, we know the cast is ok */
    if (dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE && ctx->cfg->flags & IARRAY_COMP_TRUNC_PREC) {
        cparams.filters[blosc_filter_idx] = BLOSC_TRUNC_PREC;
        cparams.filters_meta[blosc_filter_idx] = ctx->cfg->fp_mantissa_bits;
        blosc_filter_idx++;
    }
    if (ctx->cfg->flags & IARRAY_COMP_BITSHUFFLE) {
        cparams.filters[blosc_filter_idx] = BLOSC_BITSHUFFLE;
        blosc_filter_idx++;
    }
    if (ctx->cfg->flags & IARRAY_COMP_SHUFFLE) {
        cparams.filters[blosc_filter_idx] = BLOSC_SHUFFLE;
        blosc_filter_idx++;
    }
    if (ctx->cfg->flags & IARRAY_COMP_DELTA) {
        cparams.filters[blosc_filter_idx] = BLOSC_DELTA;
        blosc_filter_idx++;
    }
    ina_mem_cpy((*c)->cparams, &cparams, sizeof(blosc2_cparams));

    dparams.nthreads = (uint16_t)ctx->cfg->max_num_threads; /* Since its just a mapping, we know the cast is ok */
    ina_mem_cpy((*c)->dparams, &dparams, sizeof(blosc2_dparams));

    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        shape.dims[i] = 1;
        pshape.dims[i] = 1;
    }
    for (int i = 0; i < dtshape->ndim; ++i) { // FIXME: 1's at the beginning should be removed
        shape.dims[i] = dtshape->shape[i];
        pshape.dims[i] = dtshape->partshape[i];
    }
    shape.ndim = dtshape->ndim;
    pshape.ndim = dtshape->ndim;

    ina_mem_cpy((*c)->shape, &shape, sizeof(caterva_dims_t));
    ina_mem_cpy((*c)->pshape, &pshape, sizeof(caterva_dims_t));

    caterva_ctx_t *cat_ctx = caterva_new_ctx(NULL, NULL, cparams, dparams);

    (*c)->catarr = caterva_empty_array(cat_ctx, (*c)->frame, *(*c)->pshape);
    INA_FAIL_IF((*c)->catarr == NULL);

    return INA_SUCCESS;

fail:
    iarray_container_free(ctx, c);
    caterva_free_ctx(cat_ctx);
    return ina_err_get_rc();
}

static ina_rc_t _iarray_container_fill_float(iarray_container_t *c, float value)
{
    caterva_fill(c->catarr, *c->shape, &value);
    return INA_SUCCESS;
}

static ina_rc_t _iarray_container_fill_double(iarray_container_t *c, double value)
{
    caterva_fill(c->catarr, *c->shape, &value);
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_init()
{
    if (!_ina_inited) {
        ina_init();
        _ina_inited = 1;
    }
    if (!_blosc_inited) {
        blosc_init();
        _blosc_inited = 1;
    }
    return INA_SUCCESS;
}

INA_API(void) iarray_destroy()
{
    blosc_destroy();
    _blosc_inited = 0;
}

INA_API(ina_rc_t) iarray_context_new(iarray_config_t *cfg, iarray_context_t **ctx)
{
    INA_VERIFY_NOT_NULL(ctx);
    *ctx = ina_mem_alloc(sizeof(iarray_context_t));
    INA_RETURN_IF_NULL(ctx);
    (*ctx)->cfg = ina_mem_alloc(sizeof(iarray_config_t));
    INA_FAIL_IF((*ctx)->cfg == NULL);
    ina_mem_cpy((*ctx)->cfg, cfg, sizeof(iarray_config_t));
    if (!(cfg->flags & IARRAY_EXPR_EVAL_BLOCK) && !(cfg->flags & IARRAY_EXPR_EVAL_CHUNK)) {
        (*ctx)->cfg->flags |= IARRAY_EXPR_EVAL_CHUNK;
    }
    INA_FAIL_IF_ERROR(ina_mempool_new(_IARRAY_MEMPOOL_EVAL_SIZE, NULL, INA_MEM_DYNAMIC, &(*ctx)->mp));
    return INA_SUCCESS;

fail:
    iarray_context_free(ctx);
    return ina_err_get_rc();
}

INA_API(void) iarray_context_free(iarray_context_t **ctx)
{
    INA_VERIFY_FREE(ctx);
    ina_mempool_free(&(*ctx)->mp);
    INA_MEM_FREE_SAFE((*ctx)->cfg);
    INA_MEM_FREE_SAFE(*ctx);
}

INA_API(ina_rc_t) iarray_container_new(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    return _iarray_container_new(ctx, dtshape, store, flags, container);
}

INA_API(ina_rc_t) iarray_arange(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    int start,
    int stop,
    int step,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    /* implement arange */

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_zeros(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            INA_FAIL_IF_ERROR(_iarray_container_fill_double(*container, 0.0));
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            INA_FAIL_IF_ERROR(_iarray_container_fill_float(*container, 0.0f));
            break;
    }
    return INA_SUCCESS;
fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_ones(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    switch (dtshape->dtype) {
    case IARRAY_DATA_TYPE_DOUBLE:
        INA_FAIL_IF_ERROR(_iarray_container_fill_double(*container, 1.0));
        break;
    case IARRAY_DATA_TYPE_FLOAT:
        INA_FAIL_IF_ERROR(_iarray_container_fill_float(*container, 1.0f));
        break;
    }
    return INA_SUCCESS;
fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_fill_float(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    float value,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    INA_FAIL_IF_ERROR(_iarray_container_fill_float(*container, value));

    return INA_SUCCESS;

fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_fill_double(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    double value,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    INA_FAIL_IF_ERROR(_iarray_container_fill_double(*container, value));

    return INA_SUCCESS;

fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_rand(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    iarray_rng_t rng,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    /* implement rand */

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_slice(iarray_context_t *ctx,
    iarray_container_t *c,
    uint64_t *start_,
    uint64_t *stop_,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(start_);
    INA_VERIFY_NOT_NULL(stop_);
    INA_VERIFY_NOT_NULL(container);

    caterva_dims_t start = caterva_new_dims(start_, c->dtshape->ndim);
    caterva_dims_t stop = caterva_new_dims(stop_, c->dtshape->ndim);

    iarray_dtshape_t dtshape;
    for (int i = 0; i < c->dtshape->ndim; ++i) {
        dtshape.shape[i] = (stop_[i] - start_[i]);
        dtshape.partshape[i] = c->dtshape->partshape[i];
    }
    dtshape.ndim = c->dtshape->ndim;
    dtshape.dtype = c->dtshape->dtype;
    INA_RETURN_IF_FAILED(iarray_container_new(ctx, &dtshape, store, flags, container));

    INA_FAIL_IF(caterva_get_slice((*container)->catarr, c->catarr, start, stop) != 0);

    return INA_SUCCESS;

fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_from_buffer(iarray_context_t *ctx,
    iarray_dtshape_t *dtshape,
    void *buffer,
    size_t buffer_len,
    iarray_store_properties_t *store,
    int flags,
    iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(buffer);
    INA_VERIFY_NOT_NULL(container);

    INA_RETURN_IF_FAILED(_iarray_container_new(ctx, dtshape, store, flags, container));

    if (caterva_from_buffer((*container)->catarr, *(*container)->shape, buffer) != 0) {
        INA_ERROR(INA_ERR_FAILED);
        INA_FAIL_IF(1);
    }

    return INA_SUCCESS;

fail:
    iarray_container_free(ctx, container);
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_to_buffer(iarray_context_t *ctx,
    iarray_container_t *container,
    void *buffer,
    size_t buffer_len)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(buffer);
    INA_VERIFY_NOT_NULL(container);

    if (caterva_to_buffer(container->catarr, buffer) != 0) {
        return INA_ERROR(INA_ERR_FAILED);
    }

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_container_info(iarray_container_t *c,
    uint64_t *nbytes,
    uint64_t *cbytes)
{
    INA_VERIFY_NOT_NULL(c);

    *nbytes = (uint64_t) c->catarr->sc->nbytes;
    *cbytes = (uint64_t) c->catarr->sc->cbytes;

    return INA_SUCCESS;
}

INA_API(void) iarray_container_free(iarray_context_t *ctx, iarray_container_t **container)
{
    INA_VERIFY_FREE(container);
    if ((*container)->catarr != NULL) {
        caterva_free_array((*container)->catarr);
    }
    INA_MEM_FREE_SAFE((*container)->frame);
    INA_MEM_FREE_SAFE((*container)->cparams);
    INA_MEM_FREE_SAFE((*container)->dparams);
    INA_MEM_FREE_SAFE((*container)->shape);
    INA_MEM_FREE_SAFE((*container)->pshape);
    INA_MEM_FREE_SAFE((*container)->dtshape);
    INA_MEM_FREE_SAFE(*container);
}

INA_API(ina_rc_t) iarray_expr_new(iarray_context_t *ctx, iarray_expression_t **e)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(e);
    *e = ina_mem_alloc(sizeof(iarray_expression_t));
    INA_RETURN_IF_NULL(e);
    (*e)->ctx = ctx;
    (*e)->nvars = 0;
    (*e)->temp_vars = ina_mem_alloc(sizeof(iarray_temporary_t*)*_IARRAY_EXPR_VAR_MAX);
    ina_mem_set(&(*e)->vars, 0, sizeof(_iarray_tinyexpr_var_t)*_IARRAY_EXPR_VAR_MAX);
    return INA_SUCCESS;
}

INA_API(void) iarray_expr_free(iarray_context_t *ctx, iarray_expression_t **e)
{
    INA_ASSERT_NOT_NULL(ctx);
    INA_VERIFY_FREE(e);
    ina_mempool_reset(ctx->mp); // FIXME
    INA_MEM_FREE_SAFE((*e)->temp_vars);
    INA_MEM_FREE_SAFE(*e);
}

INA_API(ina_rc_t) iarray_expr_bind(iarray_expression_t *e, const char *var, iarray_container_t *val)
{
    if (val->dtshape->ndim > 2) {
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }
    e->vars[e->nvars].var = var;
    e->vars[e->nvars].c = val;
    e->nvars++;
    return INA_SUCCESS;
}

//INA_API(ina_rc_t) iarray_expr_bind_scalar_float(iarray_expression_t *e, const char *var, float val)
//{
//  iarray_container_t *c = ina_mempool_dalloc(e->mp, sizeof(iarray_container_t));
//  c->dtshape = ina_mempool_dalloc(e->mp, sizeof(iarray_dtshape_t));
//  c->dtshape->ndim = 0;
//  c->dtshape->dims = NULL;
//  c->dtshape->dtype = IARRAY_DATA_TYPE_FLOAT;
//  c->scalar_value.f = val;
//  return INA_SUCCESS;
//}

INA_API(ina_rc_t) iarray_expr_bind_scalar_double(iarray_expression_t *e, const char *var, double val)
{
    iarray_container_t *c = ina_mempool_dalloc(e->ctx->mp, sizeof(iarray_container_t));
    c->dtshape = ina_mempool_dalloc(e->ctx->mp, sizeof(iarray_dtshape_t));
    c->dtshape->ndim = 0;
    c->dtshape->dtype = IARRAY_DATA_TYPE_DOUBLE;
    c->scalar_value.d = val;
    e->vars[e->nvars].var = var;
    e->vars[e->nvars].c = c;
    e->nvars++;
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_expr_compile(iarray_expression_t *e, const char *expr)
{
    e->expr = ina_str_new_fromcstr(expr);
    te_variable *te_vars = ina_mempool_dalloc(e->ctx->mp, e->nvars * sizeof(te_variable));
    caterva_array_t *catarr = e->vars[0].c->catarr;
    blosc2_schunk *schunk = catarr->sc;
    int dim0 = 0;
    if (e->ctx->cfg->flags & IARRAY_EXPR_EVAL_BLOCK) {
        int typesize = schunk->typesize;
        int nchunks = schunk->nchunks;
        void *chunk;
        bool needs_free;
        int retcode = blosc2_schunk_get_chunk(schunk, 0, &chunk, &needs_free);
        size_t chunksize, cbytes, blocksize;
        blosc_cbuffer_sizes(chunk, &chunksize, &cbytes, &blocksize);
        if (needs_free) {
            free(chunk);
        }
        dim0 = (int)blocksize / typesize;
        e->nchunks = nchunks;
        e->chunksize = chunksize;
        e->blocksize = blocksize;
        e->typesize = typesize;
    }
    else if (e->ctx->cfg->flags & IARRAY_EXPR_EVAL_CHUNK) {
        dim0 = schunk->chunksize / schunk->typesize;
        e->nchunks = schunk->nchunks;
        e->chunksize = schunk->chunksize;
        e->typesize = schunk->typesize;
    }
    else {
        fprintf(stderr, "Flag %d is not supported\n", e->ctx->cfg->flags);
        return INA_ERR_NOT_SUPPORTED;
    }
    iarray_dtshape_t shape_var = {
        .ndim = 1,
        .shape = {dim0},
        .dtype = e->vars[0].c->dtshape->dtype,
    };
    for (int nvar = 0; nvar < e->nvars; nvar++) {
        iarray_temporary_new(e, e->vars[nvar].c, &shape_var, &e->temp_vars[nvar]);
        te_vars[nvar].name = e->vars[nvar].var;
        te_vars[nvar].address = &e->temp_vars[nvar];
        te_vars[nvar].type = TE_VARIABLE;
        te_vars[nvar].context = NULL;
    }
    int err = 0;
    e->texpr = te_compile(e, ina_str_cstr(e->expr), te_vars, e->nvars, &err);
    if (e->texpr == 0) {
        return INA_ERROR(INA_ERR_NOT_COMPILED);
    }
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_eval(iarray_expression_t *e, iarray_container_t *ret)
{
    blosc2_schunk *schunk0 = e->vars[0].c->catarr->sc;  // get the super-chunk of the first variable
    size_t nitems_in_schunk = schunk0->nbytes / e->typesize;
    size_t nitems_in_chunk = e->chunksize / e->typesize;
    int nvars = e->nvars;
    caterva_update_shape(ret->catarr, *e->vars[0].c->shape);
    caterva_array_t out = *ret->catarr;

    if (e->ctx->cfg->flags & IARRAY_EXPR_EVAL_BLOCK) {
        int8_t *outbuf = ina_mem_alloc(e->chunksize);  // FIXME: this could benefit from using a mempool (probably not)
        size_t nitems = e->blocksize / e->typesize;
        void **var_chunks = ina_mem_alloc(nvars * sizeof(void*));
        bool *var_needs_free = ina_mem_alloc(nvars * sizeof(bool));
        // Allocate a buffer for every (compressed) chunk
        for (int nvar = 0; nvar < nvars; nvar++) {
            //var_chunks[nvar] = ina_mem_alloc(e->chunksize);  // FIXME: looks like this does not work correctly
            var_chunks[nvar] = malloc(e->chunksize);
            var_needs_free[nvar] = false;
        }
        for (size_t nchunk = 0; nchunk < e->nchunks; nchunk++) {
            size_t chunksize = (nchunk < e->nchunks - 1) ? e->chunksize : schunk0->nbytes - nchunk * e->chunksize;
            size_t nblocks_in_chunk = chunksize / e->blocksize;
            size_t corrected_blocksize = e->blocksize;
            size_t corrected_nitems = nitems;
            if (nblocks_in_chunk * e->blocksize < e->chunksize) {
                nitems_in_chunk = chunksize / e->typesize;
                nblocks_in_chunk += 1;
            }
            // Allocate a buffer for every chunk (specially useful for reading on-disk variables)
            for (int nvar = 0; nvar < nvars; nvar++) {
                blosc2_schunk *schunk = e->vars[nvar].c->catarr->sc;
                int retcode = blosc2_schunk_get_chunk(schunk, (int)nchunk, &var_chunks[nvar], &var_needs_free[nvar]);
            }
//#pragma omp parallel for schedule(dynamic)
            for (size_t nblock = 0; nblock < nblocks_in_chunk; nblock++) {
                if ((nblock + 1 == nblocks_in_chunk) && (nblock + 1) * e->blocksize > chunksize) {
                    corrected_blocksize = chunksize - nblock * e->blocksize;
                    corrected_nitems = (int)corrected_blocksize / e->typesize;
                }
                // Decompress blocks in variables into temporaries
                for (int nvar = 0; nvar < nvars; nvar++) {
                    int dsize = blosc_getitem(var_chunks[nvar], (int)(nblock * nitems), (int)corrected_nitems, e->temp_vars[nvar]->data);
                    if (dsize < 0) {
                        printf("Decompression error.  Error code: %d\n", dsize);
                        return INA_ERR_FAILED;
                    }
                }
                // Evaluate the expression for this block
                const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
                ina_mem_cpy(outbuf + nblock * e->blocksize, expr_out->data, corrected_blocksize);
            }
            blosc2_schunk_append_buffer(out.sc, outbuf, nitems_in_chunk * e->typesize);
        }
        for (int nvar = 0; nvar < nvars; nvar++) {
            if (var_needs_free[nvar]) {
                //ina_mem_free(var_chunks[nvar]);  // this raises an error (bug in the ina library?)
                free(var_chunks[nvar]);
            }
        }
        ina_mem_free(var_chunks);
        ina_mem_free(var_needs_free);
        ina_mem_free(outbuf);
    }
    else {
        // Evaluate the expression for all the chunks in variables
        for (size_t nchunk = 0; nchunk < e->nchunks; nchunk++) {
            // Decompress chunks in variables into temporaries
            for (int nvar = 0; nvar < nvars; nvar++) {
                blosc2_schunk *schunk = e->vars[nvar].c->catarr->sc;
                int dsize = blosc2_schunk_decompress_chunk(schunk, (int)nchunk, e->temp_vars[nvar]->data, e->chunksize);
                if (dsize < 0) {
                    printf("Decompression error.  Error code: %d\n", dsize);
                    return INA_ERR_FAILED;
                }
            }
            const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
            // Correct the number of items in last chunk
            nitems_in_chunk = (nchunk < e->nchunks - 1) ? nitems_in_chunk : nitems_in_schunk - nchunk * nitems_in_chunk;
            blosc2_schunk_append_buffer(out.sc, expr_out->data, nitems_in_chunk * e->typesize);
        }
    }
    return INA_SUCCESS;
}

ina_rc_t iarray_shape_size(iarray_dtshape_t *dtshape, size_t *size)
{
    size_t type_size = 0;
    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            type_size = sizeof(double);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            type_size = sizeof(float);
            break;
    }
    for (int i = 0; i < dtshape->ndim; ++i) {
        *size += dtshape->shape[i] * type_size;
    }
    return INA_SUCCESS;
}

ina_rc_t iarray_temporary_new(iarray_expression_t *expr, iarray_container_t *c, iarray_dtshape_t *dtshape,
        iarray_temporary_t **temp)
{
    *temp = ina_mempool_dalloc(expr->ctx->mp, sizeof(iarray_temporary_t));
    (*temp)->dtshape = ina_mempool_dalloc(expr->ctx->mp, sizeof(iarray_dtshape_t));
    ina_mem_cpy((*temp)->dtshape, dtshape, sizeof(iarray_dtshape_t));
    size_t size = 0;
    iarray_shape_size(dtshape, &size);
    (*temp)->size = size;
    if (c != NULL) {
        // FIXME: support float values too
        ina_mem_cpy(&(*temp)->scalar_value, &c->scalar_value, sizeof(double));
    }
    if (size > 0) {
        (*temp)->data = ina_mempool_dalloc(expr->ctx->mp, size);
    }

    return INA_SUCCESS;
}

static iarray_temporary_t* _iarray_op(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_optype_t op)
{
    bool scalar = false;
    bool scalar_vector = false;
    bool vector_vector = false;
    iarray_dtshape_t dtshape;
    ina_mem_set(&dtshape, 0, sizeof(iarray_dtshape_t));
    iarray_blas_type_t op_type = IARRAY_OPERATION_TYPE_BLAS1;
    iarray_temporary_t *scalar_tmp = NULL;
    iarray_temporary_t *scalar_lhs = NULL;
    iarray_temporary_t *out;

    if (lhs->dtshape->ndim == 0 && rhs->dtshape->ndim == 0) {   /* scalar-scalar */
        dtshape.dtype = rhs->dtshape->dtype;
        dtshape.ndim = rhs->dtshape->ndim;
        memcpy(dtshape.shape, rhs->dtshape->shape, sizeof(int) * dtshape.ndim);
        scalar = true;
    }
    else if (lhs->dtshape->ndim == 0 || rhs->dtshape->ndim == 0) {   /* scalar-vector */
        if (lhs->dtshape->ndim == 0) {
            dtshape.dtype = rhs->dtshape->dtype;
            dtshape.ndim = rhs->dtshape->ndim;
            ina_mem_cpy(dtshape.shape, rhs->dtshape->shape, sizeof(int) * dtshape.ndim);
            scalar_tmp = lhs;
            scalar_lhs = rhs;
        }
        else {
            dtshape.dtype = lhs->dtshape->dtype;
            dtshape.ndim = lhs->dtshape->ndim;
            ina_mem_cpy(dtshape.shape, lhs->dtshape->shape, sizeof(int) * dtshape.ndim);
            scalar_tmp = rhs;
            scalar_lhs = lhs;
        }
        scalar_vector = true;
    }
    else if (lhs->dtshape->ndim == 1 && rhs->dtshape->ndim == 1) { /* vector-vector */
        dtshape.dtype = lhs->dtshape->dtype;
        dtshape.ndim = lhs->dtshape->ndim;
        ina_mem_cpy(dtshape.shape, lhs->dtshape->shape, sizeof(int)*lhs->dtshape->ndim);
        vector_vector = true;
    }
    else {
        /* FIXME: matrix/vector and matrix/matrix addition */
    }

    iarray_temporary_new(expr, NULL, &dtshape, &out);

    switch (dtshape.dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
        {
            int len = (int)out->size / sizeof(double);
            if (scalar) {
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
                    out->scalar_value.d = lhs->scalar_value.d + rhs->scalar_value.d;
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
                    out->scalar_value.d = lhs->scalar_value.d - rhs->scalar_value.d;
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
                    out->scalar_value.d = lhs->scalar_value.d * rhs->scalar_value.d;
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
                    out->scalar_value.d = lhs->scalar_value.d / rhs->scalar_value.d;
                    break;
                default:
                    printf("Operation not supported yet");
                }
            }
            else if (scalar_vector) {
                double dscalar = scalar_tmp->scalar_value.d;
                double *odata = (double*)out->data;
                double *ldata = (double*)scalar_lhs->data;
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] + dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] - dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] * dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] / dscalar;
                    }
                    break;
                default:
                    printf("Operation not supported yet");
                }
            }
            else if (vector_vector) {
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] + ((double*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] - ((double*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] * ((double*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] / ((double*)rhs->data)[i];
                    }
                    break;
                default:
                    printf("Operation not supported yet");
                }
            }
            else {
                printf("DTshape combination not supported yet\n");
                return NULL;
            }
        }
        break;
        case IARRAY_DATA_TYPE_FLOAT:
        {
            int len = (int)out->size / sizeof(float);
            if (scalar) {
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
                    out->scalar_value.f = lhs->scalar_value.f + rhs->scalar_value.f;
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
                    out->scalar_value.f = lhs->scalar_value.f - rhs->scalar_value.f;
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
                    out->scalar_value.f = lhs->scalar_value.f * rhs->scalar_value.f;
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
                    out->scalar_value.f = lhs->scalar_value.f / rhs->scalar_value.f;
                    break;
                default:
                    printf("Operation not supported yet");
                }
            }
            else if (scalar_vector) {
                float dscalar = (float)scalar_tmp->scalar_value.d;
                float *odata = (float*)out->data;
                float *ldata = (float*)scalar_lhs->data;
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] + dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] - dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] * dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] / dscalar;
                    }
                    break;
                default:
                    printf("Operation not supported yet");
                }
            }
            else if (vector_vector) {
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] + ((float*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] - ((float*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] * ((float*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] / ((float*)rhs->data)[i];
                    }
                    break;
                default:
                    printf("Operation not supported yet");
                }
            }
            else {
                printf("DTshape combination not supported yet\n");
                return NULL;
            }
        }
        break;
    }

    return out;
}

iarray_temporary_t* _iarray_op_add(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs)
{
    return _iarray_op(expr, lhs, rhs, IARRAY_OPERATION_TYPE_ADD);
}

iarray_temporary_t* _iarray_op_sub(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs)
{
    return _iarray_op(expr, lhs, rhs, IARRAY_OPERATION_TYPE_SUB);
}

iarray_temporary_t* _iarray_op_mul(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs)
{
    return _iarray_op(expr, lhs, rhs, IARRAY_OPERATION_TYPE_MUL);
}

iarray_temporary_t* _iarray_op_divide(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs)
{
    return _iarray_op(expr, lhs, rhs, IARRAY_OPERATION_TYPE_DIVIDE);
}

INA_API(ina_rc_t) iarray_expr_get_mp(iarray_expression_t *e, ina_mempool_t **mp)
{
    *mp = e->ctx->mp;
    return INA_SUCCESS;
}


static int _dtshape_equal(iarray_dtshape_t *a, iarray_dtshape_t *b) {
    if (a->dtype != b->dtype) {
        return -1;
    }
    if (a->ndim != b->ndim) {
        return -1;
    }
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return -1;
        }
    }
    return 0;
}


INA_API(ina_rc_t) iarray_almost_equal_data(iarray_container_t *a, iarray_container_t *b, double tol) {
    if(a->dtshape->dtype != b->dtshape->dtype){
        return false;
    }
    if(a->catarr->size != b->catarr->size) {
        return false;
    }
    size_t size = a->catarr->size;

    uint8_t *buf_a = malloc(a->catarr->size * a->catarr->sc->typesize);
    caterva_to_buffer(a->catarr, buf_a);
    uint8_t *buf_b = malloc(b->catarr->size * b->catarr->sc->typesize);
    caterva_to_buffer(b->catarr, buf_b);

    if(a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        double *b_a = (double *)buf_a;
        double *b_b = (double *)buf_b;

        for (size_t i = 0; i < size; ++i) {
            double vdiff = fabs((b_a[i] - b_b[i]) / b_a[i]);
            if (vdiff > tol) {
                printf("%f, %f\n", b_a[i], b_b[i]);
                printf("Values differ in (%lu nelem) (diff: %f)\n", i, vdiff);
                free(buf_a);
                free(buf_b);
                return false;
            }
        }
        free(buf_a);
        free(buf_b);
        return true;
    }
    else if(a->dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        float *b_a = (float *)buf_a;
        float *b_b = (float *)buf_b;

        for (size_t i = 0; i < size; ++i) {
            double vdiff = fabs((double)(b_a[i] - b_b[i]) / b_a[i]);
            if (vdiff > tol) {
                printf("%f, %f\n", b_a[i], b_b[i]);
                printf("Values differ in (%lu nelem) (diff: %f)\n", i, vdiff);
                free(buf_a);
                free(buf_b);
                return false;
            }
        }
        free(buf_a);
        free(buf_b);
        return true;
    }
    printf("Data type is not supported");
    free(buf_a);
    free(buf_b);
    return false;
}


INA_API(ina_rc_t) iarray_gemm(iarray_container_t *a, iarray_container_t *b, iarray_container_t *c) {

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

    for (size_t m = 0; m < M / P; m++)
    {
        for (size_t n = 0; n < N / P; n++)
        {
            memset(c_block, 0, p_size);
            for (size_t k = 0; k < K / P; k++)
            {
                size_t a_i = (m * K / P + k);
                size_t b_i = (k * N / P + n);

                int a_tam = blosc2_schunk_decompress_chunk(a->catarr->sc, (int)a_i, a_block, p_size);
                int b_tam = blosc2_schunk_decompress_chunk(b->catarr->sc, (int)b_i, b_block, p_size);

                if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, P, P, P, 1.0, (double *)a_block, P, (double *)b_block, P, 1.0, (double *)c_block, P);
                }
                else if (dtype == IARRAY_DATA_TYPE_FLOAT) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, P, P, P, 1.0, (float *)a_block, P, (float *)b_block, P, 1.0, (float *)c_block, P);
                }
            }
            blosc2_schunk_append_buffer(c->catarr->sc, &c_block[0], p_size);
        }
    }
    free(a_block);
    free(b_block);
    free(c_block);
    return 0;
}

INA_API(ina_rc_t) iarray_gemv(iarray_container_t *a, iarray_container_t *b, iarray_container_t *c) {

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

    size_t a_i, b_i;

    for (size_t m = 0; m < M / P; m++)
    {
        memset(c_block, 0, p_vsize);
        for (size_t k = 0; k < K / P; k++)
        {
            a_i = (m * K / P + k);
            b_i = (k);

            int a_tam = blosc2_schunk_decompress_chunk(a->catarr->sc, (int)a_i, a_block, p_size);
            int b_tam = blosc2_schunk_decompress_chunk(b->catarr->sc, (int)b_i, b_block, p_vsize);

            if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
                cblas_dgemv(CblasRowMajor, CblasNoTrans, P, P, 1.0, (double *) a_block, P, (double *) b_block, 1, 1.0, (double *) c_block, 1);
            }
            else if (dtype == IARRAY_DATA_TYPE_FLOAT) {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, P, P, 1.0, (float *) a_block, P, (float *) b_block, 1, 1.0, (float *) c_block, 1);
            }
        }
        blosc2_schunk_append_buffer(c->catarr->sc, &c_block[0], p_vsize);
    }
    free(a_block);
    free(b_block);
    free(c_block);
    return 0;
}

void _update_itr_index(iarray_itr_t *itr) {

    caterva_array_t *catarr = itr->container->catarr;

    int ndim = catarr->ndim;

    uint64_t cont2 = itr->cont % catarr->csize;
    itr->index[ndim - 1] = cont2 % catarr->pshape[ndim-1];
    uint64_t inc = catarr->pshape[ndim - 1];

    for (int i = ndim - 2; i >= 0; --i) {
        itr->index[i] = cont2 % (inc * catarr->pshape[i]) / inc;
        inc *= catarr->pshape[i];
    }

    uint64_t nchunk = itr->cont / catarr->csize;

    uint64_t aux_nchunk[CATERVA_MAXDIM];

    aux_nchunk[ndim - 1] = catarr->eshape[ndim - 1] / catarr->pshape[ndim - 1];
    for (int k = ndim - 2; k >= 0; --k) {
        aux_nchunk[k] = aux_nchunk[k + 1] * (catarr->eshape[k] / catarr->pshape[k]);
    }
    for (int j = 0; j < ndim; ++j) {
        itr->index[j] += nchunk % aux_nchunk[j] / (aux_nchunk[j] / (catarr->eshape[j] / catarr->pshape[j])) * catarr->pshape[j];
    }

    if (itr->container->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        itr->pointer = (void *)&((double*)itr->part)[cont2];
    } else{
        itr->pointer = (void *)&((float*)itr->part)[cont2];
    }

    itr->nelem = 0;
    inc = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        itr->nelem += itr->index[i] * inc;
        inc *= itr->container->dtshape->shape[i];
    }
}


void _iarray_itr_init(iarray_itr_t *itr) {
    itr->cont = 0;
    itr->nelem = 0;
    memset(itr->part, 0, itr->container->catarr->csize * itr->container->catarr->sc->typesize);
    for (int i = 0; i < CATERVA_MAXDIM; ++i) {
        itr->index[i] = 0;
    }
    itr->pointer = &itr->part[0];
}

void _iarray_itr_next(iarray_itr_t *itr) {

    caterva_array_t *catarr = itr->container->catarr;
    int ndim = catarr->ndim;

    itr->cont += 1;

    _update_itr_index(itr);

    uint64_t aux_inc[CATERVA_MAXDIM];
    aux_inc[ndim - 1] = 1;
    for (int m = ndim - 2; m >= 0; --m) {
        aux_inc[m] = catarr->pshape[m + 1] * aux_inc[m + 1];
    }

    for (int l = ndim - 1; l >= 0; --l) {
        if (itr->index[l] >= catarr->shape[l]) {
            itr->cont += (catarr->eshape[l] - catarr->shape[l]) * aux_inc[l];
            _update_itr_index(itr);
        }
    }

    if (itr->cont % catarr->csize == 0) {
        blosc2_schunk_append_buffer(catarr->sc, itr->part, catarr->csize * catarr->sc->typesize);
        memset(itr->part, 0, catarr->csize * catarr->sc->typesize);
    }

    _update_itr_index(itr);
}


int _iarray_itr_finished(iarray_itr_t *itr) {
    return itr->cont >= itr->container->catarr->esize;
}


INA_API(ina_rc_t) iarray_itr_new(iarray_container_t *container, iarray_itr_t **itr) {
    *itr = (iarray_itr_t*)ina_mem_alloc(sizeof(iarray_itr_t));
    INA_RETURN_IF_NULL(itr);
    caterva_update_shape(container->catarr, *container->shape);
    (*itr)->container = container;
    (*itr)->part = (uint8_t *) ina_mem_alloc(container->catarr->csize * container->catarr->sc->typesize);

    (*itr)->index = (uint64_t *) ina_mem_alloc(CATERVA_MAXDIM * sizeof(uint64_t));

    (*itr)->init = _iarray_itr_init;
    (*itr)->next = _iarray_itr_next;
    (*itr)->finished = _iarray_itr_finished;
    return 0;
}

INA_API(ina_rc_t) iarray_itr_free(iarray_itr_t *itr) {
    ina_mem_free(itr->index);
    ina_mem_free(itr->part);
    ina_mem_free(itr);
    return 0;
}
