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
#include <contribs/tinyexpr/tinyexpr.h>
#include <minjugg.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#define _IARRAY_EXPR_VAR_MAX   (BLOSC2_PREFILTER_INPUTS_MAX)

typedef struct _iarray_tinyexpr_var_s {
    const char *var;
    iarray_container_t *c;
} _iarray_tinyexpr_var_t;

struct iarray_expression_s {
    iarray_context_t *ctx;
    ina_str_t expr;
    int32_t nchunks;
    int32_t blocksize;
    int32_t typesize;
    int32_t chunksize;
    int64_t nbytes;
    int nvars;
    int32_t max_out_len;
    te_expr *texpr;
    jug_expression_t *jug_expr;
    uint64_t jug_expr_func;
    iarray_temporary_t **temp_vars;
    iarray_container_t *out;
    _iarray_tinyexpr_var_t vars[_IARRAY_EXPR_VAR_MAX];
};

INA_API(ina_rc_t) iarray_expr_new(iarray_context_t *ctx, iarray_expression_t **e)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(e);
    *e = ina_mem_alloc(sizeof(iarray_expression_t));
    INA_RETURN_IF_NULL(e);
    (*e)->ctx = ctx;
    (*e)->nvars = 0;
    (*e)->max_out_len = 0;   // helper for leftovers
    ina_mem_set(&(*e)->vars, 0, sizeof(_iarray_tinyexpr_var_t)*_IARRAY_EXPR_VAR_MAX);
    jug_expression_new(&(*e)->jug_expr);
    return INA_SUCCESS;
}

INA_API(void) iarray_expr_free(iarray_context_t *ctx, iarray_expression_t **e)
{
    INA_VERIFY_FREE(e);
    if ((*e)->jug_expr != NULL) {
        jug_expression_free(&(*e)->jug_expr);
    }
    for (int nvar=0; nvar < (*e)->nvars; nvar++) {
        free((void*)((*e)->vars[nvar].var));
    }
    ina_mempool_reset(ctx->mp);  // FIXME: should be ina_mempool_free(), but it currently crashes
    ina_mempool_reset(ctx->mp_op);  // FIXME: ditto
    ina_mempool_reset(ctx->mp_tmp_out);  // FIXME: ditto
    INA_MEM_FREE_SAFE((*e)->temp_vars);
    ina_str_free((*e)->expr);
    INA_MEM_FREE_SAFE(*e);
}

INA_API(ina_rc_t) iarray_expr_bind(iarray_expression_t *e, const char *var, iarray_container_t *val)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(var);
    INA_VERIFY_NOT_NULL(val);

    e->vars[e->nvars].var = strdup(var);   // yes, we want a copy here!
    e->vars[e->nvars].c = val;
    e->nvars++;
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_expr_bind_scalar_float(iarray_expression_t *e, const char *var, float val)
//{
//  iarray_container_t *c = ina_mempool_dalloc(e->mp, sizeof(iarray_container_t));
//  c->dtshape = ina_mempool_dalloc(e->mp, sizeof(iarray_dtshape_t));
//  c->dtshape->ndim = 0;
//  c->dtshape->dims = NULL;
//  c->dtshape->dtype = IARRAY_DATA_TYPE_FLOAT;
//  c->scalar_value.f = val;
//  return INA_SUCCESS;
//}
{
    INA_UNUSED(e);
    INA_UNUSED(var);
    INA_UNUSED(val);
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}

INA_API(ina_rc_t) iarray_expr_bind_scalar_double(iarray_expression_t *e, const char *var, double val)
//{
//    iarray_container_t *c = ina_mempool_dalloc(e->ctx->mp, sizeof(iarray_container_t));
//    c->dtshape = ina_mempool_dalloc(e->ctx->mp, sizeof(iarray_dtshape_t));
//    c->dtshape->ndim = 0;
//    c->dtshape->dtype = IARRAY_DATA_TYPE_DOUBLE;
//    c->scalar_value.d = val;
//    e->vars[e->nvars].var = var;
//    e->vars[e->nvars].c = c;
//    e->nvars++;
//    return INA_SUCCESS;
//}
{
    INA_UNUSED(e);
    INA_UNUSED(var);
    INA_UNUSED(val);
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}


static ina_rc_t _iarray_expr_prepare(iarray_expression_t *e, int *nthreads_out)
{
    ina_rc_t rc;

    int nthreads = 1;

#if defined(_OPENMP)
    if (e->ctx->cfg->eval_flags == IARRAY_EXPR_EVAL_ITERBLOCK) {
        // Set a number of threads different from one in case the compiler supports OpemMP
        // This is not the case for the clang that comes with Mac OSX, but probably the newer
        // clang that come with later LLVM releases does have support for it.
        nthreads = e->ctx->cfg->max_num_threads;
        // The number of threads in config may get overridden by the OMP_NUM_THREADS variable
        char *envvar = getenv("OMP_NUM_THREADS");
        if (envvar != NULL) {
            long value;
            value = strtol(envvar, NULL, 10);
            if ((value != EINVAL) && (value >= 0)) {
                nthreads = (int)value;
            }
        }
    }
#endif

    e->temp_vars = ina_mem_alloc(nthreads * e->nvars * sizeof(iarray_temporary_t*)); // TODO: This should be freed?
    caterva_array_t *catarr = e->vars[0].c->catarr;

    e->typesize = catarr->ctx->cparams.typesize;
    int64_t size = 1;
    for (int i = 0; i < e->vars[0].c->dtshape->ndim; ++i) {
        size *= e->vars[0].c->dtshape->shape[i];
    }

    e->nbytes = size * e->typesize;
    if (catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        // Somewhat arbitrary values follows
        e->blocksize = 1024 * e->typesize;
        e->chunksize = 16 * e->blocksize;
    }
    else {
        blosc2_schunk *schunk = catarr->sc;
        if (e->ctx->cfg->eval_flags == IARRAY_EXPR_EVAL_ITERBLOCK) {
            uint8_t *chunk;
            bool needs_free;
            int retcode = blosc2_schunk_get_chunk(schunk, 0, &chunk, &needs_free);
            if (retcode < 0) {
                printf("Cannot retrieve the chunk in position %d\n", 0);
                if (chunk != NULL) {
                    free(chunk);
                }
                INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
            }

            size_t chunksize, cbytes, blocksize;
            blosc_cbuffer_sizes(chunk, &chunksize, &cbytes, &blocksize);
            if (needs_free) {
                free(chunk);
            }
            e->chunksize = (int32_t) chunksize;
            e->blocksize = (int32_t) blocksize;
        }
        else if (e->ctx->cfg->eval_flags == IARRAY_EXPR_EVAL_ITERCHUNK ||
                 e->ctx->cfg->eval_flags == IARRAY_EXPR_EVAL_ITERBLOSC) {
            e->chunksize = schunk->chunksize;
        }
        else {
            fprintf(stderr, "Flag %d is not supported\n", e->ctx->cfg->eval_flags);
            INA_FAIL_IF_ERROR(INA_ERROR(INA_ERR_NOT_SUPPORTED));
        }
    }

    e->nchunks = (int32_t)(e->nbytes / e->chunksize);
    if (e->nchunks * e->chunksize < e->nbytes) {
        e->nchunks += 1;
    }

    *nthreads_out = nthreads;
    return INA_SUCCESS;

fail:
    INA_MEM_FREE_SAFE(e->temp_vars);
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_expr_compile_udf(iarray_expression_t *e, int llvm_bc_len, const char *llvm_bc)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(llvm_bc);

    int nthreads;
    ina_rc_t rc = _iarray_expr_prepare(e, &nthreads);
    if (rc != INA_SUCCESS) {
        return rc;
    }

    INA_FAIL_IF_ERROR(
        jug_udf_compile(e->jug_expr, llvm_bc_len, llvm_bc, &e->jug_expr_func)
    );

    return INA_SUCCESS;

fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_expr_compile(iarray_expression_t *e, const char *expr)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(expr);

    e->expr = ina_str_new_fromcstr(expr);

    int nthreads;
    ina_rc_t rc = _iarray_expr_prepare(e, &nthreads);
    if (rc != INA_SUCCESS) {
        return rc;
    }

    // Create temporaries for initial variables.
    // We don't need the temporaries to be conformant with pshape; only the buffer
    // size needs to the same.
    iarray_dtshape_t dtshape_var = {0};  // initialize to 0s
    dtshape_var.ndim = 1;
    int32_t temp_var_dim0 = 0;
    if (e->ctx->cfg->eval_flags == IARRAY_EXPR_EVAL_ITERBLOCK) {
        temp_var_dim0 = e->blocksize / e->typesize;
    } else if (e->ctx->cfg->eval_flags == IARRAY_EXPR_EVAL_ITERCHUNK ||
               e->ctx->cfg->eval_flags == IARRAY_EXPR_EVAL_ITERBLOSC) {
        temp_var_dim0 = e->chunksize / e->typesize;
        e->blocksize = 0;
    } else {
        fprintf(stderr, "Flag %d is not supported\n", e->ctx->cfg->eval_flags);
        INA_FAIL_IF_ERROR(INA_ERROR(INA_ERR_NOT_SUPPORTED));
    }
    dtshape_var.shape[0] = temp_var_dim0;
    dtshape_var.dtype = e->vars[0].c->dtshape->dtype;

    te_variable *te_vars = ina_mempool_dalloc(e->ctx->mp, e->nvars * sizeof(te_variable));
    jug_te_variable *jug_vars = ina_mempool_dalloc(e->ctx->mp, e->nvars * sizeof(jug_te_variable));
    memset(jug_vars, 0, e->nvars * sizeof(jug_te_variable));
    for (int nvar = 0; nvar < e->nvars; nvar++) {
        te_vars[nvar].name = e->vars[nvar].var;
        te_vars[nvar].type = TE_VARIABLE;
        te_vars[nvar].context = NULL;
        te_vars[nvar].address = ina_mempool_dalloc(e->ctx->mp, nthreads * sizeof(void*));
        jug_vars[nvar].name = e->vars[nvar].var;
        // Allocate different buffers for each thread too
        for (int nthread = 0; nthread < nthreads; nthread++) {
            int ntvar = nthread * e->nvars + nvar;
            INA_FAIL_IF_ERROR(iarray_temporary_new(e, e->vars[nvar].c, &dtshape_var, &e->temp_vars[ntvar]));
            te_vars[nvar].address[nthread] = *(e->temp_vars + ntvar);
        }
    }

    int err = 0;
    e->texpr = te_compile(e, ina_str_cstr(e->expr), te_vars, e->nvars, &err);
    if (e->texpr == 0) {
        INA_FAIL_IF_ERROR(INA_ERROR(INA_ERR_NOT_COMPILED));
    }
    INA_FAIL_IF_ERROR(
        jug_expression_compile(e->jug_expr, ina_str_cstr(e->expr), e->nvars, jug_vars, &e->jug_expr_func)
    );
    return INA_SUCCESS;

fail:
    INA_MEM_FREE_SAFE(e->temp_vars);
    return ina_err_get_rc();
}

int prefilter_func(blosc2_prefilter_params *pparams)
{
    struct iarray_expression_s *e = pparams->user_data;

    int ninputs = pparams->ninputs;
    for (int i = 0; i < ninputs; i++) {
        e->temp_vars[i]->data = pparams->inputs[i];
    }

    // Eval the expression for this chunk
    e->max_out_len = pparams->out_size / pparams->out_typesize;  // so as to prevent operating beyond the limits
    const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
    memcpy(pparams->out, (uint8_t*)expr_out->data, pparams->out_size);

    // Uncomment the next line and comment out the above ones if one want to do benchmarks
    // compute_out((double*)(pparams->inputs[0]), (double*)(pparams->out), pparams->out_size / pparams->out_typesize);

    return 0;
}

ina_rc_t iarray_eval_cleanup(iarray_expression_t *e, int64_t nitems_written)
{
    ina_mempool_reset(e->ctx->mp);
    ina_mempool_reset(e->ctx->mp_op);
    ina_mempool_reset(e->ctx->mp_tmp_out);

    int64_t nitems_in_schunk = e->nbytes / e->typesize;
    if (nitems_written != nitems_in_schunk) {
        printf("nitems written is different from items in final container\n");
        return INA_ERROR(INA_ERR_NOT_COMPLETE);
    }

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_eval_iterchunk(iarray_expression_t *e, iarray_container_t *ret, int64_t *out_pshape)
{
    ina_rc_t rc;
    int nvars = e->nvars;
    int64_t nitems_written = 0;

    // Create and initialize an iterator per variable
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_t *ctx = NULL;
    iarray_context_new(&cfg, &ctx);
    iarray_iter_read_block_t **iter_var = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_t));
    iarray_iter_read_block_value_t *iter_value = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_value_t));

    for (int nvar = 0; nvar < nvars; nvar++) {
        iarray_container_t *var = e->vars[nvar].c;
        if (INA_FAILED(iarray_iter_read_block_new(ctx, &iter_var[nvar], var, out_pshape, &iter_value[nvar], false))) {
            goto fail_iterchunk;
        }
    }

    // Write iterator for output
    iarray_iter_write_block_t *iter_out;
    iarray_iter_write_block_value_t out_value;
    if (INA_FAILED(iarray_iter_write_block_new(ctx, &iter_out, ret, out_pshape, &out_value, false))) {
        goto fail_iterchunk;
    }

    // Evaluate the expression for all the chunks in variables
    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter_out))) {
        if (INA_FAILED(iarray_iter_write_block_next(iter_out, NULL, 0))) {
            goto fail_iterchunk;
        }
        int32_t out_items = (int32_t)(iter_out->cur_block_size);

        // Decompress chunks in variables into temporaries
        for (int nvar = 0; nvar < nvars; nvar++) {
            if INA_FAILED(iarray_iter_read_block_next(iter_var[nvar], NULL, 0)) {
                goto fail_iterchunk;
            }
            e->temp_vars[nvar]->data = iter_value[nvar].block_pointer;
        }

        // Eval the expression for this chunk
        e->max_out_len = out_items;  // so as to prevent operating beyond the limits
        const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
        memcpy((char*)out_value.block_pointer, (uint8_t*)expr_out->data, out_items * e->typesize);
        nitems_written += out_items;
        ina_mempool_reset(e->ctx->mp_tmp_out);
    }

    if (ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0)) {
        goto fail_iterchunk;
    }

    for (int nvar = 0; nvar < nvars; nvar++) {
        iarray_iter_read_block_free(&(iter_var[nvar]));
    }
    iarray_iter_write_block_free(&iter_out);
    INA_MEM_FREE_SAFE(iter_var);
    INA_MEM_FREE_SAFE(iter_value);
    iarray_context_free(&ctx);

    rc = iarray_eval_cleanup(e, nitems_written);
    return rc;

    fail_iterchunk:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_eval_iterblosc(iarray_expression_t *e, iarray_container_t *ret, int64_t *out_pshape)
{
    ina_rc_t rc;
    int64_t nitems_written = 0;
    int nvars = e->nvars;

    if (ret->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        fprintf(stderr, "ITERBLOSC eval can't be used with a plainbuffer output container.\n");
        INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
        goto fail_iterblosc;
    }

    // Setup a new cparams with a prefilter
    blosc2_cparams *cparams = malloc(sizeof(blosc2_cparams));
    memcpy(cparams, ret->cparams, sizeof(blosc2_cparams));
    cparams->prefilter = (blosc2_prefilter_fn)e->jug_expr_func;
    blosc2_prefilter_params pparams = {0};
    pparams.ninputs = nvars;
    // TODO: add the out_value structure to the user_data also?
    pparams.user_data = (void*)e;
    cparams->pparams = &pparams;

    // Create and initialize an iterator per variable
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_t *ctx = NULL;
    iarray_context_new(&cfg, &ctx);
    iarray_iter_read_block_t **iter_var = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_t));
    iarray_iter_read_block_value_t *iter_value = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_value_t));
    for (int nvar = 0; nvar < nvars; nvar++) {
        iarray_container_t *var = e->vars[nvar].c;
        if (INA_FAILED(iarray_iter_read_block_new(ctx, &iter_var[nvar], var, out_pshape, &iter_value[nvar], false))) {
            goto fail_iterblosc;
        }
        pparams.input_typesizes[nvar] = var->catarr->sc->typesize;
    }

    // Write iterator for output
    iarray_iter_write_block_t *iter_out;
    iarray_iter_write_block_value_t out_value;
    int32_t external_buffer_size = ret->catarr->psize * ret->catarr->sc->typesize + BLOSC_MAX_OVERHEAD;
    void *external_buffer;  // to inform the iterator that we are passing an external buffer
    if (INA_FAILED(iarray_iter_write_block_new(ctx, &iter_out, ret, out_pshape, &out_value, true))) {
        goto fail_iterblosc;
    }

    // Evaluate the expression for all the chunks in variables
    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter_out))) {
        external_buffer = malloc(external_buffer_size);

        if (INA_FAILED(iarray_iter_write_block_next(iter_out, external_buffer, external_buffer_size))) {
            goto fail_iterblosc;
        }

        // Update the external buffer with freshly allocated memory
        int64_t out_items = iter_out->cur_block_size;

        // Decompress chunks in variables into temporaries
        for (int nvar = 0; nvar < nvars; nvar++) {
            if (INA_FAILED(iarray_iter_read_block_next(iter_var[nvar], NULL, 0))) {
                goto fail_iterblosc;
            }
            e->temp_vars[nvar]->data = iter_value[nvar].block_pointer;
            pparams.inputs[nvar] = iter_value[nvar].block_pointer;
        }

        // Eval the expression for this chunk
        blosc2_context *cctx = blosc2_create_cctx(*cparams);
        int csize = blosc2_compress_ctx(cctx, out_items * e->typesize,
                                        NULL, out_value.block_pointer,
                                        out_items * e->typesize + BLOSC_MAX_OVERHEAD);
        if (csize <= 0) {
            // Retry with clevel == 0 (should never fail)
            blosc2_free_ctx(cctx);
            cparams->clevel = 0;
            cctx = blosc2_create_cctx(*cparams);
            csize = blosc2_compress_ctx(cctx, out_items * e->typesize,
                                        NULL, out_value.block_pointer,
                                        out_items * e->typesize + BLOSC_MAX_OVERHEAD);
        }
        blosc2_free_ctx(cctx);
        if (csize <= 0) {
            INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
            goto fail_iterblosc;
        }

        if (out_items != ret->catarr->psize) {
            // Not a complete chunk.  Decompress and append it as a regular buffer.
            uint8_t *temp = malloc(csize);
            memcpy(temp, out_value.block_pointer, csize);
            int nbytes = blosc_decompress(temp, out_value.block_pointer, out_items * e->typesize);
            free(temp);
            if (nbytes <= 0) {
                INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
                goto fail_iterblosc;
            }
            iter_out->compressed_chunk_buffer = false;
        }
        else {
            iter_out->compressed_chunk_buffer = true;
        }

        nitems_written += out_items;
        ina_mempool_reset(e->ctx->mp_tmp_out);
    }

    if (ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0)) {
        goto fail_iterblosc;
    }

    for (int nvar = 0; nvar < nvars; nvar++) {
        iarray_iter_read_block_free(&iter_var[nvar]);
    }
    iarray_iter_write_block_free(&iter_out);
    INA_MEM_FREE_SAFE(iter_var);
    INA_MEM_FREE_SAFE(iter_value);
    iarray_context_free(&ctx);

    rc = iarray_eval_cleanup(e, nitems_written);
    return rc;

    fail_iterblosc:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_eval_iterblock(iarray_expression_t *e, iarray_container_t *ret, int64_t *out_pshape)
{
    ina_rc_t rc;
    int64_t nitems_written = 0;
    int nvars = e->nvars;

    // This version of the evaluation engine works by using a chunk iterator and use OpenMP
    // for performing the computations.  The OpenMP loop split the chunk into smaller *blocks* that
    // are passed the tinyexpr evaluator.
    // In the future we may want to get rid of the cost of creating/destroying the thread per every chunk.
    // One possibility is to use pthreads, but this would require more complex code, so we need to discuss it more.
    int32_t blocksize = e->blocksize;
    int32_t chunksize = e->chunksize;

    // Create and initialize an iterator per each variable
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_t *ctx = NULL;
    iarray_context_new(&cfg, &ctx);
    iarray_iter_read_block_t **iter_var = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_t));
    iarray_iter_read_block_value_t *iter_value = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_value_t));
    for (int nvar = 0; nvar < nvars; nvar++) {
        iarray_container_t *var = e->vars[nvar].c;
        if (INA_FAILED(iarray_iter_read_block_new(ctx, &iter_var[nvar], var, out_pshape, &iter_value[nvar], false))) {
            goto fail_iterblock;
        }
    }

    // Write iterator for output
    iarray_iter_write_block_t *iter_out;
    iarray_iter_write_block_value_t out_value;
    if (INA_FAILED(iarray_iter_write_block_new(ctx, &iter_out, ret, out_pshape, &out_value, false))) {
        goto fail_iterblock;
    }

    // Evaluate the expression for all the chunks in variables
    int8_t *outbuf = ina_mem_alloc((size_t)chunksize);

    int32_t nblocks;
    int32_t out_items;

    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter_out))) {
        if (INA_FAILED(iarray_iter_write_block_next(iter_out, NULL, 0))) {
            goto fail_iterblock;
        }

        for (int nvar = 0; nvar < nvars; nvar++) {
            if (INA_FAILED(iarray_iter_read_block_next(iter_var[nvar], NULL, 0))) {
                goto fail_iterblock;
            }
        }

        out_items = (int32_t)(iter_out->cur_block_size);  // TODO: add a protection against cur_block_size > 2**31
        nblocks = out_items * e->typesize / blocksize;

        int nthread = 0;

#if defined(_OPENMP)
        omp_set_num_threads(e->ctx->cfg->max_num_threads);
#pragma omp parallel for
#endif
        for (int nblock = 0; nblock < nblocks; nblock++) {
#if defined(_OPENMP)
            nthread = omp_get_thread_num();
#endif
            for (int nvar = 0; nvar < nvars; nvar++) {

                int ntvar = nthread * e->nvars + nvar;
                e->temp_vars[ntvar]->data = (char *) iter_value[nvar].block_pointer + nblock * blocksize;
            }
            e->max_out_len = blocksize / e->typesize;  // so as to prevent operating beyond the limits
            const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
            memcpy((char*)out_value.block_pointer + nblock * blocksize, (uint8_t*)expr_out->data, blocksize);
        }

        // Do a possible last evaluation with the leftovers
        int32_t leftover = out_items * e->typesize - nblocks * blocksize;
        if (leftover > 0) {
            for (int nvar = 0; nvar < nvars; nvar++) {
                e->temp_vars[nvar]->data = (char *) iter_value[nvar].block_pointer + nblocks * blocksize;
            }
            e->max_out_len = leftover / e->typesize;  // so as to prevent operating beyond the leftover
            const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
            e->max_out_len = 0;
            memcpy((char*)out_value.block_pointer + nblocks * blocksize, (uint8_t*)expr_out->data, leftover);
        }

        // Write the resulting chunk in output
        nitems_written += out_items;
        ina_mempool_reset(e->ctx->mp_tmp_out);
    }

    if (ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0)) {
        goto fail_iterblock;
    }

    for (int nvar = 0; nvar < nvars; nvar++) {
        iarray_iter_read_block_free(&iter_var[nvar]);
    }
    iarray_iter_write_block_free(&iter_out);
    INA_MEM_FREE_SAFE(iter_var);
    INA_MEM_FREE_SAFE(iter_value);
    INA_MEM_FREE_SAFE(outbuf);
    iarray_context_free(&ctx);
    rc = iarray_eval_cleanup(e, nitems_written);
    return rc;

    fail_iterblock:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_eval(iarray_expression_t *e, iarray_container_t *ret)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(ret);

    int64_t *out_pshape;
    if (ret->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        // Compute a decent pshape for a plainbuffer output
        out_pshape = (int64_t *) malloc(ret->dtshape->ndim * sizeof(int64_t));
        int32_t nelems = e->chunksize / e->typesize;
        for (int i = ret->dtshape->ndim - 1; i >= 0; i--) {
            int32_t pshapei = nelems < ret->dtshape->shape[i] ? nelems : ret->dtshape->shape[i];
            out_pshape[i] = pshapei;
            nelems = nelems / pshapei;
        }
    } else {
        out_pshape = ret->dtshape->pshape;
    }

    switch (e->ctx->cfg->eval_flags) {
        case IARRAY_EXPR_EVAL_ITERCHUNK:
            return iarray_eval_iterchunk(e, ret, out_pshape);
        case IARRAY_EXPR_EVAL_ITERBLOCK:
            return iarray_eval_iterblock(e, ret, out_pshape);
        case IARRAY_EXPR_EVAL_ITERBLOSC:
            return iarray_eval_iterblosc(e, ret, out_pshape);
        default:
            fprintf(stderr, "Invalid eval method.\n");
            return IARRAY_ERR_INVALID_EVAL_METHOD;
    }
}

ina_rc_t iarray_shape_size(iarray_dtshape_t *dtshape, size_t *size)
{
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(size);

    size_t type_size = 0;
    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            type_size = sizeof(double);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            type_size = sizeof(float);
            break;
        default:
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }
    for (int i = 0; i < dtshape->ndim; ++i) {
        *size += dtshape->shape[i] * type_size;
    }
    return INA_SUCCESS;
    fail:
    return ina_err_get_rc();
}

ina_rc_t iarray_temporary_new(iarray_expression_t *expr, iarray_container_t *c, iarray_dtshape_t *dtshape,
        iarray_temporary_t **temp)
{
    INA_VERIFY_NOT_NULL(expr);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(temp);

    // When c == NULL means a temporary for output, which should go to its own memory pool for being
    // able to reset it during each block/chunk evaluation
    ina_mempool_t *mempool = (c != NULL) ? expr->ctx->mp : expr->ctx->mp_tmp_out;
    *temp = ina_mempool_dalloc(mempool, sizeof(iarray_temporary_t));
    (*temp)->dtshape = ina_mempool_dalloc(mempool, sizeof(iarray_dtshape_t));
    ina_mem_cpy((*temp)->dtshape, dtshape, sizeof(iarray_dtshape_t));
    size_t typesize = dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ? 8 : 4;
    size_t size = expr->max_out_len * typesize;
    (*temp)->size = size;
    if (c != NULL) {
        // FIXME: support float values too
        ina_mem_cpy(&(*temp)->scalar_value, &c->scalar_value, sizeof(double));
    }
    if (size > 0) {
        (*temp)->data = ina_mempool_dalloc(mempool, size);
    }

    return INA_SUCCESS;
}

iarray_temporary_t* _iarray_func(iarray_expression_t *expr, iarray_temporary_t *operand1,
                                 iarray_temporary_t *operand2, iarray_functype_t func)
{
    if (expr == NULL) {
        goto fail;
    }
    if (operand1 == NULL) {
        goto fail;
    }
    if (operand2 != NULL && (
        (operand1->dtshape->ndim != operand2->dtshape->ndim) ||
        (operand1->size != operand2->size))) {
        printf("The 2 operands do not match dims or sizes");
        goto fail;
    }

    iarray_dtshape_t dtshape = {0};  // initialize to 0s
    iarray_temporary_t *out;
    bool scalar = true;
    if (operand1->dtshape->ndim > 0) {
        scalar = false;
        dtshape.dtype = operand1->dtshape->dtype;
        dtshape.ndim = operand1->dtshape->ndim;
        memcpy(dtshape.shape, operand1->dtshape->shape, sizeof(int64_t) * dtshape.ndim);
    }

    // Creating the temporary means interacting with the INA memory allocator, which is not thread-safe.
    // We should investigate on how to overcome this syncronization point (if possible at all).
    ina_rc_t err;
#if defined(_OPENMP)
#pragma omp critical
#endif

    err = iarray_temporary_new(expr, NULL, &dtshape, &out);
    INA_FAIL_IF_ERROR(err);

    switch (dtshape.dtype) {
        case IARRAY_DATA_TYPE_DOUBLE: {
            double *operand1_pointer;
            double *operand2_pointer = NULL;
            double *out_pointer;
            int32_t len;
            if (scalar) {
                len = 1;
                operand1_pointer = &operand1->scalar_value.d;
                out_pointer = &out->scalar_value.d;
            }
            else {
                len = expr->max_out_len == 0 ? (int32_t)(out->size / sizeof(double)) : expr->max_out_len;
                operand1_pointer = operand1->data;
                if (operand2 != NULL) operand2_pointer = operand2->data;
                out_pointer = out->data;
            }
            switch (func) {
                case IARRAY_FUNC_ABS:
                    vdAbs(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ACOS:
                    vdAcos(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ASIN:
                    vdAsin(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ATAN:
                    vdAtan(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ATAN2:
                    vdAtan2(len, operand1_pointer, operand2_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_CEIL:
                    vdCeil(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_COS:
                    vdCos(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_COSH:
                    vdCosh(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_EXP:
                    vdExp(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_FLOOR:
                    vdFloor(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_LN:
                    vdLn(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_LOG10:
                    vdLog10(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_NEGATE:
                    for (int i = 0; i < len; i++) {
                        out_pointer[i] = -operand1_pointer[i];
                    }
                    break;
                case IARRAY_FUNC_POW:
                    vdPow(len, operand1_pointer, operand2_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_SIN:
                    vdSin(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_SINH:
                    vdSinh(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_SQRT:
                    vdSqrt(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_TAN:
                    vdTan(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_TANH:
                    vdTanh(len, operand1_pointer, out_pointer);
                    break;
                default:
                    printf("Operation not supported yet");
                    goto fail;
            }
        }
        break;
        case IARRAY_DATA_TYPE_FLOAT: {
            int32_t len;
            float *operand1_pointer;
            float *operand2_pointer = NULL;
            float *out_pointer;
            if (scalar) {
                len = 1;
                operand1_pointer = &operand1->scalar_value.f;
                out_pointer = &out->scalar_value.f;
            }
            else {
                len = expr->max_out_len == 0 ? (int32_t)(out->size / sizeof(float)) : expr->max_out_len;
                operand1_pointer = operand1->data;
                if (operand2 != NULL) operand2_pointer = operand2->data;
                out_pointer = out->data;
            }
            switch (func) {
                case IARRAY_FUNC_ABS:
                    vsAbs(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ACOS:
                    vsAcos(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ASIN:
                    vsAsin(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ATAN:
                    vsAtan(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ATAN2:
                    vsAtan2(len, operand1_pointer, operand2_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_CEIL:
                    vsCeil(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_COS:
                    vsCos(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_COSH:
                    vsCosh(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_EXP:
                    vsExp(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_FLOOR:
                    vsFloor(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_LN:
                    vsLn(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_LOG10:
                    vsLog10(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_NEGATE:
                    for (int i = 0; i < len; i++) {
                        out_pointer[i] = -operand1_pointer[i];
                    }
                    break;
                case IARRAY_FUNC_POW:
                    vsPow(len, operand1_pointer, operand2_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_SIN:
                    vsSin(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_SINH:
                    vsSinh(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_SQRT:
                    vsSqrt(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_TAN:
                    vsTan(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_TANH:
                    vsTanh(len, operand1_pointer, out_pointer);
                    break;
                default:
                    printf("Operation not supported yet");
                    goto fail;
            }
        }
        break;
        default:
            printf("Operation not supported yet");
            goto fail;
    }

    return out;

    fail:
    // TODO: Free temporary
    return NULL;
}

static iarray_temporary_t* _iarray_op(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_optype_t op)
{
    if (expr == NULL) {
        goto fail;
    }
    if (lhs == NULL) {
        goto fail;
    }

    if (rhs == NULL) {
        goto fail;
    }

    bool scalar = false;
    bool scalar_vector = false;
    bool vector_vector = false;

    iarray_dtshape_t dtshape = {0};  // initialize to 0s
    iarray_temporary_t *scalar_tmp = NULL;
    iarray_temporary_t *scalar_lhs = NULL;
    iarray_temporary_t *out;

    if (lhs->dtshape->ndim == 0 && rhs->dtshape->ndim == 0) {   /* scalar-scalar */
        dtshape.dtype = rhs->dtshape->dtype;
        dtshape.ndim = rhs->dtshape->ndim;
        memcpy(dtshape.shape, rhs->dtshape->shape, sizeof(int64_t) * dtshape.ndim);
        scalar = true;
    }
    else if (lhs->dtshape->ndim == 0 || rhs->dtshape->ndim == 0) {   /* scalar-vector */
        if (lhs->dtshape->ndim == 0) {
            dtshape.dtype = rhs->dtshape->dtype;
            dtshape.ndim = rhs->dtshape->ndim;
            ina_mem_cpy(dtshape.shape, rhs->dtshape->shape, sizeof(int64_t) * dtshape.ndim);
            scalar_tmp = lhs;
            scalar_lhs = rhs;
        }
        else {
            dtshape.dtype = lhs->dtshape->dtype;
            dtshape.ndim = lhs->dtshape->ndim;
            ina_mem_cpy(dtshape.shape, lhs->dtshape->shape, sizeof(int64_t) * dtshape.ndim);
            scalar_tmp = rhs;
            scalar_lhs = lhs;
        }
        scalar_vector = true;
    }
    else if (lhs->dtshape->ndim == 1 && rhs->dtshape->ndim == 1) { /* vector-vector */
        dtshape.dtype = lhs->dtshape->dtype;
        dtshape.ndim = lhs->dtshape->ndim;
        ina_mem_cpy(dtshape.shape, lhs->dtshape->shape, sizeof(int64_t) * lhs->dtshape->ndim);
        vector_vector = true;
    }
    else {
        /* FIXME: matrix/vector and matrix/matrix addition */
    }

    // Creating the temporary means interacting with the INA memory allocator, which is not thread-safe.
    // We should investigate on how to overcome this syncronization point (if possible at all).

    ina_rc_t err;
#if defined(_OPENMP)
#pragma omp critical
#endif

    err = iarray_temporary_new(expr, NULL, &dtshape, &out);
    INA_FAIL_IF_ERROR(err);

    switch (dtshape.dtype) {
        case IARRAY_DATA_TYPE_DOUBLE: {
            int32_t len = expr->max_out_len == 0 ? (int32_t)(out->size / sizeof(double)) : expr->max_out_len;
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
                    goto fail;
                }
            }
            else if (scalar_vector) {
                double dscalar = scalar_tmp->scalar_value.d;
                double *odata = (double*)out->data;
                double *ldata = (double*)scalar_lhs->data;
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] + dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] - dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] * dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] / dscalar;
                    }
                    break;
                default:
                    printf("Operation not supported yet");
                    goto fail;
                }
            }
            else if (vector_vector) {
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] + ((double*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] - ((double*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] * ((double*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] / ((double*)rhs->data)[i];
                    }
                    break;
                default:
                    printf("Operation not supported yet");
                    goto fail;
                }
            }
            else {
                printf("DTshape combination not supported yet\n");
                goto fail;
            }
        }
        break;
        case IARRAY_DATA_TYPE_FLOAT: {
            int32_t len = expr->max_out_len == 0 ? (int32_t)(out->size / sizeof(float)) : expr->max_out_len;
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
                    goto fail;
                }
            }
            else if (scalar_vector) {
                float dscalar = (float)scalar_tmp->scalar_value.d;
                float *odata = (float*)out->data;
                float *ldata = (float*)scalar_lhs->data;
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] + dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] - dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] * dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] / dscalar;
                    }
                    break;
                default:
                    printf("Operation not supported yet");
                    goto fail;
                }
            }
            else if (vector_vector) {
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] + ((float*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] - ((float*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] * ((float*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] / ((float*)rhs->data)[i];
                    }
                    break;
                default:
                    printf("Operation not supported yet");
                    goto fail;
                }
            }
            else {
                printf("DTshape combination not supported yet\n");
                goto fail;
            }
        }
        break;
        default:  // switch (dtshape.dtype)
            printf("data type not supported yet\n");
            goto fail;
    }

    return out;

    fail:
        // TODO: Free temporary
        return NULL;
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
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(mp);
    *mp = e->ctx->mp;
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_expr_get_nthreads(iarray_expression_t *e, int *nthreads)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(nthreads);
    *nthreads = e->ctx->cfg->max_num_threads;
    return INA_SUCCESS;
}
