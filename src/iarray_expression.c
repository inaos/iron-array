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
#ifndef __clang__
#include <omp.h>
#endif

#define _IARRAY_EXPR_VAR_MAX      (128)

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
    int nvars;
    int max_out_len;
    te_expr *texpr;
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
    return INA_SUCCESS;
}

INA_API(void) iarray_expr_free(iarray_context_t *ctx, iarray_expression_t **e)
{
    INA_ASSERT_NOT_NULL(ctx);
    INA_VERIFY_FREE(e);
    for (int nvar=0; nvar < (*e)->nvars; nvar++) {
        free((void*)((*e)->vars[nvar].var));
    }
    ina_mempool_reset(ctx->mp);  // FIXME: should be ina_mempool_free(), but it currently crashes
    ina_mempool_reset(ctx->mp_op);  // FIXME: ditto
    ina_mempool_reset(ctx->mp_tmp_out);  // FIXME: ditto
    INA_MEM_FREE_SAFE((*e)->temp_vars);
    INA_MEM_FREE_SAFE(*e);
}

INA_API(ina_rc_t) iarray_expr_bind(iarray_expression_t *e, const char *var, iarray_container_t *val)
{
    if (val->dtshape->ndim > 2) {
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }
    e->vars[e->nvars].var = strdup(var);   // yes, we want a copy here!
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

//INA_API(ina_rc_t) iarray_expr_bind_scalar_double(iarray_expression_t *e, const char *var, double val)
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

INA_API(ina_rc_t) iarray_expr_compile(iarray_expression_t *e, const char *expr)
{
    int nthreads = 1;

    if (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERCHUNKPARA) {
        // Set a number of threads different from one in case the compiler supports OpemMP
        // This is not the case for the clang that comes with Mac OSX, but probably the newer
        // clang that come with later LLVM releases does have support for it.
#ifndef __clang__
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
#endif
    }

    e->expr = ina_str_new_fromcstr(expr);
    e->temp_vars = ina_mem_alloc(nthreads * e->nvars * sizeof(iarray_temporary_t*));
    te_variable *te_vars = ina_mempool_dalloc(e->ctx->mp, e->nvars * sizeof(te_variable));
    caterva_array_t *catarr = e->vars[0].c->catarr;
    blosc2_schunk *schunk = catarr->sc;
    int dim0 = 0;
    if ((e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_BLOCK) ||
        (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERCHUNKPARA) ||
        (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERBLOCK)) {
        int32_t typesize = schunk->typesize;
        int32_t nchunks = schunk->nchunks;
        uint8_t *chunk;
        bool needs_free;
        int retcode = blosc2_schunk_get_chunk(schunk, 0, &chunk, &needs_free);
        if (retcode < 0) {
            printf("Cannot retrieve the chunk in position %d\n", 0);
            return INA_ERR_FAILED;
        }
        size_t chunksize, cbytes, blocksize;
        blosc_cbuffer_sizes(chunk, &chunksize, &cbytes, &blocksize);
        if (needs_free) {
            free(chunk);
        }
        dim0 = (int)blocksize / typesize;
        e->nchunks = nchunks;
        e->chunksize = (int32_t)chunksize;
        e->blocksize = (int32_t)blocksize;
        e->typesize = (int32_t)typesize;
    }
    else if ((e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_CHUNK) ||
             (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERCHUNK)) {
        dim0 = schunk->chunksize / schunk->typesize;
        e->nchunks = schunk->nchunks;
        e->chunksize = schunk->chunksize;
        e->typesize = schunk->typesize;
    }
    else {
        fprintf(stderr, "Flag %d is not supported\n", e->ctx->cfg->eval_flags);
        return INA_ERR_NOT_SUPPORTED;
    }

    // Create temporaries for initial variables
    // TODO: make this more general and accept multidimensional containers
    iarray_dtshape_t dtshape_var = {0};  // initialize to 0s
    dtshape_var.ndim = 1;
    dtshape_var.shape[0] = dim0;
    dtshape_var.dtype = e->vars[0].c->dtshape->dtype;
    for (int nvar = 0; nvar < e->nvars; nvar++) {
        te_vars[nvar].name = e->vars[nvar].var;
        te_vars[nvar].type = TE_VARIABLE;
        te_vars[nvar].context = NULL;
        te_vars[nvar].address = ina_mem_alloc(nthreads * sizeof(void*));
        // Allocate different buffers for each thread too
        for (int nthread = 0; nthread < nthreads; nthread++) {
            int ntvar = nthread * e->nvars + nvar;
            iarray_temporary_new(e, e->vars[nvar].c, &dtshape_var, &e->temp_vars[ntvar]);
            te_vars[nvar].address[nthread] = *(e->temp_vars + ntvar);
        }
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
    int64_t nitems_in_schunk = schunk0->nbytes / e->typesize;
    int64_t nitems_in_chunk = e->chunksize / e->typesize;
    int nvars = e->nvars;
    caterva_dims_t shape = caterva_new_dims(e->vars[0].c->dtshape->shape, e->vars[0].c->dtshape->ndim);
    caterva_update_shape(ret->catarr, &shape);
    caterva_array_t out = *ret->catarr;

    if (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_BLOCK) {
        int8_t *outbuf = ina_mem_alloc((size_t)e->chunksize);
        int32_t nitems_in_block = e->blocksize / e->typesize;
        uint8_t **var_chunks = ina_mem_alloc(nvars * sizeof(uint8_t*));
        bool *var_needs_free = ina_mem_alloc(nvars * sizeof(bool));
        for (int nchunk = 0; nchunk < e->nchunks; nchunk++) {
            int32_t chunksize = (int32_t)((nchunk < e->nchunks - 1) ? e->chunksize : schunk0->nbytes - nchunk * e->chunksize);
            int32_t nblocks_in_chunk = chunksize / e->blocksize;
            int32_t corrected_blocksize = e->blocksize;
            int32_t corrected_nitems = nitems_in_block;
            if (nblocks_in_chunk * e->blocksize < e->chunksize) {
                nitems_in_chunk = chunksize / e->typesize;
                nblocks_in_chunk += 1;
            }
            // Allocate a buffer for every chunk (specially useful for reading on-disk variables)
            for (int nvar = 0; nvar < nvars; nvar++) {
                blosc2_schunk *schunk = e->vars[nvar].c->catarr->sc;
                int retcode = blosc2_schunk_get_chunk(schunk, nchunk, &var_chunks[nvar], &var_needs_free[nvar]);
                if (retcode < 0) {
                    printf("Cannot retrieve the chunk in position %d\n", nchunk);
                    return INA_ERR_FAILED;
                }
            }
            for (int32_t nblock = 0; nblock < nblocks_in_chunk; nblock++) {
                if ((nblock + 1 == nblocks_in_chunk) && (nblock + 1) * e->blocksize > chunksize) {
                    corrected_blocksize = chunksize - nblock * e->blocksize;
                    corrected_nitems = (int)corrected_blocksize / e->typesize;
                }
                // Decompress blocks in variables into temporaries
                for (int nvar = 0; nvar < nvars; nvar++) {
                    int dsize = blosc_getitem(var_chunks[nvar], (int)(nblock * nitems_in_block),
                                              (int)corrected_nitems, e->temp_vars[nvar]->data);
                    if (dsize < 0) {
                        printf("Decompression error.  Error code: %d\n", dsize);
                        return INA_ERR_FAILED;
                    }
                }
                // Evaluate the expression for this block
                const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
                ina_mem_cpy(outbuf + nblock * e->blocksize, expr_out->data, (size_t)corrected_blocksize);
                ina_mempool_reset(e->ctx->mp_tmp_out);
            }
            blosc2_schunk_append_buffer(out.sc, outbuf, (size_t)nitems_in_chunk * e->typesize);
            for (int nvar = 0; nvar < nvars; nvar++) {
                if (var_needs_free[nvar]) {
                    //ina_mem_free(var_chunks[nvar]);  // this raises an error (bug in the ina library?)
                    free(var_chunks[nvar]);
                }
            }
        }
        ina_mem_free(var_chunks);
        ina_mem_free(var_needs_free);
        ina_mem_free(outbuf);
    }
    else if (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERBLOCK) {
        // TODO: refine this and choose the nitems_in_block that works 'best' for all the variables
        int32_t chunksize = e->chunksize;
        int32_t blocksize = e->blocksize;
        int8_t *outbuf = ina_mem_alloc((size_t)chunksize);
        int64_t nitems_in_block = blocksize / e->typesize;

        // Create and initialize an iterator per variable
        iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
        iarray_context_t *ctx = NULL;
        iarray_context_new(&cfg, &ctx);
        iarray_iter_read_block_t **iter_var = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_t));
        iarray_iter_read_block_value_t *iter_value = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_value_t));

        for (int nvar = 0; nvar < nvars; nvar++) {
            iarray_container_t *var = e->vars[nvar].c;
            iarray_iter_read_block_new(ctx, &iter_var[nvar], var, &nitems_in_block, &iter_value[nvar]);
        }

        // Evaluate the expression for all the chunks in variables
        int64_t nitems_written = 0;
        int32_t nblocks_to_write = 0;
        int32_t leftover = 0;
        bool write_chunk = false;
        while (iarray_iter_read_block_has_next(iter_var[0])) {

            // Decompress blocks in variables into temporaries
            for (int nvar = 0; nvar < nvars; nvar++) {
                iarray_iter_read_block_next(iter_var[nvar]);
                e->temp_vars[nvar]->data = iter_value[nvar].pointer;
            }

            // Eval the expression for this block
            const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
            nblocks_to_write += 1;

            int32_t corrected_blocksize = blocksize;
            if (nblocks_to_write * blocksize + leftover >= chunksize) {
                corrected_blocksize = chunksize - ((nblocks_to_write - 1) * blocksize + leftover);
                write_chunk = true;
            }
            memcpy(outbuf + (nblocks_to_write - 1) * blocksize + leftover, (uint8_t*)expr_out->data, corrected_blocksize);
            ina_mempool_reset(e->ctx->mp_tmp_out);

            if (write_chunk) {
                blosc2_schunk_append_buffer(out.sc, outbuf, (size_t)chunksize);
                nitems_written += nitems_in_chunk;
                nblocks_to_write = 0;
                write_chunk = false;
                leftover = blocksize - corrected_blocksize;
                // Copy the leftover at the beginning of the chunk for the next iteration
                memcpy(outbuf, (uint8_t*)expr_out->data + corrected_blocksize, leftover);
            }
        }

        // Write the leftovers of the expression in output
        int64_t items_left = nitems_in_schunk - nitems_written;
        if (items_left > 0) {
            blosc2_schunk_append_buffer(out.sc, outbuf, (size_t)items_left * e->typesize);
            // nitems_written += items_left;  // commented out to avoid an 'unused variable' warning
        }
        assert(nitems_written == nitems_in_schunk);

        for (int nvar = 0; nvar < nvars; nvar++) {
            iarray_iter_read_block_free(iter_var[nvar]);
        }
        iarray_context_free(&ctx);
        ina_mem_free(iter_var);
        ina_mem_free(iter_value);
        ina_mem_free(outbuf);
    }
    else if (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_CHUNK) {
        // Evaluate the expression for all the chunks in variables
        for (int32_t nchunk = 0; nchunk < e->nchunks; nchunk++) {
            // Decompress chunks in variables into temporaries
            for (int nvar = 0; nvar < nvars; nvar++) {
                blosc2_schunk *schunk = e->vars[nvar].c->catarr->sc;
                int dsize = blosc2_schunk_decompress_chunk(schunk, (int)nchunk, e->temp_vars[nvar]->data,
                                                           (size_t)e->chunksize);
                if (dsize < 0) {
                    printf("Decompression error.  Error code: %d\n", dsize);
                    return INA_ERR_FAILED;
                }
            }
            const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
            blosc2_schunk_append_buffer(out.sc, expr_out->data, (size_t)nitems_in_chunk * e->typesize);
            ina_mempool_reset(e->ctx->mp_tmp_out);
        }
    }
    else if (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERCHUNK) {
        // For the chunksize, choose the minimum of the partition shapes (chunks in Blosc parlance)
        int64_t chunksize = INT64_MAX;
        for (int nvar = 0; nvar < nvars; nvar++) {
            iarray_container_t *var = e->vars[nvar].c;
            chunksize = INA_MIN(chunksize, var->dtshape->pshape[0]);
        }

        // Create and initialize an iterator per variable
        iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
        iarray_context_t *ctx = NULL;
        iarray_context_new(&cfg, &ctx);
        iarray_iter_read_block_t **iter_var = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_t));
        iarray_iter_read_block_value_t *iter_value = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_value_t));

        for (int nvar = 0; nvar < nvars; nvar++) {
            iarray_container_t *var = e->vars[nvar].c;
            iarray_iter_read_block_new(ctx, &iter_var[nvar], var, &blocksize, &iter_value[nvar]);
        }

        // Evaluate the expression for all the chunks in variables
        int64_t nitems_written = 0;
        while (iarray_iter_read_block_has_next(iter_var[0])) {

            // Decompress chunks in variables into temporaries
            for (int nvar = 0; nvar < nvars; nvar++) {
                iarray_iter_read_block_next(iter_var[nvar]);
                e->temp_vars[nvar]->data = iter_value[nvar].pointer;
            }

            // Eval the expression for this chunk
            const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
            blosc2_schunk_append_buffer(out.sc, expr_out->data, (size_t)nitems_in_chunk * e->typesize);
            nitems_written += nitems_in_chunk;
            ina_mempool_reset(e->ctx->mp_tmp_out);
        }

        for (int nvar = 0; nvar < nvars; nvar++) {
            iarray_iter_read_block_free(iter_var[nvar]);
        }
        ina_mem_free(iter_var);
        ina_mem_free(iter_value);
        assert(nitems_written == nitems_in_schunk);
    }
    else if (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERCHUNKPARA) {
        // This version of the evaluation engine works by using a chunk iterator and use OpenMP
        // for performing the computations.  The OpenMP loop split the chunk into smaller blocks that
        // are passed the tinyexpr evaluator.
        // Although this works perfectly well, this is still preliminary because we may want to
        // get rid of the overhead of creating/destroying the thread per every chunk.  One possibility
        // is to use pthreads, but we need more discussion about this.
        int32_t blocksize = e->blocksize;
        int64_t chunksize = e->chunksize;

        // Create and initialize an iterator per variable
        iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
        iarray_context_t *ctx = NULL;
        iarray_context_new(&cfg, &ctx);
        iarray_iter_read_block_t **iter_var = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_t));
        int64_t nitems = chunksize / e->typesize;
        for (int nvar = 0; nvar < nvars; nvar++) {
            iarray_container_t *var = e->vars[nvar].c;
            iarray_iter_read_block_new(ctx, var, &iter_var[nvar], &nitems);
            iarray_iter_read_block_init(iter_var[nvar]);
        }

        // Evaluate the expression for all the chunks in variables
        iarray_iter_read_block_value_t *iter_value = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_value_t));
        int64_t nitems_written = 0;
        int nblocks = (int)chunksize / blocksize;
        int8_t *outbuf = ina_mem_alloc((size_t)chunksize);
        while (nitems_written < nitems_in_schunk) {
            // Decompress chunks in variables into temporaries
            for (int nvar = 0; nvar < nvars; nvar++) {
                iarray_iter_read_block_value(iter_var[nvar], &iter_value[nvar]);
            }

            // Eval the expression for this chunk, split by blocks
#ifndef __clang__
#pragma omp parallel for // schedule(dynamic)
#endif
            for (int nblock = 0; nblock < nblocks ; nblock++) {
                for (int nvar = 0; nvar < nvars; nvar++) {
                    int nthread = 0;
#ifndef __clang__
                    nthread = omp_get_thread_num();
#endif
                    int ntvar = nthread * e->nvars + nvar;
                    e->temp_vars[ntvar]->data = iter_value[nvar].pointer + nblock * blocksize;
                }
                const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
                memcpy(outbuf + nblock * blocksize, (uint8_t*)expr_out->data, blocksize);
            }

            // Do a possible last evaluation with the leftovers
            int leftover = chunksize - nblocks * blocksize;
            if (leftover > 0) {
                for (int nvar = 0; nvar < nvars; nvar++) {
                    e->temp_vars[nvar]->data = iter_value[nvar].pointer + nblocks * blocksize;
                }
                e->max_out_len = leftover / e->typesize;  // so as to prevent operating beyond the leftover
                const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
                e->max_out_len = 0;
                memcpy(outbuf + nblocks * blocksize, (uint8_t*)expr_out->data, leftover);
            }

            // Write the resulting chunk in output
            blosc2_schunk_append_buffer(out.sc, outbuf, (size_t)chunksize);
            nitems_written += nitems_in_chunk;

            ina_mempool_reset(e->ctx->mp_tmp_out);

            // Get ready for the next iteration
            for (int nvar = 0; nvar < nvars; nvar++) {
                iarray_iter_read_block_next(iter_var[nvar]);
            }
        }

        for (int nvar = 0; nvar < nvars; nvar++) {
            iarray_iter_read_block_free(iter_var[nvar]);
        }
        ina_mem_free(iter_var);
        ina_mem_free(iter_value);
        ina_mem_free(outbuf);

        assert(nitems_written == nitems_in_schunk);
    }

    ina_mempool_reset(e->ctx->mp);
    ina_mempool_reset(e->ctx->mp_op);
    ina_mempool_reset(e->ctx->mp_tmp_out);
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
        default:
            return INA_ERR_EXCEEDED;
    }
    for (int i = 0; i < dtshape->ndim; ++i) {
        *size += dtshape->shape[i] * type_size;
    }
    return INA_SUCCESS;
}

ina_rc_t iarray_temporary_new(iarray_expression_t *expr, iarray_container_t *c, iarray_dtshape_t *dtshape,
        iarray_temporary_t **temp)
{
    // When c == NULL means a temporary for output, which should go to its own memory pool for being
    // able to reset it during each block/chunk evaluation
    ina_mempool_t *mempool = (c != NULL) ? expr->ctx->mp : expr->ctx->mp_tmp_out;
    *temp = ina_mempool_dalloc(mempool, sizeof(iarray_temporary_t));
    (*temp)->dtshape = ina_mempool_dalloc(mempool, sizeof(iarray_dtshape_t));
    ina_mem_cpy((*temp)->dtshape, dtshape, sizeof(iarray_dtshape_t));
    size_t size = 0;
    iarray_shape_size(dtshape, &size);
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

static iarray_temporary_t* _iarray_op(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_optype_t op)
{
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

    // Creating the temporary means interacting with the INA memory allocator, which is not thread-safe.
    // We should investigate on how to overcome this syncronization point (if possible at all).
#pragma omp critical
    iarray_temporary_new(expr, NULL, &dtshape, &out);

    switch (dtshape.dtype) {
        case IARRAY_DATA_TYPE_DOUBLE: {
            int len = expr->max_out_len == 0 ? (int)(out->size / sizeof(double)) : expr->max_out_len;
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
                    return NULL;
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
                    return NULL;
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
                    return NULL;
                }
            }
            else {
                printf("DTshape combination not supported yet\n");
                return NULL;
            }
        }
        break;
        case IARRAY_DATA_TYPE_FLOAT: {
            int len = expr->max_out_len == 0 ? (int)(out->size / sizeof(float)) : expr->max_out_len;
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
                    return NULL;
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
                    return NULL;
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
                    return NULL;
                }
            }
            else {
                printf("DTshape combination not supported yet\n");
                return NULL;
            }
        }
        break;
        default:  // switch (dtshape.dtype)
            printf("data type not supported yet\n");
            return NULL;
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

INA_API(ina_rc_t) iarray_expr_get_nthreads(iarray_expression_t *e, int *nthreads)
{
    *nthreads = e->ctx->cfg->max_num_threads;
    return INA_SUCCESS;
}
