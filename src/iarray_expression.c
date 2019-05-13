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
#if defined(_OPENMP)
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
    int64_t nbytes;
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
    ina_str_free((*e)->expr);
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

#if defined(_OPENMP)
    if (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERBLOCK) {
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

    e->expr = ina_str_new_fromcstr(expr);
    e->temp_vars = ina_mem_alloc(nthreads * e->nvars * sizeof(iarray_temporary_t*));
    te_variable *te_vars = ina_mempool_dalloc(e->ctx->mp, e->nvars * sizeof(te_variable));
    caterva_array_t *catarr = e->vars[0].c->catarr;

    e->typesize = catarr->ctx->cparams.typesize;
    e->nbytes = catarr->size * e->typesize;
    if (catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        // Somewhat arbitrary values follows
        e->blocksize = 1024 * e->typesize;
        e->chunksize = 16 * e->blocksize;
    }
    else {
        blosc2_schunk *schunk = catarr->sc;
        if (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERBLOCK) {
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
            e->chunksize = (int32_t) chunksize;
            e->blocksize = (int32_t) blocksize;
        }
        else if (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERCHUNK) {
            e->chunksize = schunk->chunksize;
        }
        else {
            fprintf(stderr, "Flag %d is not supported\n", e->ctx->cfg->eval_flags);
            return INA_ERR_NOT_SUPPORTED;
        }
    }

    e->nchunks = e->nbytes / e->chunksize;
    if (e->nchunks * e->chunksize < e->nbytes) {
        e->nchunks += 1;
    }

    // Create temporaries for initial variables
    // TODO: make this more general and accept multidimensional containers
    iarray_dtshape_t dtshape_var = {0};  // initialize to 0s
    dtshape_var.ndim = 1;
    int temp_var_dim0 = 0;
    if (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERBLOCK) {
        temp_var_dim0 = e->blocksize / e->typesize;
    } else if (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERCHUNK) {
        temp_var_dim0 = e->chunksize / e->typesize;
        e->blocksize = 0;
    } else {
        fprintf(stderr, "Flag %d is not supported\n", e->ctx->cfg->eval_flags);
        return INA_ERR_NOT_SUPPORTED;
    }
    dtshape_var.shape[0] = temp_var_dim0;
    dtshape_var.dtype = e->vars[0].c->dtshape->dtype;
    for (int nvar = 0; nvar < e->nvars; nvar++) {
        te_vars[nvar].name = e->vars[nvar].var;
        te_vars[nvar].type = TE_VARIABLE;
        te_vars[nvar].context = NULL;
        te_vars[nvar].address = ina_mempool_dalloc(e->ctx->mp, nthreads * sizeof(void*));
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
    int64_t nitems_in_schunk = e->nbytes / e->typesize;
    int64_t nitems_written = 0;
    int nvars = e->nvars;
    caterva_dims_t shape = caterva_new_dims(e->vars[0].c->dtshape->shape, e->vars[0].c->dtshape->ndim);
    caterva_update_shape(ret->catarr, &shape);
    ret->catarr->size = 1;  // TODO: fix this workaround (see caterva_update_shape() call above)
    int64_t out_pshape = e->chunksize / e->typesize;

    if (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERCHUNK) {

        // Create and initialize an iterator per variable
        iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
        iarray_context_t *ctx = NULL;
        iarray_context_new(&cfg, &ctx);
        iarray_iter_read_block_t **iter_var = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_t));
        iarray_iter_read_block_value_t *iter_value = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_value_t));
        for (int nvar = 0; nvar < nvars; nvar++) {
            iarray_container_t *var = e->vars[nvar].c;
            iarray_iter_read_block_new(ctx, &iter_var[nvar], var, &out_pshape, &iter_value[nvar]);
        }

        // Write iterator for output
        iarray_iter_write_block_t *iter_out;
        iarray_iter_write_block_value_t out_value;
        ina_rc_t err = iarray_iter_write_block_new(ctx, &iter_out, ret, &out_pshape, &out_value);
        if (err != INA_SUCCESS) {
            return err;
        }

        // Evaluate the expression for all the chunks in variables
        while (iarray_iter_write_block_has_next(iter_out)) {
            iarray_iter_write_block_next(iter_out);
            int out_items = iter_out->cur_block_size;

            // Decompress chunks in variables into temporaries
            for (int nvar = 0; nvar < nvars; nvar++) {
                iarray_iter_read_block_next(iter_var[nvar]);
                e->temp_vars[nvar]->data = iter_value[nvar].pointer;
            }

            // Eval the expression for this chunk
            e->max_out_len = out_items;  // so as to prevent operating beyond the limits
            const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
            memcpy((char*)out_value.pointer, (uint8_t*)expr_out->data, out_items * e->typesize);
            nitems_written += out_items;
            ina_mempool_reset(e->ctx->mp_tmp_out);
        }

        for (int nvar = 0; nvar < nvars; nvar++) {
            iarray_iter_read_block_free(iter_var[nvar]);
        }
        iarray_iter_write_block_free(iter_out);
        ina_mem_free(iter_var);
        ina_mem_free(iter_value);
        iarray_context_free(&ctx);
    }
    else if (e->ctx->cfg->eval_flags & IARRAY_EXPR_EVAL_ITERBLOCK) {
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
            iarray_iter_read_block_new(ctx, &iter_var[nvar], var, &out_pshape, &iter_value[nvar]);
        }

        // Write iterator for output
        iarray_iter_write_block_t *iter_out;
        iarray_iter_write_block_value_t out_value;
        ina_rc_t err = iarray_iter_write_block_new(ctx, &iter_out, ret, &out_pshape, &out_value);
        if (err != INA_SUCCESS) {
            return err;
        }

        // Evaluate the expression for all the chunks in variables
        int8_t *outbuf = ina_mem_alloc((size_t)chunksize);
        bool has_next = iarray_iter_write_block_has_next(iter_out);
        int nblocks;
        int out_items;
#if defined(_OPENMP)
#pragma omp parallel
{
#endif
#if defined(_OPENMP)
        #pragma omp master
        {
#endif
        while (has_next) {
            int nthread_ = 0;
#if defined(_OPENMP)
            nthread_ = omp_get_thread_num();
#endif

            iarray_iter_write_block_next(iter_out);
            for (int nvar = 0; nvar < nvars; nvar++) {
                iarray_iter_read_block_next(iter_var[nvar]);
            }

            printf("Chunk %lld (thread %d)\n", out_value.nblock, nthread_);
            out_items = iter_out->cur_block_size;
            nblocks = out_items * e->typesize / blocksize;

            printf("Blocksize: %d\n", blocksize);
            // Decompress chunks in variables into temporaries

            // Eval the expression for this chunk, split by blocks
#if defined(_OPENMP)
            //}
#endif

            int nthread__ = 0;

#if defined(_OPENMP)
//#pragma omp for schedule(runtime)
#endif
            for (int nblock = 0; nblock < nblocks; nblock++) {
#if defined(_OPENMP)
                nthread__ = omp_get_thread_num();
#endif
                printf("- Block %d (thread %d)\n", nblock, nthread__);
                for (int nvar = 0; nvar < nvars; nvar++) {
                    int nthread = 0;
#if defined(_OPENMP)
                    nthread = omp_get_thread_num();
#endif
                    int ntvar = nthread * e->nvars + nvar;
                    e->temp_vars[ntvar]->data = (char *) iter_value[nvar].pointer + nblock * blocksize;
                }
                e->max_out_len = blocksize / e->typesize;  // so as to prevent operating beyond the limits
                const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
                memcpy((char *) out_value.pointer + nblock * blocksize, (uint8_t *) expr_out->data, blocksize);
            }

#if defined(_OPENMP)
//#pragma omp barrier
//#pragma omp single
//            {
#endif
            // Do a possible last evaluation with the leftovers
            int leftover = out_items * e->typesize - nblocks * blocksize;
            if (leftover > 0) {
                for (int nvar = 0; nvar < nvars; nvar++) {
                    e->temp_vars[nvar]->data = (char *) iter_value[nvar].pointer + nblocks * blocksize;
                }
                e->max_out_len = leftover / e->typesize;  // so as to prevent operating beyond the leftover
                const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
                e->max_out_len = 0;
                memcpy((char *) out_value.pointer + nblocks * blocksize, (uint8_t *) expr_out->data, leftover);
            }

            // Write the resulting chunk in output
            nitems_written += out_items;
            ina_mempool_reset(e->ctx->mp_tmp_out);

            has_next = iarray_iter_write_block_has_next(iter_out);

        }
#if defined(_OPENMP)
        }
#endif

#if defined(_OPENMP)
        }
#endif

        for (int nvar = 0; nvar < nvars; nvar++) {
            iarray_iter_read_block_free(iter_var[nvar]);
        }

        iarray_iter_write_block_free(iter_out);
        ina_mem_free(iter_var);
        ina_mem_free(iter_value);
        ina_mem_free(outbuf);
        iarray_context_free(&ctx);
    }

    ina_mempool_reset(e->ctx->mp);
    ina_mempool_reset(e->ctx->mp_op);
    ina_mempool_reset(e->ctx->mp_tmp_out);

    if (nitems_written != nitems_in_schunk) {
        printf("nitems written is different from items in final container\n");
        return INA_ERR_ERROR;
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
#if defined(_OPENMP)
#pragma omp critical
#endif
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
