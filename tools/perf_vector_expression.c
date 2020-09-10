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

#define NELEM (20 * 1000 * 1000)  // multiple of NITEMS_CHUNK for now
#define NITEMS_CHUNK (4000 * 1000)
#define NITEMS_BLOCK (16000)
#define XMAX 10.

static double _poly(const double x)
{
    return (x - 1.35) * (x - 4.45) * (x - 8.5);
}

static double _trans1(const double x)
{
    return sin(x) * sin(x) + cos(x) * cos(x);
}

static double _trans2(const double x)
{
    return (cos(x) - 1.35) * (x - 4.45) * (sin(x) - 8.5);
}

// Fill X values in regular array
static int _fill_x(double* x)
{
    double incx = XMAX / NELEM;

    /* Fill even values between 0 and 10 */
    for (int i = 0; i < NELEM; i++) {
        x[i] = incx * i;
    }
    return 0;
}

// Compute and fill Y values in regular array
static void _compute_y(const double* x, double* y, int expr_type)
{
// If compiled with OpenMP executes, it prevents the pthreads in Blosc (e.g. EVAL_ITERBLOSC) to run in parallel (!)
// See #176
// #pragma omp parallel for
    switch (expr_type) {
        case 0:
            for (int i = 0; i < NELEM; i++) {
                y[i] = x[i];
            }
            break;
        case 1:
            for (int i = 0; i < NELEM; i++) {
                y[i] = _poly(x[i]);
            }
            break;
        case 2:
            for (int i = 0; i < NELEM; i++) {
                y[i] = _trans1(x[i]);
            }
            break;
        case 3:
            for (int i = 0; i < NELEM; i++) {
                y[i] = _trans2(x[i]);
            }
            break;
        default:
            printf("Wrong expr-type value!\n");
    }
}

static void ina_cleanup_handler(int error, int *exitcode)
{
    INA_UNUSED(error);
    INA_UNUSED(exitcode);
    iarray_destroy();
}

static double *x = NULL;
static double *y = NULL;

int main(int argc, char** argv)
{
    int64_t shape[] = {NELEM};
    int64_t cshape[] = {NITEMS_CHUNK};
    int64_t bshape[] = {NITEMS_BLOCK};
    int8_t ndim = 1;
    ina_stopwatch_t *w;
    iarray_context_t *ctx = NULL;
    char *mat_x_name = NULL;
    char *mat_y_name = NULL;
    char *mat_out_name = NULL;

    INA_OPTS(opt,
             INA_OPT_INT("e", "expr-type", 1, "COPY = 0, POLY = 1, TRANS1 = 2, , TRANS2 = 3"),
             INA_OPT_INT("M", "eval-method", 1, "EVAL_ITERCHUNK = 1, EVAL_ITERBLOSC = 2, EVAL_ITERBLOSC2 = 3"),
             INA_OPT_INT("E", "eval-engine", 1, "EVAL_INTERPRETER = 1, EVAL_COMPILER = 2"),
             INA_OPT_INT("n", "eval-niter", 1, "Number of times to evaluate (default 1)"),
             INA_OPT_INT("c", "clevel", 5, "Compression level"),
             INA_OPT_INT("l", "codec", 1, "Compression codec"),
             INA_OPT_INT("f", "filter", 1, "SHUFFLE = 1, BITSHUFFLE = 2"),
             INA_OPT_INT("t", "nthreads", 1, "Use number of threads for the evaluation"),
             INA_OPT_INT("m", "mantissa-bits", 0, "The number of significant bits in mantissa (0 means no truncation"),
             INA_OPT_FLAG("d", "dict", "Use dictionary (only for Zstd (codec 5))"),
             INA_OPT_FLAG("P", "plainbuffer", "Use plain buffer arrays"),
             INA_OPT_FLAG("i", "iter", "Use iterator for filling values"),
             INA_OPT_FLAG("I", "iter-chunk", "Use chunk iterator for filling values"),
             INA_OPT_FLAG("p", "persistence", "Use persistent containers"),
             INA_OPT_FLAG("r", "remove", "Remove the previous persistent containers (only valid w/ -p)")
    );

    if (!INA_SUCCEED(ina_app_init(argc, argv, opt))) {
        return EXIT_FAILURE;
    }
    ina_set_cleanup_handler(ina_cleanup_handler);

    int expr_type;
    INA_MUST_SUCCEED(ina_opt_get_int("e", &expr_type));
    int eval_method;
    INA_MUST_SUCCEED(ina_opt_get_int("M", &eval_method));
    int eval_niter;
    INA_MUST_SUCCEED(ina_opt_get_int("n", &eval_niter));
    int clevel;
    INA_MUST_SUCCEED(ina_opt_get_int("c", &clevel));
    int codec;
    INA_MUST_SUCCEED(ina_opt_get_int("l", &codec));
    int filter;
    INA_MUST_SUCCEED(ina_opt_get_int("f", &filter));
    int nthreads;
    INA_MUST_SUCCEED(ina_opt_get_int("t", &nthreads));
    int mantissa_bits;
    INA_MUST_SUCCEED(ina_opt_get_int("m", &mantissa_bits));

    if (INA_SUCCEED(ina_opt_isset("p"))) {
        mat_x_name = "mat_x.b2frame";
        mat_y_name = "mat_y.b2frame";
        mat_out_name = "mat_out.b2frame";
        if (INA_SUCCEED(ina_opt_isset("r"))) {
            remove(mat_x_name);
            remove(mat_y_name);
            remove(mat_out_name);
        }
    }

    iarray_storage_t mat_x = {
        .backend = INA_SUCCEED(ina_opt_isset("P")) ? IARRAY_STORAGE_PLAINBUFFER : IARRAY_STORAGE_BLOSC,
        .enforce_frame = INA_SUCCEED(ina_opt_isset("p")),
        .filename = mat_x_name
    };
    if (!INA_SUCCEED(ina_opt_isset("P"))) {
        mat_x.chunkshape[0] = cshape[0];
        mat_x.blockshape[0] = bshape[0];
    }
    iarray_storage_t mat_y = {
        .backend = INA_SUCCEED(ina_opt_isset("P")) ? IARRAY_STORAGE_PLAINBUFFER : IARRAY_STORAGE_BLOSC,
        .enforce_frame = INA_SUCCEED(ina_opt_isset("p")),
        .filename = mat_y_name
    };
    if (!INA_SUCCEED(ina_opt_isset("P"))) {
        mat_y.chunkshape[0] = cshape[0];
        mat_y.blockshape[0] = bshape[0];
    }
    iarray_storage_t mat_out = {
        .backend = INA_SUCCEED(ina_opt_isset("P")) ? IARRAY_STORAGE_PLAINBUFFER : IARRAY_STORAGE_BLOSC,
        .enforce_frame = INA_SUCCEED(ina_opt_isset("p")),
        .filename = mat_out_name
    };
    if (!INA_SUCCEED(ina_opt_isset("P"))) {
        mat_out.chunkshape[0] = cshape[0];
        mat_out.blockshape[0] = bshape[0];
    }

    int flags = INA_SUCCEED(ina_opt_isset("p"))? IARRAY_CONTAINER_PERSIST : 0;

    INA_MUST_SUCCEED(iarray_init());

    iarray_config_t config = IARRAY_CONFIG_DEFAULTS;
    config.compression_level = clevel;
    config.compression_codec = codec;
    config.filter_flags = filter;
    if (clevel == 0) {
        // If there is no compression, there is no point in using filters.
        config.filter_flags = 0;
    }
    else {
        if (mantissa_bits >  0) {
            config.filter_flags |= IARRAY_COMP_TRUNC_PREC;
            config.fp_mantissa_bits = (uint8_t) mantissa_bits;
        }
    }
    config.use_dict = INA_SUCCEED(ina_opt_isset("d")) ? 1 : 0;
    config.max_num_threads = nthreads;

    const char *expr_type_str = NULL;
    if (expr_type == 0) {
        expr_type_str = "COPY";
    }
    else if (expr_type == 1) {
        expr_type_str = "POLY";
    }
    else if (expr_type == 2) {
        expr_type_str = "TRANS1";
    }
    else if (expr_type == 3) {
        expr_type_str = "TRANS2";
    }
    else {
        printf("expr-type must be 0, 1, 2, 3\n");
        return EXIT_FAILURE;
    }

    const char *eval_method_str = NULL;

    if (eval_method == 1) {
        eval_method_str = "ITERCHUNK";
        eval_method = IARRAY_EVAL_METHOD_ITERCHUNK;
    }
    else if (eval_method == 2) {
        eval_method_str = "ITERBLOSC";
        eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
    }
    else {
        printf("eval_method must be 1, 2\n");
        return EXIT_FAILURE;
    }

    config.eval_method = eval_method;

    INA_MUST_SUCCEED(iarray_context_new(&config, &ctx));

    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    size_t buffer_len = sizeof(double) * NELEM;

    iarray_dtshape_t dtshape;
    dtshape.ndim = ndim;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
    }

    int64_t nbytes = 0;
    int64_t cbytes = 0;
    double nbytes_mb = 0;
    double cbytes_mb = 0;

    iarray_container_t *con_x;
    iarray_container_t *con_y;

    bool x_allocated = false, y_allocated = false;

    if (INA_SUCCEED(ina_opt_isset("p")) && _iarray_file_exists(mat_x.filename)) {
        INA_STOPWATCH_START(w);
        INA_MUST_SUCCEED(iarray_container_load(ctx, mat_x.filename, false, &con_x));
        INA_STOPWATCH_STOP(w);
        INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
        printf("Time for *opening* X values: %.3g s, %.1f GB/s\n",
               elapsed_sec, buffer_len / (elapsed_sec * _IARRAY_SIZE_GB));
    }
    else {
        if (INA_SUCCEED(ina_opt_isset("i"))) {
            INA_STOPWATCH_START(w);
            iarray_container_new(ctx, &dtshape, &mat_x, flags, &con_x);
            iarray_iter_write_t *I;
            iarray_iter_write_value_t val;
            iarray_iter_write_new(ctx, &I, con_x, &val);
            double incx = XMAX / NELEM;
            while (iarray_iter_write_has_next(I)) {
                iarray_iter_write_next(I);
                double value = incx * (double) val.elem_flat_index;
                memcpy(val.elem_pointer, &value, sizeof(double));
            }
            iarray_iter_write_free(&I);
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf("Time for computing and filling X values via iterator: %.3g s, %.1f MB/s\n",
                   elapsed_sec, buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
        }
        else if (INA_SUCCEED(ina_opt_isset("I"))) {
            INA_STOPWATCH_START(w);
            iarray_container_new(ctx, &dtshape, &mat_x, flags, &con_x);
            iarray_iter_write_block_t *I;
            iarray_iter_write_block_value_t val;
            INA_MUST_SUCCEED(iarray_iter_write_block_new(ctx, &I, con_x, NULL, &val, false));
            double incx = XMAX / NELEM;
            while (iarray_iter_write_block_has_next(I)) {
                iarray_iter_write_block_next(I, NULL, 0);
                int64_t chunk_size = val.block_size;  // 1-dim vector
                for (int64_t i = 0; i < chunk_size; ++i) {
                    ((double *) val.block_pointer)[i] = incx * (double) (i + val.nblock * chunk_size);
                }
            }
            iarray_iter_write_block_free(&I);
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf("Time for computing and filling X values via chunk iterator: %.3g s, %.1f MB/s\n",
                   elapsed_sec, buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
        }
        else {
            INA_STOPWATCH_START(w);
            x = (double *) ina_mem_alloc(buffer_len);
            x_allocated = true;
            // Fill the plain x operand
            _fill_x(x);
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf("Time for computing and filling X values: %.3g s, %.1f MB/s\n",
                   elapsed_sec, buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
            INA_STOPWATCH_START(w);
            INA_MUST_SUCCEED(iarray_from_buffer(ctx, &dtshape, x, buffer_len, &mat_x, flags, &con_x));
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf("Time for compressing and *storing* X values: %.3g s, %.1f MB/s\n",
                   elapsed_sec, buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
        }
    }

    iarray_container_info(con_x, &nbytes, &cbytes);
    nbytes_mb = ((double)nbytes / _IARRAY_SIZE_MB);
    cbytes_mb = ((double)cbytes / _IARRAY_SIZE_MB);
    printf("Compression for X values: %.1f MB -> %.1f MB (%.1fx)\n",
           nbytes_mb, cbytes_mb, (1.*nbytes)/cbytes);

    if (INA_SUCCEED(ina_opt_isset("p")) && _iarray_file_exists(mat_y.filename)) {
        INA_STOPWATCH_START(w);
        INA_MUST_SUCCEED(iarray_container_load(ctx, mat_y.filename, false, &con_y));
        INA_STOPWATCH_STOP(w);
        INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
        printf("Time for *opening* Y values: %.3g s, %.1f GB/s\n",
               elapsed_sec, buffer_len / (elapsed_sec * _IARRAY_SIZE_GB));
    }
    else {
        if (INA_SUCCEED(ina_opt_isset("i"))) {
            INA_STOPWATCH_START(w);
            iarray_container_new(ctx, &dtshape, &mat_y, flags, &con_y);
            iarray_iter_write_t *I;
            iarray_iter_write_value_t val;
            iarray_iter_write_new(ctx, &I, con_y, &val);
            double incx = XMAX / NELEM;
            while (iarray_iter_write_has_next(I)) {
                iarray_iter_write_next(I);
                double value;
                double _x = incx * (double) val.elem_flat_index;
                switch (expr_type) {
                    case 0:
                        value = _x;
                        break;
                    case 1:
                        value = _poly(_x);
                        break;
                    case 2:
                        value = _trans1(_x);
                        break;
                    case 3:
                        value = _trans2(_x);
                        break;
                    default:
                        printf("Wrong expr-type value!\n");
                        return 1;
                }
                memcpy(val.elem_pointer, &value, sizeof(double));
            }
            iarray_iter_write_free(&I);
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf("Time for computing and filling Y values via iterator: %.3g s, %.1f MB/s\n",
                   elapsed_sec, buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
        }
        else if (INA_SUCCEED(ina_opt_isset("I"))) {
            INA_STOPWATCH_START(w);
            iarray_container_new(ctx, &dtshape, &mat_y, flags, &con_y);
            iarray_iter_write_block_t *I;
            iarray_iter_write_block_value_t val;
            iarray_iter_write_block_new(ctx, &I, con_y, mat_y.chunkshape, &val, false);
            double incx = XMAX / NELEM;
            while (iarray_iter_write_block_has_next(I)) {
                iarray_iter_write_block_next(I, NULL, 0);
                int64_t chunk_size = val.block_size;
                switch (expr_type) {
                    case 0:
                        for (int64_t i = 0; i < chunk_size; ++i) {
                            double _x = incx * (double) (i + val.nblock * chunk_size);
                            ((double *) val.block_pointer)[i] = _x;
                        }
                        break;
                    case 1:
                        for (int64_t i = 0; i < chunk_size; ++i) {
                            double _x = incx * (double) (i + val.nblock * chunk_size);
                            ((double *) val.block_pointer)[i] = _poly(_x);
                        }
                        break;
                    case 2:
                        for (int64_t i = 0; i < chunk_size; ++i) {
                            double _x = incx * (double) (i + val.nblock * chunk_size);
                            ((double *) val.block_pointer)[i] = _trans1(_x);
                        }
                        break;
                    case 3:
                        for (int64_t i = 0; i < chunk_size; ++i) {
                            double _x = incx * (double) (i + val.nblock * chunk_size);
                            ((double *) val.block_pointer)[i] = _trans2(_x);
                        }
                        break;
                    default:
                        printf("Wrong expr-type value!\n");
                }
            }
            iarray_iter_write_block_free(&I);
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf(
                "Time for computing and filling Y values via block iterator: %.3g s, %.1f MB/s\n",
                elapsed_sec, buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
        }
        else {
            // Compute the plain y vector
            INA_STOPWATCH_START(w);
            y = (double*)ina_mem_alloc(buffer_len);
            y_allocated = true;
            _compute_y(x, y, expr_type);
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf("Time for computing and filling Y values: %.3g s, %.1f MB/s\n",
                   elapsed_sec, buffer_len/(elapsed_sec*_IARRAY_SIZE_MB));
            INA_STOPWATCH_START(w);
            INA_MUST_SUCCEED(iarray_from_buffer(ctx, &dtshape, y, buffer_len, &mat_y, flags, &con_y));
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf("Time for compressing and *storing* Y values: %.3g s, %.1f MB/s\n",
                   elapsed_sec, buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
        }
    }

    iarray_container_info(con_y, &nbytes, &cbytes);
    nbytes_mb = ((double)nbytes / _IARRAY_SIZE_MB);
    cbytes_mb = ((double)cbytes / _IARRAY_SIZE_MB);
    printf("Compression for Y values: %.1f MB -> %.1f MB (%.1fx)\n",
           nbytes_mb, cbytes_mb, (1.*nbytes) / cbytes);

    // Check IronArray performance
    iarray_container_t *con_out = NULL;

    iarray_expression_t *e;
    iarray_expr_new(ctx, &e);
    iarray_expr_bind(e, "x", con_x);
    iarray_expr_bind_out_properties(e, &dtshape, &mat_out);
    switch (expr_type) {
        case 0:
            iarray_expr_compile(e, "x");
            break;
        case 1:
            iarray_expr_compile(e, "(x - 1.35) * (x - 4.45) * (x - 8.5)");
            break;
        case 2:
            iarray_expr_compile(e, "sin(x) * sin(x) + cos(x) * cos(x)");
            break;
        case 3:
            iarray_expr_compile(e, "(cos(x) - 1.35) * (x - 4.45) * (sin(x) - 8.5)");
            break;
        default:
            printf("Wrong expr-type value!\n");
    }

    INA_STOPWATCH_START(w);
    for (int i = 0; i < eval_niter; i++) {
        ina_rc_t errcode = iarray_eval(e, &con_out);
        if (errcode != INA_SUCCESS) {
            printf("Error during evaluation.  Giving up...\n");
            return -1;
        }
    }
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    iarray_container_info(con_out, &nbytes, &cbytes);
    printf("\n");
    printf("Time for computing and filling OUT values using iarray (%s, %s):  %.3g s, %.1f MB/s\n",
           expr_type_str, eval_method_str, elapsed_sec, (nbytes * eval_niter) / (elapsed_sec * _IARRAY_SIZE_MB));
    nbytes_mb = ((double)nbytes / (double)_IARRAY_SIZE_MB);
    cbytes_mb = ((double)cbytes / (double)_IARRAY_SIZE_MB);
    printf("Compression for OUT values: %.1f MB -> %.1f MB (%.1fx)\n",
           nbytes_mb, cbytes_mb, (1.*nbytes)/cbytes);

    iarray_expr_free(ctx, &e);

    printf("Checking that the outcome of the expression is correct...");
    fflush(stdout);
    bool not_equal = false;
    INA_STOPWATCH_START(w);
    IARRAY_FAIL_IF_ERROR(iarray_container_almost_equal(NULL, con_y, con_out));
    printf(" Yes!\n");
    goto success;
  fail:
    printf(" No!\n");
    not_equal = true;
  success:
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time for checking that two iarrays are equal:  %.3g s, %.1f MB/s\n",
           elapsed_sec, (nbytes * 2.) / (elapsed_sec * _IARRAY_SIZE_MB));

    iarray_container_free(ctx, &con_x);
    iarray_container_free(ctx, &con_y);
    iarray_container_free(ctx, &con_out);

    iarray_context_free(&ctx);

    if (x_allocated) ina_mem_free(x);
    if (y_allocated) ina_mem_free(y);

    INA_STOPWATCH_FREE(&w);

    if (not_equal) {
      return 1;
    }
    else {
      return 0;
    }
}
