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

#include <libiarray/iarray.h>
#include <math.h>
#include "src/iarray_private.h"

#define NITEMS_BLOCK (8 * 1024)
#define NITEMS_CHUNK (64 * NITEMS_BLOCK)
#define NELEM (500 * NITEMS_CHUNK)

#define XMAX 10.

static double _poly(const double x)
{
    //return (x - 1.35) * (x - 4.45) * (x - 8.5);
    return sin(x) * sin(x) + cos(x) * cos(x);
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
static void _compute_y(const double* x, double* y)
{
// If compiled with OpenMP executes, it prevents the pthreads in Blosc (e.g. EVAL_ITERBLOSC) to run in parallel (!)
// See #176
// #pragma omp parallel for
    for (int i = 0; i < NELEM; i++) {
        y[i] = _poly(x[i]);
    }
}

static void ina_cleanup_handler(int error, const int *exitcode)
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
    int8_t ndim = 1;
    ina_stopwatch_t *w;
    iarray_context_t *ctx = NULL;
    char *mat_x_name = NULL;
    char *mat_y_name = NULL;
    char *mat_out_name = NULL;
    char *eval_method_str = NULL;

    INA_OPTS(opt,
             INA_OPT_INT("e", "eval-method", 1, "EVAL_ITERCHUNK = 1, EVAL_ITERBLOSC = 2"),
             INA_OPT_INT("c", "clevel", 5, "Compression level (default 5)"),
             INA_OPT_INT("l", "codec", 1, "Compression codec (default LZ4/1)"),
             INA_OPT_INT("t", "nthreads", 1, "Use number of threads for the evaluation (default 1)"),
             INA_OPT_INT("m", "mantissa-bits", 0, "The number of significant bits in mantissa (0 means no truncation"),
             INA_OPT_FLAG("d", "dict", "Use dictionary (only for Zstd (codec 5))"),
             INA_OPT_FLAG("i", "iter", "Use iterator for filling values"),
             INA_OPT_FLAG("I", "iter-chunk", "Use chunk iterator for filling values"),
             INA_OPT_FLAG("p", "persistence", "Use persistent containers"),
             INA_OPT_FLAG("r", "remove", "Remove the previous persistent containers (only valid w/ -p)"),
             INA_OPT_FLAG("B", "btune", "Activate BTune"),
             INA_OPT_INT("b", "btune-favor", 1, "BALANCE = 0, SPEED = 1, CRATIO = 2 (default: 1)")
    );

    if (!INA_SUCCEED(ina_app_init(argc, argv, opt))) {
        return EXIT_FAILURE;
    }
    ina_set_cleanup_handler((ina_cleanup_handler_t)ina_cleanup_handler);

    int eval_method;
    INA_MUST_SUCCEED(ina_opt_get_int("e", &eval_method));
    int clevel;
    INA_MUST_SUCCEED(ina_opt_get_int("c", &clevel));
    int codec;
    INA_MUST_SUCCEED(ina_opt_get_int("l", &codec));
    int btune_favor;
    INA_MUST_SUCCEED(ina_opt_get_int("b", &btune_favor));
    int nthreads;
    INA_MUST_SUCCEED(ina_opt_get_int("t", &nthreads));
    int mantissa_bits;
    INA_MUST_SUCCEED(ina_opt_get_int("m", &mantissa_bits));

    if (INA_SUCCEED(ina_opt_isset("p"))) {
        mat_x_name = "mat_x.b2frame";
        mat_y_name = "mat_y.b2frame";
        mat_out_name = "mat_out.b2frame";
        if (INA_SUCCEED(ina_opt_isset("r"))) {
            blosc2_remove_urlpath(mat_x_name);
            blosc2_remove_urlpath(mat_y_name);
            blosc2_remove_urlpath(mat_out_name);
        }
    }

    iarray_storage_t mat_x = {
        .contiguous = INA_SUCCEED(ina_opt_isset("p")),
        .urlpath = mat_x_name
    };

    iarray_storage_t mat_y = {
        .contiguous = INA_SUCCEED(ina_opt_isset("p")),
        .urlpath = mat_y_name
    };

    iarray_storage_t mat_out = {
        .contiguous = INA_SUCCEED(ina_opt_isset("p")),
        .urlpath = mat_out_name
    };

    INA_MUST_SUCCEED(iarray_init());

    iarray_config_t config = IARRAY_CONFIG_DEFAULTS;
    config.compression_level = clevel;
    config.compression_codec = codec;
    config.btune = false;
    config.compression_favor = btune_favor;
    if (INA_SUCCEED(ina_opt_isset("B"))) {
        config.btune = true;
    }
    if (clevel == 0) {
        // If there is no compression, there is no point in using filters.
        config.filter_flags = 0;
        config.btune = false;
    }
    else {
        config.filter_flags = IARRAY_COMP_SHUFFLE;
        if (mantissa_bits > 0) {
            config.filter_flags |= (int) IARRAY_COMP_TRUNC_PREC;
            config.fp_mantissa_bits = (uint8_t) mantissa_bits;
        }
    }
    config.use_dict = INA_SUCCEED(ina_opt_isset("d")) ? 1 : 0;
    config.max_num_threads = nthreads;
    config.eval_method = eval_method;
    if (eval_method == IARRAY_EVAL_METHOD_ITERCHUNK) {
        eval_method_str = "EVAL_ITERCHUNK";
    }
    else if (eval_method == IARRAY_EVAL_METHOD_ITERBLOSC) {
        eval_method_str = "EVAL_ITERBLOSC";
    }
    else {
        printf("eval_method must be 1, 2\n");
        return EXIT_FAILURE;
    }
    //config.blocksize = 16 * _IARRAY_SIZE_KB;  // 16 KB seems optimal for evaluating expressions

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
    double nbytes_mb;
    double cbytes_mb;

    iarray_container_t *con_x;
    iarray_container_t *con_y;

    bool x_allocated = false, y_allocated = false;

    if (INA_SUCCEED(ina_opt_isset("p")) && _iarray_path_exists(mat_x.urlpath)) {
        INA_STOPWATCH_START(w);
        INA_MUST_SUCCEED(iarray_container_open(ctx, mat_x.urlpath, &con_x));
        INA_STOPWATCH_STOP(w);
        INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
        printf("Time for *opening* X values: %.3g s, %.1f GB/s\n",
               elapsed_sec, (double)buffer_len / (elapsed_sec * _IARRAY_SIZE_GB));
    }
    else {
        if (INA_SUCCEED(ina_opt_isset("i"))) {
            INA_STOPWATCH_START(w);
            iarray_container_new(ctx, &dtshape, &mat_x, &con_x);
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
                   elapsed_sec, (double)buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
        }
        else if (INA_SUCCEED(ina_opt_isset("I"))) {
            INA_STOPWATCH_START(w);
            iarray_container_new(ctx, &dtshape, &mat_x, &con_x);
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
                   elapsed_sec, (double)buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
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
                   elapsed_sec, (double)buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
            INA_STOPWATCH_START(w);
            INA_MUST_SUCCEED(iarray_from_buffer(ctx, &dtshape, x, buffer_len, &mat_x, &con_x));
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf("Time for compressing and *storing* X values: %.3g s, %.1f MB/s\n",
                   elapsed_sec, (double)buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
        }
    }

    iarray_container_info(con_x, &nbytes, &cbytes);
    nbytes_mb = ((double)nbytes / _IARRAY_SIZE_MB);
    cbytes_mb = ((double)cbytes / _IARRAY_SIZE_MB);
    printf("Compression for X values: %.1f MB -> %.1f MB (%.1fx)\n",
           nbytes_mb, cbytes_mb, (1.*(double)nbytes)/(double)cbytes);

    if (INA_SUCCEED(ina_opt_isset("p")) && _iarray_path_exists(mat_y.urlpath)) {
        INA_STOPWATCH_START(w);
        INA_MUST_SUCCEED(iarray_container_open(ctx, mat_y.urlpath, &con_y));
        INA_STOPWATCH_STOP(w);
        INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
        printf("Time for *opening* Y values: %.3g s, %.1f GB/s\n",
               elapsed_sec, (double)buffer_len / (elapsed_sec * _IARRAY_SIZE_GB));
    }
    else {
        if (INA_SUCCEED(ina_opt_isset("i"))) {
            INA_STOPWATCH_START(w);
            iarray_container_new(ctx, &dtshape, &mat_y, &con_y);
            iarray_iter_write_t *I;
            iarray_iter_write_value_t val;
            iarray_iter_write_new(ctx, &I, con_y, &val);
            double incx = XMAX / NELEM;
            while (iarray_iter_write_has_next(I)) {
                iarray_iter_write_next(I);
                double value = _poly(incx * (double) val.elem_flat_index);
                memcpy(val.elem_pointer, &value, sizeof(double));
            }
            iarray_iter_write_free(&I);
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf("Time for computing and filling Y values via iterator: %.3g s, %.1f MB/s\n",
                   elapsed_sec, (double)buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
        }
        else if (INA_SUCCEED(ina_opt_isset("I"))) {
            INA_STOPWATCH_START(w);
            iarray_container_new(ctx, &dtshape, &mat_y, &con_y);
            iarray_iter_write_block_t *I;
            iarray_iter_write_block_value_t val;
            iarray_iter_write_block_new(ctx, &I, con_y, mat_y.chunkshape, &val, false);
            double incx = XMAX / NELEM;
            while (iarray_iter_write_block_has_next(I)) {
                iarray_iter_write_block_next(I, NULL, 0);
                int64_t chunk_size = val.block_size;
                for (int64_t i = 0; i < chunk_size; ++i) {
                    ((double *) val.block_pointer)[i] = _poly(incx * (double) (i + val.nblock * chunk_size));
                }
            }
            iarray_iter_write_block_free(&I);
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf(
                "Time for computing and filling Y values via block iterator: %.3g s, %.1f MB/s\n",
                elapsed_sec, (double)buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
        }
        else {
            // Compute the plain y vector
            INA_STOPWATCH_START(w);
            y = (double*)ina_mem_alloc(buffer_len);
            y_allocated = true;
            _compute_y(x, y);
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf("Time for computing and filling Y values: %.3g s, %.1f MB/s\n",
                   elapsed_sec, (double)buffer_len/(elapsed_sec*_IARRAY_SIZE_MB));
            INA_STOPWATCH_START(w);
            INA_MUST_SUCCEED(iarray_from_buffer(ctx, &dtshape, y, buffer_len, &mat_y, &con_y));
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf("Time for compressing and *storing* Y values: %.3g s, %.1f MB/s\n",
                   elapsed_sec, (double)buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
        }
    }

    iarray_container_info(con_y, &nbytes, &cbytes);
    nbytes_mb = ((double)nbytes / _IARRAY_SIZE_MB);
    cbytes_mb = ((double)cbytes / _IARRAY_SIZE_MB);
    printf("Compression for Y values: %.1f MB -> %.1f MB (%.1fx)\n",
           nbytes_mb, cbytes_mb, (1.*(double)nbytes) / (double)cbytes);

    // Check IronArray performanc
    iarray_container_t *con_out;

    iarray_expression_t *e;
    iarray_expr_new(ctx, dtshape.dtype, &e);
    iarray_expr_bind(e, "x", con_x);
    iarray_expr_bind_out_properties(e, &dtshape, &mat_out);
    //iarray_expr_compile(e, "(x - 1.35) * (x - 4.45) * (x - 8.5)");
    iarray_expr_compile(e, "sin(x) * sin(x) + cos(x) * cos(x)");


    INA_STOPWATCH_START(w);
    ina_rc_t errcode = iarray_eval(e, &con_out);
    if (errcode != INA_SUCCESS) {
        printf("Error during evaluation.  Giving up...\n");
        return -1;
    }
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    iarray_container_info(con_out, &nbytes, &cbytes);
    printf("\n");
    printf("Time for computing and filling OUT values using iarray (%s):  %.3g s, %.1f MB/s\n",
           eval_method_str, elapsed_sec, (double)nbytes / (elapsed_sec * _IARRAY_SIZE_MB));
    nbytes_mb = ((double)nbytes / (double)_IARRAY_SIZE_MB);
    cbytes_mb = ((double)cbytes / (double)_IARRAY_SIZE_MB);
    printf("Compression for OUT values: %.1f MB -> %.1f MB (%.1fx)\n",
           nbytes_mb, cbytes_mb, (1.*(double)nbytes)/(double)cbytes);

    iarray_expr_free(ctx, &e);

    printf("Checking that the outcome of the expression is correct...");
    fflush(stdout);
    bool not_equal = false;
    INA_STOPWATCH_START(w);
    IARRAY_FAIL_IF_ERROR(iarray_container_almost_equal(con_y, con_out, 1e-05));
    printf(" Yes!\n");
    goto success;
  fail:
    printf(" No!\n");
    not_equal = true;
  success:
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time for checking that two iarrays are equal:  %.3g s, %.1f MB/s\n",
           elapsed_sec, (double)(nbytes * 2) / (elapsed_sec * _IARRAY_SIZE_MB));

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
