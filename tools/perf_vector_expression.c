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
#define NITEMS_CHUNK (200 * 1000)
#define XMAX 10.

static double _poly(const double x)
{
    return (x - 1.35) * (x - 4.45) * (x - 8.5);
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
    for (int i = 0; i < NELEM; i++) {
        y[i] = _poly(x[i]);
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
    int64_t pshape[] = {NITEMS_CHUNK};
    int8_t ndim = 1;
    ina_stopwatch_t *w;
    iarray_context_t *ctx = NULL;
    const char *mat_x_name = NULL;
    const char *mat_y_name = NULL;
    const char *mat_out_name = NULL;
    const char *eval_method = NULL;

    INA_OPTS(opt,
             INA_OPT_INT("e", "eval-method", 1, "EVAL_ITERCHUNK = 1, EVAL_ITERBLOCK = 2, EVAL_ITERBLOSC = 3"),
             INA_OPT_INT("c", "clevel", 5, "Compression level"),
             INA_OPT_INT("l", "codec", 1, "Compression codec"),
             INA_OPT_INT("b", "blocksize", 0, "Use blocksize for chunks (0 means automatic)"),
             INA_OPT_INT("t", "nthreads", 1, "Use number of threads for the evaluation"),
             INA_OPT_FLAG("d", "dict", "Use dictionary (only for Zstd (codec 5))"),
             INA_OPT_FLAG("P", "plainbuffer", "Use plain buffer arrays"),
             INA_OPT_FLAG("i", "iter", "Use iterator for filling values"),
             INA_OPT_FLAG("I", "iter-part", "Use partition iterator for filling values"),
             INA_OPT_FLAG("p", "persistence", "Use persistent containers"),
             INA_OPT_FLAG("r", "remove", "Remove the previous persistent containers (only valid w/ -p)")
    );

    if (!INA_SUCCEED(ina_app_init(argc, argv, opt))) {
        return EXIT_FAILURE;
    }
    ina_set_cleanup_handler(ina_cleanup_handler);

    int eval_flags;
    INA_MUST_SUCCEED(ina_opt_get_int("e", &eval_flags));
    int clevel;
    INA_MUST_SUCCEED(ina_opt_get_int("c", &clevel));
    int codec;
    INA_MUST_SUCCEED(ina_opt_get_int("l", &codec));
    int blocksize;
    INA_MUST_SUCCEED(ina_opt_get_int("b", &blocksize));
    int nthreads;
    INA_MUST_SUCCEED(ina_opt_get_int("t", &nthreads));

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

    iarray_store_properties_t mat_x = { .id = mat_x_name };
    iarray_store_properties_t mat_y = { .id = mat_y_name };
    iarray_store_properties_t mat_out = { .id = mat_out_name };

    int flags = INA_SUCCEED(ina_opt_isset("p"))? IARRAY_CONTAINER_PERSIST : 0;

    INA_MUST_SUCCEED(iarray_init());

    iarray_config_t config = IARRAY_CONFIG_DEFAULTS;
    config.compression_level = clevel;
    config.compression_codec = codec;
    if (clevel == 0) {
        // If there is no compression, there is no point in using filters.
        config.filter_flags = 0;
    }
    else {
        config.filter_flags = IARRAY_COMP_SHUFFLE;
    }
    config.use_dict = INA_SUCCEED(ina_opt_isset("d")) ? 1 : 0;
    config.blocksize = blocksize;
    config.max_num_threads = nthreads;
    config.eval_flags = eval_flags;
    if (eval_flags == IARRAY_EXPR_EVAL_ITERCHUNK) {
        eval_method = "EVAL_ITERCHUNK";
    }
    else if (eval_flags == IARRAY_EXPR_EVAL_ITERBLOCK) {
        eval_method = "EVAL_ITERBLOCK";
    }
    else if (eval_flags == IARRAY_EXPR_EVAL_ITERBLOSC) {
        eval_method = "EVAL_ITERBLOSC";
    }
    else {
        printf("eval_flags must be 1, 2, 3\n");
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
        dtshape.pshape[i] = INA_SUCCEED(ina_opt_isset("P")) ? 0 : pshape[i];
    }

    int64_t nbytes = 0;
    int64_t cbytes = 0;
    double nbytes_mb = 0;
    double cbytes_mb = 0;

    iarray_container_t *con_x;
    iarray_container_t *con_y;

    bool x_allocated = false, y_allocated = false;

    if (INA_SUCCEED(ina_opt_isset("p")) && _iarray_file_exists(mat_x.id)) {
        INA_STOPWATCH_START(w);
        INA_MUST_SUCCEED(iarray_from_file(ctx, &mat_x, &con_x));
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
                memcpy(val.pointer, &value, sizeof(double));
            }
            iarray_iter_write_free(I);
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
            iarray_iter_write_block_new(ctx, &I, con_x, NULL, &val, false);
            double incx = XMAX / NELEM;
            while (iarray_iter_write_block_has_next(I)) {
                iarray_iter_write_block_next(I, NULL, 0);
                int64_t part_size = val.block_size;  // 1-dim vector
                for (int64_t i = 0; i < part_size; ++i) {
                    ((double *) val.pointer)[i] = incx * (double) (i + val.nblock * part_size);
                }
            }
            iarray_iter_write_block_free(I);
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf("Time for computing and filling X values via partition iterator: %.3g s, %.1f MB/s\n",
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

    if (INA_SUCCEED(ina_opt_isset("p")) && _iarray_file_exists(mat_y.id)) {
        INA_STOPWATCH_START(w);
        INA_MUST_SUCCEED(iarray_from_file(ctx, &mat_y, &con_y));
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
                double value = _poly(incx * (double) val.elem_flat_index);
                memcpy(val.pointer, &value, sizeof(double));
            }
            iarray_iter_write_free(I);
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
            iarray_iter_write_block_new(ctx, &I, con_y, NULL, &val, false);
            double incx = XMAX / NELEM;
            while (iarray_iter_write_block_has_next(I)) {
                iarray_iter_write_block_next(I, NULL, 0);
                int64_t part_size = val.block_size;
                for (int64_t i = 0; i < part_size; ++i) {
                    ((double *) val.pointer)[i] = _poly(incx * (double) (i + val.nblock * part_size));
                }
            }
            iarray_iter_write_block_free(I);
            INA_STOPWATCH_STOP(w);
            INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
            printf(
                "Time for computing and filling Y values via partition iterator: %.3g s, %.1f MB/s\n",
                elapsed_sec, buffer_len / (elapsed_sec * _IARRAY_SIZE_MB));
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
    iarray_expression_t *e;
    iarray_expr_new(ctx, &e);
    iarray_expr_bind(e, "x", con_x);
    iarray_expr_compile(e, "(x - 1.35) * (x - 4.45) * (x - 8.5)");

    iarray_container_t *con_out;
    INA_MUST_SUCCEED(iarray_container_new(ctx, &dtshape, &mat_out, flags, &con_out));

    INA_STOPWATCH_START(w);
    ina_rc_t errcode = iarray_eval(e, con_out);
    if (errcode != INA_SUCCESS) {
        printf("Error during evaluation.  Giving up...\n");
        return -1;
    }
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    iarray_container_info(con_out, &nbytes, &cbytes);
    printf("\n");
    printf("Time for computing and filling OUT values using iarray (%s):  %.3g s, %.1f MB/s\n",
           eval_method, elapsed_sec, nbytes / (elapsed_sec * _IARRAY_SIZE_MB));
    nbytes_mb = ((double)nbytes / (double)_IARRAY_SIZE_MB);
    cbytes_mb = ((double)cbytes / (double)_IARRAY_SIZE_MB);
    printf("Compression for OUT values: %.1f MB -> %.1f MB (%.1fx)\n",
           nbytes_mb, cbytes_mb, (1.*nbytes)/cbytes);

    iarray_expr_free(ctx, &e);

    printf("Checking that the outcome of the expression is correct...");
    fflush(stdout);
    INA_STOPWATCH_START(w);
    if (iarray_container_almost_equal(con_y, con_out, 1e-06) == INA_ERR_FAILED) {
        printf(" No!\n");
        return 1;
    }
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    printf(" Yes!\n");
    printf("Time for checking that two iarrays are equal:  %.3g s, %.1f MB/s\n",
           elapsed_sec, (nbytes * 2) / (elapsed_sec * _IARRAY_SIZE_MB));

    iarray_container_free(ctx, &con_x);
    iarray_container_free(ctx, &con_y);
    iarray_container_free(ctx, &con_out);

    iarray_context_free(&ctx);

    if (x_allocated) ina_mem_free(x);
    if (y_allocated) ina_mem_free(y);

    INA_STOPWATCH_FREE(&w);

    return 0;
}
