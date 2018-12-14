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

#define NCHUNKS  100
#define NITEMS_CHUNK (200 * 1000)  // fits well in modern L3 caches
#define NELEM (NCHUNKS * NITEMS_CHUNK)  // multiple of NITEMS_CHUNKS for now
#define PART_SIZE NITEMS_CHUNK
#define NTHREADS 1

static double _poly(const double x)
{
    return (x - 1.35) * (x - 4.45) * (x - 8.5);
}

// Fill X values in regular array
static int _fill_x(double* x)
{
    double incx = 10. / NELEM;

    /* Fill even values between 0 and 10 */
    for (int i = 0; i<NELEM; i++) {
        x[i] = incx * i;
    }
    return 0;
}

// Compute and fill Y values in regular array
static void _compute_y(const double* x, double* y)
{
    for (int i = 0; i<NELEM; i++) {
        y[i] = _poly(x[i]);
    }
}

static void ina_cleanup_handler(int error, int *exitcode)
{
    iarray_destroy();
}

/*
 * Check if a file exist using fopen() function
 * return 1 if the file exist otherwise return 0
 */
bool cfileexists(const char * filename){
    /* try to open file to read */
    FILE *file;
    if ((file = fopen(filename, "r"))) {
        fclose(file);
        return true;
    }
    return false;
}

static double *x = NULL;
static double *y = NULL;

int main(int argc, char** argv)
{
    ina_stopwatch_t *w;
    iarray_context_t *ctx = NULL;
    const char *mat_x_name = NULL;
    const char *mat_y_name = NULL;
    const char *mat_out_name = NULL;
    int eval_flag = 0;
    const char *eval_method = NULL;

    INA_OPTS(opt,
        INA_OPT_INT("f", "eval-flag", 1, "EVAL_BLOCK = 1, EVAL_CHUNK = 2"),
        INA_OPT_FLAG("p", "persistence", "Use persistent containers")
    );

    if (!INA_SUCCEED(ina_app_init(argc, argv, opt))) {
        return EXIT_FAILURE;
    }
    ina_set_cleanup_handler(ina_cleanup_handler);

    INA_MUST_SUCCEED(ina_opt_get_int("f", &eval_flag));
    if (INA_SUCCEED(ina_opt_isset("p"))) {
        mat_x_name = "mat_x.b2frame";
        mat_y_name = "mat_y.b2frame";
        mat_out_name = "mat_out.b2frame";
    }
    iarray_store_properties_t mat_x = {.id = mat_x_name};
    iarray_store_properties_t mat_y = {.id = mat_y_name};
    iarray_store_properties_t mat_out = {.id = mat_out_name};

    INA_MUST_SUCCEED(iarray_init());

    iarray_config_t config = IARRAY_CONFIG_DEFAULTS;
    config.compression_codec = IARRAY_COMPRESSION_BLOSCLZ;
    config.compression_level = 9;
    config.max_num_threads = NTHREADS;
    if (eval_flag == 2) {
        eval_method = "EVAL_CHUNK";
        config.flags = IARRAY_EXPR_EVAL_CHUNK;
    }
    else {
        eval_method = "EVAL_BLOCK";
        config.flags = IARRAY_EXPR_EVAL_BLOCK;
    }
    config.blocksize = 16 * _IARRAY_SIZE_KB;  // 16 KB seems optimal for evaluating expressions

    INA_MUST_SUCCEED(iarray_context_new(&config, &ctx));

    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    size_t buffer_len = sizeof(double)*NELEM;
    x = (double*)ina_mem_alloc(buffer_len);
    y = (double*)ina_mem_alloc(buffer_len);

    // Fill the plain x operand
    _fill_x(x);

    iarray_dtshape_t shape;
    shape.ndim = 1;
    shape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    shape.shape[0] = NELEM;
    shape.partshape[0] = PART_SIZE;

    iarray_container_t *con_x;
    iarray_container_t *con_y;

    // Compute the plain y vector
    INA_STOPWATCH_START(w);
    _compute_y(x, y);
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time for computing and filling Y values: %.3g s, %.1f MB/s\n",
            elapsed_sec, buffer_len/(elapsed_sec*_IARRAY_SIZE_MB));
    // To prevent the optimizer going too smart and removing 'dead' code
    int retcode = y[0] > y[1];

    int flags = INA_SUCCEED(ina_opt_isset("p"))? IARRAY_CONTAINER_PERSIST : 0;
    if (INA_SUCCEED(ina_opt_isset("p")) && cfileexists(mat_x.id)) {
        INA_MUST_SUCCEED(iarray_from_file(ctx, &mat_x, flags, &con_x));
    }
    else {
        INA_MUST_SUCCEED(iarray_from_buffer(ctx, &shape, x, buffer_len, &mat_x, flags, &con_x));
    }
    INA_STOPWATCH_START(w);
    if (INA_SUCCEED(ina_opt_isset("p")) && cfileexists(mat_y.id)) {
        INA_MUST_SUCCEED(iarray_from_file(ctx, &mat_y, flags, &con_y));
    }
    else {
        INA_MUST_SUCCEED(iarray_from_buffer(ctx, &shape, y, buffer_len, &mat_y, flags, &con_y));
    }
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    uint64_t nbytes = 0;
    uint64_t cbytes = 0;
    double nbytes_mb = 0;
    double cbytes_mb = 0;

    iarray_container_info(con_x, &nbytes, &cbytes);
    printf("Time for compressing Y values: %.3g s, %.1f MB/s\n",
            elapsed_sec, nbytes/(elapsed_sec*_IARRAY_SIZE_MB));
    nbytes_mb = ((double)nbytes / _IARRAY_SIZE_MB);
    cbytes_mb = ((double)cbytes / _IARRAY_SIZE_MB);
    printf("Compression for Y values: %.1f MB -> %.1f MB (%.1fx)\n",
            nbytes_mb, cbytes_mb, (1.*nbytes)/cbytes);

    // Check IronArray performance
    iarray_expression_t *e;
    iarray_expr_new(ctx, &e);
    iarray_expr_bind(e, "x", con_x);
    iarray_expr_compile(e, "(x - 1.35) * (x - 4.45) * (x - 8.5)");

    iarray_container_t *con_out;
    INA_MUST_SUCCEED(iarray_container_new(ctx, &shape, &mat_out, flags, &con_out));

    INA_STOPWATCH_START(w);
    iarray_eval(e, con_out);
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

    INA_MUST_SUCCEED(iarray_container_almost_equal(con_y, con_out, 1e-06));

    iarray_expr_free(ctx, &e);

    iarray_container_free(ctx, &con_x);
    iarray_container_free(ctx, &con_y);
    iarray_container_free(ctx, &con_out);

    iarray_context_free(&ctx);

    ina_mem_free(x);
    ina_mem_free(y);

    INA_STOPWATCH_FREE(&w);

    return retcode;
}
