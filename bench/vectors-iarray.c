//
// Created by Francesc Alted on 6/11/2018.
//

/*
  Example program demonstrating how to execute an expression with super-chunks as operands.
  This is the version for using frames (either in-memory or on-disk) backing the super-chunks.

  To compile this program:

  $ gcc -O3 vectors-iarray.c -o vectors-iarray -lblosc

  To run:

  $ ./vectors-iarray memory
  ...
  $ ./vectors-iarray disk
  ...

*/

#include <libiarray/iarray.h>
#include <iarray_private.h>

#define NCHUNKS  100
#define NITEMS_CHUNK (200 * 1000)  // fits well in modern L3 caches
#define NELEM (NCHUNKS * NITEMS_CHUNK)  // multiple of NITEMS_CHUNKS for now
#define PART_SIZE 1000
#define NTHREADS 1

static double poly(const double x)
{
    return (x - 1.35) * (x - 4.45) * (x - 8.5);
}

/* FIXME: how to handle
void fill_buffer(double* x, int nchunk)
{
    double incx = 10./NELEM;

    for (int i=0; i<NITEMS_CHUNK; i++) {
        x[i] = incx*(nchunk * NITEMS_CHUNK + i);
    }
}

void fill_cta_x(caterva_array* cta_x, const size_t isize)
{
    static double buffer_x[NITEMS_CHUNK];

    // Fill with even values between 0 and 10
    for (int nchunk = 0; nchunk<NCHUNKS; nchunk++) {
        fill_buffer(buffer_x, nchunk);
        blosc2_schunk_append_buffer(cta_x->sc, buffer_x, isize);
    }
}*/

// Compute and fill Y values in a buffer
void fill_buffer_y(const double* x, double* y)
{
    for (int i = 0; i<NITEMS_CHUNK; i++) {
        y[i] = poly(x[i]);
    }
}

// Fill X values in regular array
static int fill_x(double* x)
{
    double incx = 10. / NELEM;

    /* Fill even values between 0 and 10 */
    for (int i = 0; i<NELEM; i++) {
        x[i] = incx * i;
    }
    return 0;
}

// Compute and fill Y values in regular array
static void compute_y(const double* x, double* y)
{
    for (int i = 0; i<NELEM; i++) {
        y[i] = poly(x[i]);
    }
}

// STONI: here we should compare the original C buffers to the exported array-containers, maybe after 
// -> this should rather be a test than a benchmark
// ...
// Check that two super-chunks with the same partitions are equal
/*int test_schunks_equal(blosc2_schunk* sc1, blosc2_schunk* sc2) {
    size_t chunksize = (size_t)sc1->chunksize;
    int nitems_in_chunk = (int)chunksize / sc1->typesize;
    double *buffer_sc1 = malloc(chunksize);
    double *buffer_sc2 = malloc(chunksize);
    for (int nchunk=0; nchunk < sc1->nchunks; nchunk++) {
        int dsize = blosc2_schunk_decompress_chunk(sc1, nchunk, buffer_sc1, chunksize);
        if (dsize < 0) {
            fprintf(stderr, "Error in decompressing a chunk from sc1\n");
            return 0;
        }
        dsize = blosc2_schunk_decompress_chunk(sc2, nchunk, buffer_sc2, chunksize);
        if (dsize < 0) {
            fprintf(stderr, "Error in decompressing a chunk from sc2\n");
            return 0;
        }
        for (int nelem=0; nelem < nitems_in_chunk; nelem++) {
            double vdiff = fabs(buffer_sc1[nelem] - buffer_sc2[nelem]);
            if (vdiff > 1e-6) {
                printf("Values differ in (%d nchunk, %d nelem) (diff: %f)\n", nchunk, nelem, vdiff);
                free(buffer_sc1);
                free(buffer_sc2);
                return 0;
            }
        }
    }
    free(buffer_sc1);
    free(buffer_sc2);
    return 1;
}*/

static void ina_cleanup_handler(int error, int *exitcode)
{
    iarray_destroy();
}

static double *x = NULL;
static double *y = NULL;

// FIXME: pparams.cshape[CATERVA_MAXDIM - 1] = NITEMS_CHUNK;  // FIXME: 1's at the beginning should be removed

int main(int argc, char** argv)
{
    ina_stopwatch_t *w;
    iarray_context_t *ctx = NULL;
    const char *mat_x_name = NULL;
    const char *mat_y_name = NULL;
    const char *mat_out_name = NULL;
    int eval_flag;

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
        mat_x_name = "mat_x";
        mat_y_name = "mat_y";
        mat_out_name = "mat_out";
    }

    INA_MUST_SUCCEED(iarray_init());
    
    iarray_config_t config = IARRAY_CONFIG_DEFAULTS;
    config.compression_codec = IARRAY_COMPRESSION_BLOSCLZ;
    config.compression_level = 9;
    config.max_num_threads = NTHREADS;
    config.flags = eval_flag; // (IARRAY_EXPR_EVAL_BLOCK || IARRAY_EXPR_EVAL_CHUNK)
    config.blocksize = 16 * _IARRAY_SIZE_KB;  // 16 KB seems optimal for evaluating expressions

    INA_MUST_SUCCEED(iarray_context_new(&config, &ctx));

    double elapsed_sec;
    INA_STOPWATCH_NEW(1, -1, &w);
   
    x = (double*)ina_mem_alloc(sizeof(double)*NELEM);
    y = (double*)ina_mem_alloc(sizeof(double)*NELEM);

    // Fill the plain x operand
    fill_x(x);

    iarray_dtshape_t shape;
    shape.ndim = 2;
    shape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    shape.shape[0] = NELEM;
    shape.shape[1] = NELEM;
    shape.partshape[0] = PART_SIZE;
    shape.partshape[1] = PART_SIZE;

    iarray_container_t *con_x;
    iarray_container_t *con_y;


    // FIXME: always fill from C buffer for now!
    


    // Compute the plain y vector
    INA_STOPWATCH_START(w);
    compute_y(x, y);
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    printf("Time for computing and filling Y values: %.3g s, %.1f MB/s\n",
            elapsed_sec, sizeof(y)/(elapsed_sec*_IARRAY_SIZE_MB));
    // To prevent the optimizer going too smart and removing 'dead' code
    int retcode = y[0] > y[1];

    INA_STOPWATCH_START(w);
    // FIXME: how to do this properly?
    /*for (int nchunk = 0; nchunk < sc_x->nchunks; nchunk++) {
        int dsize = blosc2_schunk_decompress_chunk(sc_x, nchunk, buffer_x, isize);
        if (dsize < 0) {
            printf("Decompression error.  Error code: %d\n", dsize);
            return dsize;
        }
        fill_buffer_y(buffer_x, buffer_y);
        blosc2_schunk_append_buffer(sc_y, buffer_y, isize);
    }*/
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    size_t nbytes = 0;
    size_t cbytes = 0;

    iarray_container_info(con_x, &nbytes, &cbytes);
    printf("Time for computing and filling Y values (compressed): %.3g s, %.1f MB/s\n",
            elapsed_sec, nbytes/(elapsed_sec*_IARRAY_SIZE_MB));
    printf("Compression for Y values: %.1f MB -> %.1f MB (%.1fx)\n",
            (nbytes/ _IARRAY_SIZE_MB), (cbytes/ _IARRAY_SIZE_MB),
            (1.*nbytes)/cbytes);

    // Check IronArray performance
    // First for the chunk evaluator

    iarray_expression_t *e;
    iarray_expr_new(ctx, &e);
    iarray_expr_bind(e, "x", con_x);
    iarray_expr_compile(e, "(x - 1.35) * (x - 4.45) * (x - 8.5)");

    iarray_container_t *con_out;
    iarray_container_new(ctx, &shape, mat_out_name, 0, &con_out);

    INA_STOPWATCH_START(w);
    iarray_eval(e, con_out);
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
    iarray_container_info(con_out, &nbytes, &cbytes);
    printf("\n");
    printf("Time for computing and filling OUT values using iarray (chunk eval):  %.3g s, %.1f MB/s\n",
        elapsed_sec, nbytes / (elapsed_sec * _IARRAY_SIZE_MB));
    printf("Compression for OUT values: %.1f MB -> %.1f MB (%.1fx)\n",
            (nbytes/ _IARRAY_SIZE_MB), (cbytes/ _IARRAY_SIZE_MB),
            (1.*nbytes)/cbytes);

    // Check that we are getting the same results than through manual computation
    // FIXME: how to do this
    /*if (!test_schunks_equal(sc_y, sc_out)) {
        return -1;
    }*/

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
