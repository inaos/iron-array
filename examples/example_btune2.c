#include <libiarray/iarray.h>
#include <inttypes.h>

#define SHAPE (100 * 1000 * 1000)
#define CHUNK_SHAPE (2 * 1000 * 1000)
#define BLOCK_SHAPE (20 * 1000)


double fexpr(double x) {
    return (x - 2) * (x - 1) * (x - 3);
}


int main(void) {
    iarray_init();
    ina_stopwatch_t *w;

    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    char *expr = "(x - 2) * (x - 1) * (x - 3)";

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;

    cfg.max_num_threads = 4;
    iarray_context_new(&cfg, &ctx);

    int64_t shape[] = {SHAPE};
    int8_t ndim = 1;
    int8_t typesize = sizeof(double);

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape.ndim = ndim;

    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }

    int32_t xchunkshape[] = {CHUNK_SHAPE};
    int32_t xblockshape[] = {BLOCK_SHAPE};

    iarray_storage_t xstorage;
    xstorage.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    int32_t outchunkshape[] = {CHUNK_SHAPE};
    int32_t outblockshape[] = {BLOCK_SHAPE};

    iarray_storage_t outstorage;
    outstorage.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        outstorage.chunkshape[i] = outchunkshape[i];
        outstorage.blockshape[i] = outblockshape[i];
    }

    iarray_container_t *c_x;
    iarray_linspace(ctx, &dtshape, 0, 1, &xstorage, &c_x);

    int64_t c_nbytes;
    int64_t c_cbytes;
    IARRAY_RETURN_IF_FAILED(iarray_container_info(c_x, &c_nbytes, &c_cbytes));
    printf("** input:\n");
    printf("-   compression level: %d\n", cfg.compression_level);
    printf("-   nbytes: %" PRId64 " x\n", c_nbytes);
    printf("-   Ratio: %8.2f x\n", (double) c_nbytes / (double)c_cbytes);

    iarray_expression_t *e;
    iarray_expr_new(ctx, dtshape.dtype, &e);
    iarray_expr_bind(e, "x", c_x);

    iarray_expr_bind_out_properties(e, &dtshape, &outstorage);
    iarray_expr_compile(e, expr);

    iarray_container_t *c_out;
    int nrep = 1;
    INA_STOPWATCH_START(w);
    for (int i = 0; i < nrep; ++i) {
        IARRAY_RETURN_IF_FAILED(iarray_eval(e, &c_out));
    }
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("** output:\n");
    IARRAY_RETURN_IF_FAILED(iarray_container_info(c_out, &c_nbytes, &c_cbytes));
    printf("-   nbytes: %" PRId64 " x\n", c_nbytes);
    printf("-   Ratio: %8.2f x\n", (double) c_nbytes / (double)c_cbytes);

    int64_t nbytes = nelem * typesize;
    printf("Time to eval expression: %.3g s, %.1f GB/s\n",
           elapsed_sec, (double)nbytes * nrep / (elapsed_sec * (1u << 20u)));

    iarray_expr_free(ctx, &e);
    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);
    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);

    return 0;
}
