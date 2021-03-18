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

int main(void)
{
    iarray_init();

    int8_t ndim = 2;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int64_t shape[] = {3523, 4246};
    int64_t cshape[] = {512, 512};
    int64_t bshape[] = {128, 128};

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.btune = true;

    iarray_dtshape_t dtshape;
    dtshape.ndim = ndim;
    dtshape.dtype = dtype;

    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        size *= shape[i];
    }
    int64_t nbytes = size * sizeof(double);

    iarray_storage_t store;
    store.backend = IARRAY_STORAGE_BLOSC;
    store.enforce_frame = false;
    store.urlpath = NULL;
    for (int i = 0; i < ndim; ++i) {
        store.chunkshape[i] = cshape[i];
        store.blockshape[i] = bshape[i];
    }

    double *src = malloc(nbytes);
    for (int i = 0; i < size; ++i) {
        src[i] = (double) i;
    }
    iarray_compression_favor_t favors[3] = {
            IARRAY_COMPRESSION_FAVOR_SPEED,
            IARRAY_COMPRESSION_FAVOR_BALANCE,
            IARRAY_COMPRESSION_FAVOR_CRATIO
    };
    char* favors_names[] = {"Speed", "Balance", "Cratio"};

    double *dest = malloc(nbytes);

    ina_stopwatch_t *w;

    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);

    for (int f = 0; f < 3; ++f) {
        printf("Favor %s\n", favors_names[f]);

        cfg.compression_favor = favors[f];
        iarray_context_t *ctx;
        IARRAY_RETURN_IF_FAILED(iarray_context_new(&cfg, &ctx));

        INA_STOPWATCH_START(w);
        iarray_container_t *cont;
        IARRAY_RETURN_IF_FAILED(iarray_from_buffer(ctx, &dtshape, src, nbytes, &store, 0, &cont));
        INA_STOPWATCH_STOP(w);

        int64_t c_nbytes;
        int64_t c_cbytes;
        IARRAY_RETURN_IF_FAILED(iarray_container_info(cont, &c_nbytes, &c_cbytes));
        printf("- Ratio: %8.2f x\n", (double) c_nbytes / c_cbytes);

        INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));
        printf("- Time: %9.2f s\n", elapsed_sec);

        IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, cont, dest, nbytes));

        for (int i = 0; i < size; ++i) {
            if (src[i] != dest[i]) {
                printf("%d: %f != %f\n", i, src[i], dest[i]);
            }
        }
        iarray_container_free(ctx, &cont);
        iarray_context_free(&ctx);
    }

    free(src);
    free(dest);

    INA_STOPWATCH_FREE(&w);

    iarray_destroy();

    return INA_SUCCESS;
}
