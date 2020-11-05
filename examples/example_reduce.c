/*
 * Copyright INAOS GmbH, Thalwil, 2019.
 * Copyright Francesc Alted, 2019.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of INAOS GmbH
 * and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <libiarray/iarray.h>
#include <math.h>


int64_t min(double *min, double *v, int64_t vlen) {
    *min = INFINITY;
    for (int i = 0; i < vlen; ++i) {
        if (v[i] < *min)
            *min = v[i];
    }
    return 0;
}


double max(double *v, int64_t vlen) {
    double max = -INFINITY;
    for (int i = 0; i < vlen; ++i) {
        if (v[i] > max)
            max = v[i];
    }
    return max;
}


double sum(double *v, int64_t vlen) {
    double sum = 0;
    for (int i = 0; i < vlen; ++i) {
        sum += v[i];
    }
    return sum;
}


double prod(double *v, int64_t vlen) {
    double prod = 1;
    for (int i = 0; i < vlen; ++i) {
        prod *= v[i];
    }
    return prod;
}


double mean(double *v, int64_t vlen) {
    double mean = 0;
    for (int i = 0; i < vlen; ++i) {
        mean += v[i];
    }
    mean /= vlen;
    return mean;
}


double std(double *v, int64_t vlen) {
    double mean = 0;
    for (int i = 0; i < vlen; ++i) {
        mean += v[i];
    }
    mean /= vlen;

    double std = 0;
    for (int i = 0; i < vlen; ++i) {
        std += pow((v[i] - mean),  2);
    }
    std /= vlen;
    std = sqrt(std);
    return std;
}


int main(void) {
    iarray_init();
    ina_stopwatch_t *w;

    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);


    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_level = 0;
    cfg.eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;

    cfg.max_num_threads = 1;
    iarray_context_new(&cfg, &ctx);


    int64_t shape[] = {200, 100};
    int8_t ndim = 2;
    int8_t typesize = sizeof(double);

    iarray_dtshape_t dtshape;
    dtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    dtshape.ndim = ndim;

    int64_t nelem = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        nelem *= shape[i];
    }

    int32_t xchunkshape[] = {30, 15};
    int32_t xblockshape[] = {12, 7};

    iarray_storage_t xstorage;
    xstorage.backend = IARRAY_STORAGE_BLOSC;
    xstorage.enforce_frame = false;
    xstorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        xstorage.chunkshape[i] = xchunkshape[i];
        xstorage.blockshape[i] = xblockshape[i];
    }

    iarray_container_t *c_x;
    IARRAY_RETURN_IF_FAILED(iarray_arange(ctx, &dtshape, 0, nelem, 1, &xstorage, 0, &c_x));

    int32_t outchunkshape[] = {40};
    int32_t outblockshape[] = {21};

    iarray_storage_t outstorage;
    outstorage.backend = IARRAY_STORAGE_BLOSC;
    outstorage.enforce_frame = false;
    outstorage.filename = NULL;
    for (int i = 0; i < ndim; ++i) {
        outstorage.chunkshape[i] = outchunkshape[i];
        outstorage.blockshape[i] = outblockshape[i];
    }

    iarray_container_t *c_out;
    IARRAY_RETURN_IF_FAILED(iarray_reduce_double(ctx, c_x, &mean, 0, &outstorage, &c_out));

    iarray_container_free(ctx, &c_out);
    iarray_container_free(ctx, &c_x);

    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);

    return 0;
}
