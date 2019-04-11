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


int main()
{
    int8_t ndim = 2;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int64_t shape[] = {100, 100};
    int64_t pshape[] = {0, 0};
    int64_t blockshape[] = {10, 10};

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_t *ctx;
    iarray_context_new(&cfg, &ctx);

    iarray_dtshape_t dtshape;
    dtshape.ndim = ndim;
    dtshape.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        dtshape.pshape[i] = pshape[i];
    }
    iarray_container_t *cont;
    iarray_container_new(ctx, &dtshape, NULL, 0, &cont);


    iarray_iter_write_block_t *iter_w;
    iarray_iter_write_block_value_t val_w;
    iarray_iter_write_block_new(ctx, &iter_w, cont, NULL, &val_w);
    int64_t n = 0;
    while (iarray_iter_write_block_has_next(iter_w)) {
        iarray_iter_write_block_next(iter_w);
        for (int i = 0; i < val_w.block_size; ++i) {
            ((double *) val_w.pointer)[i] = (double) i + n;
        }
        n += val_w.block_size;
    }
    iarray_iter_write_block_free(iter_w);


    iarray_iter_read_t *iter;
    iarray_iter_read_value_t val;
    iarray_iter_read_new(ctx, &iter, cont, &val);
    while (iarray_iter_read_has_next(iter)) {
        iarray_iter_read_next(iter);
        printf("%f\n", ((double *) val.pointer)[0]);
    }
    iarray_iter_read_free(iter);

    return EXIT_SUCCESS;
}
