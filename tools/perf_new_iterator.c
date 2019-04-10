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
    int64_t pshape[] = {10, 10};
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


    iarray_iter_write_block2_t *iter_w;
    iarray_iter_write_block2_value_t val_w;
    iarray_iter_write_block2_new(ctx, &iter_w, cont, NULL, &val_w);

    int64_t n = 0;
    while (iarray_iter_write_block2_has_next(iter_w)) {
        iarray_iter_write_block2_next(iter_w);
        int64_t size = 1;
        for (int i = 0; i < ndim; ++i) {
            size *= val_w.part_shape[i];
        }
        for (int i = 0; i < size; ++i) {
            ((double *) val_w.pointer)[i] = (double) i + n;
        }
        n += size;
    }
    iarray_iter_write_block2_free(iter_w);


    iarray_iter_read_block2_t *iter;
    iarray_iter_read_block_value_t val;
    iarray_iter_read_block2_new(ctx, &iter, cont, blockshape, &val);
    while (iarray_iter_read_block2_has_next(iter)) {
        iarray_iter_read_block2_next(iter);
        int64_t size = 1;
        for (int i = 0; i < ndim; ++i) {
            size *= val.block_shape[i];
        }
        for (int i = 0; i < size; ++i) {
            printf("%f\n", ((double *) val.pointer)[i]);
        }
    }
    iarray_iter_read_block2_free(iter);

    return EXIT_SUCCESS;
}
