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
    int64_t shape[] = {10, 10};
    int64_t pshape[] = {2, 2};
    int64_t bshape[] = {2, 10};

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


    iarray_iter_write_t *iter_w;
    iarray_iter_write_value_t val_w;
    iarray_iter_write_new(ctx, &iter_w, cont, &val_w);

    while (iarray_iter_write_has_next(iter_w)) {
        iarray_iter_write_next(iter_w);
        ((double *) val_w.pointer)[0] = (double) val_w.elem_flat_index;
    }
    iarray_iter_write_free(iter_w);


    iarray_iter_read_block_t *iter;
    iarray_iter_read_block_value_t val;
    iarray_iter_read_block_new(ctx, &iter, cont, bshape, &val, NULL, 0);
    while (iarray_iter_read_block_has_next(iter)) {
        iarray_iter_read_block_next(iter);
        for (int i = 0; i < val.block_size; ++i) {
            double value = ((double *) val.pointer)[i];
            printf("%f - ", value);
        }
        printf("\n");
    }
    iarray_iter_read_block_free(iter);

    return EXIT_SUCCESS;
}
