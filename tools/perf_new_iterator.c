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
    int64_t size = 10000;

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
    INA_MUST_SUCCEED(iarray_linspace(ctx, &dtshape, size, 0, 1, NULL, 0, &cont));
    int64_t buffer_size = 1;
    for (int i = 0; i < ndim; ++i) {
        buffer_size *= shape[i];
    }

    double *buffer = (double *) ina_mem_alloc(buffer_size * sizeof(double));
    INA_MUST_SUCCEED(iarray_to_buffer(ctx, cont, buffer, (size_t) buffer_size));


    iarray_iter_read_block2_t *iter;
    iarray_iter_read_block_value_t val;
    iarray_iter_read_block2_new(ctx, &iter, cont, blockshape, &val);

    while (iarray_iter_read_block2_has_next(iter)) {
        iarray_iter_read_block2_next(iter);
    }

    iarray_iter_read_block2_free(iter);

    ina_mem_free(buffer);

    return EXIT_SUCCESS;
}
