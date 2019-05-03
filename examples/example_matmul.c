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
#include <unistd.h>

int main()
{
    int8_t ndim = 2;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int64_t shape[] = {2000, 2000};
    int64_t size = 2000 * 2000;
    int64_t pshape[] = {0, 0};
    int64_t *bshape = NULL;

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
    iarray_container_t *c_x;
    iarray_linspace(ctx, &dtshape, size, 0, 1, NULL, 0, &c_x);

    iarray_container_t *c_y;
    iarray_linspace(ctx, &dtshape, size, 0, 1, NULL, 0, &c_y);

    iarray_container_t *c_z;
    iarray_container_new(ctx, &dtshape, NULL, 0, &c_z);

    iarray_linalg_matmul(ctx, c_x, c_y ,c_z, bshape, bshape, IARRAY_OPERATOR_GENERAL);

    for (int i = 0; i <10e6; ++i) {

    }

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_z);
    iarray_context_free(&ctx);

    return EXIT_SUCCESS;
}
