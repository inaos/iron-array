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
#include "../../../../../../usr/local/Cellar/gcc/9.3.0_1/include/c++/9.3.0/bits/codecvt.h"

int main(void)
{
    ina_rc_t rc;

    printf("Starting iarray...\n");
    iarray_init();

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_new(&cfg, &ctx);

    iarray_container_t *c_x, *c_out;

    // Create c_x container
    int8_t xndim = 3;
    int64_t xshape[] = {100, 100, 100};

    iarray_dtshape_t xdtshape;
    xdtshape.ndim = xndim;
    xdtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    for (int i = 0; i < xdtshape.ndim; ++i) {
        xdtshape.shape[i] = xshape[i];
    }

    iarray_storage_t store;
    store.backend = IARRAY_STORAGE_BLOSC;
    store.enforce_frame = false;
    store.filename = NULL;
    
    if (INA_FAILED(iarray_partition_advice(ctx, &xdtshape, result, 16 * 1024, 128 * 1024))) {
        printf("Error in getting advice for pshape: %s\n", ina_err_strerror(ina_err_get_rc()));
        exit(1);
    }
    printf("pshape: %d %d %d\n", (int)xdtshape.pshape[0], (int)xdtshape.pshape[1], (int)xdtshape.pshape[2]);

    printf("Initializing c_x container...\n");
    printf("- c_x shape: ");
    for (int i = 0; i < xdtshape.ndim; ++i) {
        printf("%d ", (int)xdtshape.shape[i]);
    }
    printf("\n");

    IARRAY_FAIL_IF_ERROR(iarray_fill_double(ctx, &xdtshape, 3.14, &store, 0, &c_x));

    // Create out container (empty)
    int8_t outndim = 3;
    int64_t start[] = {10, 20, 30};
    int64_t stop[] = {40, 21, 80};
    int64_t outpshape[] = {5, 1, 20};

    printf("Defining start and stop for slicing...\n");
    printf("- start: ");
    for (int i = 0; i < outndim; ++i) {
        printf("%d ", (int)start[i]);
    }
    printf("\n");
    printf("- stop: ");
    for (int i = 0; i < outndim; ++i) {
        printf("%d ", (int)stop[i]);
    }
    printf("\n");

    // Slicing c_x into c_out
    printf("Slicing c_x into c_out container...\n");
    IARRAY_FAIL_IF_ERROR(iarray_get_slice(ctx, c_x, start, stop, false, 0, &c_out, NULL));
    iarray_dtshape_t out_dtshape;
    IARRAY_FAIL_IF_ERROR(iarray_get_dtshape(ctx, c_out, &out_dtshape));

    printf("- c_out shape: ");
    for (int i = 0; i < out_dtshape.ndim; ++i) {
        printf("%d ", (int) out_dtshape.shape[i]);
    }
    printf("\n");

    //Squeezing c_out
    printf("Squeezing c_out...\n");
    IARRAY_FAIL_IF_ERROR(iarray_squeeze(ctx, c_out));
    IARRAY_FAIL_IF_ERROR(iarray_get_dtshape(ctx, c_out, &out_dtshape));

    printf("- c_out shape: ");
    for (int i = 0; i < out_dtshape.ndim; ++i) {
        printf("%d ", (int)out_dtshape.shape[i]);
    }
    printf("\n");

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
    rc = ina_err_get_rc();
    cleanup:
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_out);
    iarray_context_free(&ctx);

    printf("Destroying iarray...\n");
    iarray_destroy();
    return rc;
}