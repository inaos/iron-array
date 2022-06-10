/*
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <libiarray/iarray.h>

int main(void)
{
    printf("Starting iarray...\n");
    iarray_init();

    iarray_context_t *ctx;
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_new(&cfg, &ctx);

    // Create c_x container
    int8_t xndim = 3;
    int64_t xshape[] = {10, 10, 10};

    iarray_dtshape_t xdtshape;
    xdtshape.ndim = xndim;
    xdtshape.dtype = IARRAY_DATA_TYPE_DOUBLE;
    for (int i = 0; i < xdtshape.ndim; ++i) {
        xdtshape.shape[i] = xshape[i];
    }

    iarray_storage_t store;
    store.contiguous = false;
    store.urlpath = NULL;
    
    if (INA_FAILED(iarray_partition_advice(ctx, &xdtshape, &store, 0, 0, 0, 8 * 1024))) {
        printf("Error in getting advice for chunkshape: %s\n", ina_err_strerror(ina_err_get_rc()));
        exit(1);
    }
    int64_t dataitems = 1;
    for (int i = 0; i < xndim; ++i) {
        dataitems *= xshape[i];
    }
    int64_t datasize = dataitems * xdtshape.dtype_size;
    double *data = malloc(datasize);
    for (int i = 0; i < dataitems; ++i) {
        data[i] = (double) i;
    }
    iarray_container_t *c;
    IARRAY_RETURN_IF_FAILED(iarray_from_buffer(ctx, &xdtshape, data, datasize, &store, &c));
    free(data);

    int64_t sel0[] = {3, 1, 2};
    int64_t sel1[] = {2, 5};
    int64_t sel2[] = {3, 3, 3, 9,3, 1, 0};
    int64_t *selection[] = {sel0, sel1, sel2};
    int64_t selection_size[] = {sizeof(sel0)/sizeof(int64_t), sizeof(sel1)/(sizeof(int64_t)), sizeof(sel2)/(sizeof(int64_t))};
    int64_t *buffershape = selection_size;
    int64_t nitems = 1;
    for (int i = 0; i < xdtshape.ndim; ++i) {
        nitems *= buffershape[i];
    }
    int64_t buffersize = nitems * xdtshape.dtype_size;
    double *buffer = calloc(nitems, xdtshape.dtype_size);

    iarray_set_orthogonal_selection(ctx, c, selection, selection_size, buffer, buffershape, buffersize);
    iarray_get_orthogonal_selection(ctx, c, selection, selection_size, buffer, buffershape, buffersize);

    printf("Results: \n");
    for (int i = 0; i < nitems; ++i) {
        if (i % buffershape[1] == 0) {
            printf("\n");
        }
        printf(" %f ", buffer[i]);
    }
    printf("\n");
    free(buffer);

    iarray_container_free(ctx, &c);
    iarray_context_free(&ctx);

    printf("Destroying iarray...\n");
    iarray_destroy();
    return INA_SUCCESS;
}
