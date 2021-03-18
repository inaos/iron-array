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


int main(void)
{
    iarray_init();

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {12, 12, 12};
    int64_t cshape[] = {6, 9, 6};
    int64_t bshape[] = {3, 3, 3};
    int8_t axis = 2;

    int64_t dest_cshape[] = {6, 6};
    int64_t dest_bshape[] = {3, 3};
    bool dest_frame = false;
    char *dest_urlpath = NULL;

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.btune = true;

    iarray_context_t *ctx;
    IARRAY_RETURN_IF_FAILED(iarray_context_new(&cfg, &ctx));

    // Create dtshape
    iarray_dtshape_t dtshape;

    dtshape.dtype = dtype;
    dtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        size *= shape[i];
    }

    iarray_storage_t storage = {0};
    storage.backend = IARRAY_STORAGE_BLOSC;
    for (int i = 0; i < ndim; ++i) {
        if (cshape != NULL) {
            storage.chunkshape[i] = i == axis ? shape[i] : 1;
            storage.blockshape[i] = i == axis ? shape[i] : 1;
        }
    }

    iarray_container_t *c_x;
    IARRAY_RETURN_IF_FAILED(iarray_container_new(ctx, &dtshape, &storage, 0, &c_x));


    iarray_iter_write_block_t *iter;
    iarray_iter_write_block_value_t iter_value;
    IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_new(ctx, &iter, c_x, storage.chunkshape,
                                                        &iter_value, false));
    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_next(iter, NULL, 0));
        for (int i = 0; i < iter_value.block_size; ++i) {
            switch (c_x->dtshape->dtype) {
                case IARRAY_DATA_TYPE_DOUBLE:
                    ((double *) iter_value.block_pointer)[i] = (double) i;
                    break;
                case IARRAY_DATA_TYPE_FLOAT:
                    ((float *) iter_value.block_pointer)[i] = (float) i;
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Invalid dtype");
                    return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
            }
        }
    }
    iarray_iter_write_block_free(&iter);
    IARRAY_ITER_FINISH();


    storage.backend = cshape == NULL ? IARRAY_STORAGE_PLAINBUFFER : IARRAY_STORAGE_BLOSC;
    for (int i = 0; i < ndim; ++i) {
        if (cshape != NULL) {
            storage.chunkshape[i] = cshape[i];
            storage.blockshape[i] = bshape[i];
        }
    }

    iarray_container_t *c_y;
    IARRAY_RETURN_IF_FAILED(iarray_copy(ctx, c_x, false, &storage, 0, &c_y));

    iarray_storage_t dest_storage = {0};
    dest_storage.backend = IARRAY_STORAGE_BLOSC;
    dest_storage.enforce_frame = dest_frame;
    dest_storage.urlpath = dest_urlpath;
    for (int i = 0; i < ndim - 1; ++i) {
        dest_storage.blockshape[i] = dest_bshape[i];
        dest_storage.chunkshape[i] = dest_cshape[i];
    }

    iarray_container_t *c_z;
    IARRAY_RETURN_IF_FAILED(iarray_reduce(ctx, c_y, IARRAY_REDUCE_SUM, axis, &dest_storage, &c_z));

    int64_t buffer_nitems = c_z->catarr->nitems;
    int64_t buffer_size = buffer_nitems * c_z->catarr->itemsize;
    uint8_t *buffer = malloc(buffer_size);

    blosc2_schunk *sc = c_z->catarr->sc;
    for (int i = 0; i < sc->nchunks; ++i) {
        uint8_t *chunk;
        bool needs_free;
        blosc2_schunk_get_chunk(sc, i, &chunk, &needs_free);
        int32_t nbytes, cbytes, blocksize;
        blosc2_cbuffer_sizes(chunk, &nbytes, &cbytes, &blocksize);
        printf("blocksize: %d, nbytes: %d\n", blocksize, nbytes);
    }

    IARRAY_RETURN_IF_FAILED(iarray_to_buffer(ctx, c_z, buffer, buffer_size));

    double val = shape[axis] * (shape[axis] - 1.) / 2;
    for (int i = 0; i < buffer_nitems; ++i) {
        // printf("%d: %f - %f\n", i, ((double *) buffer)[i], val);
        switch (c_z->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                INA_TEST_ASSERT_EQUAL_FLOATING(((double *) buffer)[i], val);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                INA_TEST_ASSERT_EQUAL_FLOATING(((float *) buffer)[i], val);
                break;
            default:
                IARRAY_TRACE1(iarray.error, "Invalid dtype");
                return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
        }
    }

    iarray_container_free(ctx, &c_z);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_x);

    iarray_context_free(&ctx);


    return INA_SUCCESS;
}
