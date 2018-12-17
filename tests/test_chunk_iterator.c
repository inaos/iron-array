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

#include <tests/iarray_test.h>

static ina_rc_t test_chunk_iterator(iarray_context_t *ctx, iarray_data_type_t dtype, size_t type_size, uint8_t ndim,
                                    const uint64_t *shape, const uint64_t *pshape, const uint64_t *ichunk) {

    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        xdtshape.partshape[i] = pshape[i];
    }

    iarray_container_t *c_x;

    iarray_container_new(ctx, &xdtshape, NULL, 0, &c_x);

    // Start Iterator
    iarray_itr_chunk_t *I;
    iarray_itr_chunk_new(ctx, c_x, &I);

    for (iarray_itr_chunk_init(I); !iarray_itr_chunk_finished(I); iarray_itr_chunk_next(I)) {

        iarray_itr_chunk_value_t val;
        iarray_itr_chunk_value(I, &val);

        uint64_t part_size = 1;
        for (int i = 0; i < ndim; ++i) {
            part_size *= val.shape[i];
        }

        uint8_t *data = malloc(part_size * type_size);

        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (uint64_t i = 0; i < part_size; ++i) {
                ( (double *)data)[i] = (double) val.nelem * part_size + i;
            }
        } else {
            for (uint64_t i = 0; i < part_size; ++i) {
                ( (float *)data)[i] = (float) (val.nelem + 1) * i;
            }
        }
        memcpy(val.pointer, &data[0], part_size * type_size);
        free(data);
        //FIXME: Error in some malloc
    }

    iarray_itr_chunk_free(ctx, I);

    // Testing

    uint64_t start[IARRAY_DIMENSION_MAX], stop[IARRAY_DIMENSION_MAX];

    for (int i = 0; i < ndim; ++i) {
        start[i] = ichunk[i] * pshape[i];
        stop[i] = start[i] + pshape[i];
    }
    iarray_dtshape_t ydtshape;

    ydtshape.dtype = dtype;
    ydtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        ydtshape.shape[i] = stop[i] - start[i];
        ydtshape.partshape[i] = ydtshape.shape[i];
    }

    uint64_t nchunk = 0;
    uint64_t  inc = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (shape[i] % pshape[i] == 0) {
            nchunk += ichunk[i] * inc;
            inc *= shape[i] / pshape[i];
        } else {
            nchunk += ichunk[i] * inc;
            inc *= shape[i] / pshape[i] + 1;
        }
    }

    iarray_container_t *c_y;
    iarray_slice(ctx, c_x, start, stop, &ydtshape, NULL, 0, &c_y);


    uint64_t buf_size = 1;
    for (int i = 0; i < ndim; ++i) {
        buf_size *= pshape[i];
    }
    uint8_t *bufdest = malloc(buf_size * type_size);

    iarray_to_buffer(ctx, c_y, bufdest, buf_size);

    if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
        for (uint64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((double *) bufdest)[i], (double) nchunk * buf_size + i);
        }
    } else {
        for (uint64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(((float *) bufdest)[i], (float) nchunk * buf_size + i);
        }
    }


    // Free
    free(bufdest);
    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);

    return INA_SUCCESS;
}

INA_TEST_DATA(chunk_iterator) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(chunk_iterator) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(chunk_iterator) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(chunk_iterator, double_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    uint8_t ndim = 2;
    uint64_t shape[] = {1230, 1423};
    uint64_t pshape[] = {113, 99};
    uint64_t  nchunk[] = {6, 7};

    INA_TEST_ASSERT_SUCCEED(test_chunk_iterator(data->ctx, dtype, type_size, ndim, shape, pshape, nchunk));
}

INA_TEST_FIXTURE(chunk_iterator, float_3) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(double);

    uint8_t ndim = 2;
    uint64_t shape[] = {123, 154};
    uint64_t pshape[] = {23, 31};
    uint64_t  nchunk[] = {4, 3};

    INA_TEST_ASSERT_SUCCEED(test_chunk_iterator(data->ctx, dtype, type_size, ndim, shape, pshape, nchunk));
}