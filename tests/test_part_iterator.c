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
#include <src/iarray_private.h>

static ina_rc_t test_part_iterator(iarray_context_t *ctx, iarray_data_type_t dtype,
                                   int32_t type_size, int8_t ndim, const int64_t *shape,
                                   const int64_t *pshape)
{
    // Create dtshape
    iarray_dtshape_t xdtshape;
    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        xdtshape.pshape[i] = pshape[i];
        size *= shape[i];
    }

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &xdtshape, NULL, 0, &c_x));

    // Start Iterator
    iarray_iter_write_part_t *I;
    iarray_iter_write_part_new(ctx, c_x, &I);

    for (iarray_iter_write_part_init(I);
         !iarray_iter_write_part_finished(I);
         iarray_iter_write_part_next(I)) {

        iarray_iter_write_part_value_t val;
        iarray_iter_write_part_value(I, &val);

        int64_t part_size = 1;
        for (int i = 0; i < ndim; ++i) {
            part_size *= val.part_shape[i];
        }

        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (int64_t i = 0; i < part_size; ++i) {
                ((double *)val.pointer)[i] = (double) val.nelem * part_size + i;
            }
        } else {
            for (int64_t i = 0; i < part_size; ++i) {
                ((float *)val.pointer)[i] = (float) val.nelem * part_size + i;
            }
        }
    }

    iarray_iter_write_part_free(I);

    uint8_t *buf = ina_mem_alloc((size_t)c_x->catarr->size * type_size);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buf, (size_t)c_x->catarr->size * type_size));

    if (c_x->dtshape->ndim == 2) {
        switch (c_x->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_dimatcopy('R', 'T', (size_t)c_x->dtshape->shape[0], (size_t)c_x->dtshape->shape[1], 1.0,
                              (double *) buf, (size_t)c_x->dtshape->shape[1], (size_t)c_x->dtshape->shape[0]);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_simatcopy('R', 'T', (size_t)c_x->dtshape->shape[0], (size_t)c_x->dtshape->shape[1], 1.0,
                              (float *) buf, (size_t)c_x->dtshape->shape[1], (size_t)c_x->dtshape->shape[0]);
                break;
            default:
                return INA_ERR_EXCEEDED;
        }

        int64_t aux = xdtshape.shape[0];
        xdtshape.shape[0] = xdtshape.shape[1];
        xdtshape.shape[1] = aux;
    }

    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buf, (size_t)c_x->catarr->size * type_size, NULL, 0, &c_y));

    //Testing

    if (ndim == 2) {
        INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_x));
    }

    // Start Iterator
    iarray_iter_read_block_t *I2;
    iarray_iter_read_block_new(ctx, c_x, &I2, pshape);

    iarray_iter_read_block_t *I3;
    iarray_iter_read_block_new(ctx, c_y, &I3, pshape);


    for (iarray_iter_read_block_init(I2), iarray_iter_read_block_init(I3);
         !iarray_iter_read_block_finished(I2);
         iarray_iter_read_block_next(I2), iarray_iter_read_block_next(I3)) {

        iarray_iter_read_block_value_t val2;
        iarray_iter_read_block_value(I2, &val2);


        iarray_iter_read_block_value_t val3;
        iarray_iter_read_block_value(I3, &val3);

        int64_t block_size = 1;
        for (int i = 0; i < ndim; ++i) {
            block_size *= val2.block_shape[i];
        }

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                for (int64_t i = 0; i < block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((double *) val2.pointer)[i],
                                                   ((double *) val3.pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                for (int64_t i = 0; i < block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((float *) val2.pointer)[i],
                                                   ((float *) val3.pointer)[i]);
                }
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }


    iarray_iter_read_block_free(I2);
    iarray_iter_read_block_free(I3);

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);

    ina_mem_free(buf);

    return INA_SUCCESS;
}

INA_TEST_DATA(part_iterator) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(part_iterator) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.eval_flags = IARRAY_EXPR_EVAL_CHUNK;

    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(part_iterator) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(part_iterator, double_2) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t pshape[] = {2, 2};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}


INA_TEST_FIXTURE(part_iterator, float_3) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t pshape[] = {23, 32, 35};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}


INA_TEST_FIXTURE(part_iterator, double_4) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 4;
    int64_t shape[] = {80, 64, 80, 99};
    int64_t pshape[] = {11, 8, 12, 21};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(part_iterator, float_5) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 5;
    int64_t shape[] = {40, 26, 35, 23, 21};
    int64_t pshape[] = {5, 8, 10, 7, 9};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(part_iterator, double_6) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 6;
    int64_t shape[] = {12, 13, 21, 19, 13, 15};
    int64_t pshape[] = {5, 4, 7, 3, 4, 12};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(part_iterator, float_7) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 7;
    int64_t shape[] = {10, 8, 6, 7, 13, 9, 10};
    int64_t pshape[] = {2, 3, 1, 3, 2, 4, 5};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape));
}
