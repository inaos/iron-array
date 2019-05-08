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
                                   const int64_t *pshape, const int64_t *blockshape)
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
    iarray_iter_write_block_t *I;
    iarray_iter_write_block_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_new(ctx, &I, c_x, blockshape, &val));

    while (iarray_iter_write_block_has_next(I)) {
        iarray_iter_write_block_next(I);

        int64_t nelem = 0;
        int64_t inc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            nelem += val.elem_index[i] * inc;
            inc *= c_x->dtshape->shape[i];
        }
        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((double *)val.pointer)[i] = (double) nelem + i;
            }
        } else {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((float *)val.pointer)[i] = (float) nelem  + i;
            }
        }
    }

    iarray_iter_write_block_free(I);

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
    iarray_iter_read_block_value_t val2;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I2, c_x, blockshape, &val2));

    iarray_iter_read_block_t *I3;
    iarray_iter_read_block_value_t val3;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I3, c_y, blockshape, &val3));

    while (iarray_iter_read_block_has_next(I2) & iarray_iter_read_block_has_next(I3)) {
        iarray_iter_read_block_next(I2);
        iarray_iter_read_block_next(I3);

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                for (int64_t i = 0; i < val2.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((double *) val2.pointer)[i],
                        ((double *) val3.pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                for (int64_t i = 0; i < val3.block_size; ++i) {
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
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(part_iterator) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(part_iterator, 2_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {5, 5};
    int64_t pshape[] = {0, 0};
    int64_t blockshape[] = {3, 2};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape,
                                               blockshape));
}


INA_TEST_FIXTURE(part_iterator, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t pshape[] = {23, 32, 35};
    int64_t *blockshape = pshape;

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape,
                                               blockshape));
}


INA_TEST_FIXTURE(part_iterator, 4_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t pshape[] = {11, 8, 12, 21};
    int64_t *blockshape = pshape;

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape,
                                               blockshape));
}

INA_TEST_FIXTURE(part_iterator, 5_f_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 5;
    int64_t shape[] = {40, 26, 35, 23, 21};
    int64_t pshape[] = {0, 0, 0, 0, 0};
    int64_t blockshape[] = {12, 12, 12, 12, 12};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape,
                                               blockshape));
}

INA_TEST_FIXTURE(part_iterator, 6_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 6;
    int64_t shape[] = {12, 13, 21, 19, 13, 15};
    int64_t pshape[] = {0, 0, 0, 0, 0, 0};
    int64_t blockshape[] = {2, 3, 5, 4, 3, 2};

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape,
                                               blockshape));
}

INA_TEST_FIXTURE(part_iterator, 7_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 7;
    int64_t shape[] = {10, 8, 6, 7, 13, 9, 10};
    int64_t pshape[] = {2, 3, 1, 3, 2, 4, 5};
    int64_t *blockshape = pshape;

    INA_TEST_ASSERT_SUCCEED(test_part_iterator(data->ctx, dtype, type_size, ndim, shape, pshape,
                                               blockshape));
}
