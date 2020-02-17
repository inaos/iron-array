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

static ina_rc_t test_block_iterator(iarray_context_t *ctx, iarray_data_type_t dtype,
                                    int32_t type_size, int8_t ndim, const int64_t *shape,
                                    const int64_t *pshape, const int64_t *blockshape)
{
    iarray_dtshape_t xdtshape;
    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        if (pshape)
            xdtshape.pshape[i] = pshape[i];
        size *= shape[i];
    }

    iarray_store_properties_t xstore;
    xstore.backend = pshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    xstore.enforce_frame = false;
    xstore.filename = NULL;

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &xdtshape, &xstore, 0, &c_x));

    // Test write iterator
    iarray_iter_write_block_t *I;
    iarray_iter_write_block_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_new(ctx, &I, c_x, blockshape, &val, false));

    while (INA_SUCCEED(iarray_iter_write_block_has_next(I))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_next(I, NULL, 0));

        int64_t nelem = 0;
        int64_t inc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            nelem += val.elem_index[i] * inc;
            inc *= c_x->dtshape->shape[i];
        }
        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((double *) val.block_pointer)[i] = (double) nelem + i;
            }
        } else {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((float *) val.block_pointer)[i] = (float) nelem  + i;
            }
        }
    }

    iarray_iter_write_block_free(&I);
    
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));

    uint8_t *buf = ina_mem_alloc((size_t)c_x->catarr->size * type_size);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buf, (size_t)c_x->catarr->size * type_size));

    if (c_x->dtshape->ndim == 2) {
        switch (c_x->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                mkl_dimatcopy('R', 'T', (size_t)c_x->dtshape->shape[0], (size_t)c_x->dtshape->shape[1], 1.0,
                              (double *) buf, (size_t)c_x->dtshape->shape[1], (size_t)c_x->dtshape->shape[0]);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                mkl_simatcopy('R', 'T', (size_t)c_x->dtshape->shape[0], (size_t)c_x->dtshape->shape[1], 1.0f,
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
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buf, (size_t)c_x->catarr->size * type_size, &xstore, 0, &c_y));

    if (ndim == 2) {
        INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_x));
    }

    // Test read iterator
    iarray_iter_read_block_t *I2;
    iarray_iter_read_block_value_t val2;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I2, c_x, blockshape, &val2, false));

    iarray_iter_read_block_t *I3;
    iarray_iter_read_block_value_t val3;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I3, c_y, blockshape, &val3, false));

    while (INA_SUCCEED(iarray_iter_read_block_has_next(I2)) && INA_SUCCEED(iarray_iter_read_block_has_next(I3))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I2, NULL, 0));
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I3, NULL, 0));

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                for (int64_t i = 0; i < val2.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((double *) val2.block_pointer)[i],
                        ((double *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((float *) val2.block_pointer)[i],
                                                   ((float *) val3.block_pointer)[i]);
                }
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    iarray_iter_read_block_free(&I2);
    iarray_iter_read_block_free(&I3);

    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));


    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);

    ina_mem_free(buf);

    return INA_SUCCESS;
}

INA_TEST_DATA(block_iterator) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(block_iterator) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(block_iterator) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(block_iterator, 2_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {5, 5};
    int64_t *pshape = NULL;
    int64_t blockshape[] = {3, 2};

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                blockshape));
}


INA_TEST_FIXTURE(block_iterator, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t pshape[] = {5, 6};
    int64_t *blockshape = pshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                blockshape));
}

INA_TEST_FIXTURE(block_iterator, 4_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t pshape[] = {11, 8, 12, 21};
    int64_t *blockshape = pshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                blockshape));
}

INA_TEST_FIXTURE(block_iterator, 5_f_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 5;
    int64_t shape[] = {40, 26, 35, 23, 21};
    int64_t *pshape = NULL;
    int64_t blockshape[] = {12, 12, 12, 12, 12};

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                blockshape));
}

INA_TEST_FIXTURE(block_iterator, 6_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 6;
    int64_t shape[] = {12, 13, 21, 19, 13, 15};
    int64_t *pshape = NULL;
    int64_t blockshape[] = {2, 3, 5, 4, 3, 2};

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                blockshape));
}

INA_TEST_FIXTURE(block_iterator, 7_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 7;
    int64_t shape[] = {10, 8, 6, 7, 13, 9, 10};
    int64_t pshape[] = {2, 3, 1, 3, 2, 4, 5};
    int64_t *blockshape = pshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                blockshape));
}

static ina_rc_t test_block_iterator_ext_part(iarray_context_t *ctx, iarray_data_type_t dtype,
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
        if (pshape)
            xdtshape.pshape[i] = pshape[i];
        size *= shape[i];
    }

    iarray_store_properties_t xstore;
    xstore.backend = pshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    xstore.enforce_frame = false;
    xstore.filename = NULL;

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &xdtshape, &xstore, 0, &c_x));

    // Start Iterator
    iarray_iter_write_block_t *I;
    iarray_iter_write_block_value_t val;

    int64_t partsize_x = 0;

    switch (c_x->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            partsize_x = sizeof(double);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            partsize_x = sizeof(float);
            break;
        default:
            break;
    }

    for (int i = 0; i < c_x->dtshape->ndim; ++i) {
        if (c_x->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
            partsize_x *= c_x->dtshape->shape[i];
        } else {
            partsize_x *= c_x->dtshape->pshape[i];
        }
    }

    partsize_x += BLOSC_MAX_OVERHEAD;

    uint8_t *part_x = (uint8_t *) malloc(partsize_x);

    INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_new(ctx, &I, c_x, blockshape, &val, true));

    while (INA_SUCCEED(iarray_iter_write_block_has_next(I))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_next(I, (void *) part_x, partsize_x));

        int64_t nelem = 0;
        int64_t inc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            nelem += val.elem_index[i] * inc;
            inc *= c_x->dtshape->shape[i];
        }
        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((double *)val.block_pointer)[i] = (double) nelem + i;
            }
        } else {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((float *)val.block_pointer)[i] = (float) nelem  + i;
            }
        }
    }

    iarray_iter_write_block_free(&I);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));


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
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buf, (size_t)c_x->catarr->size * type_size, &xstore, 0, &c_y));

    //Testing

    if (ndim == 2) {
        INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_x));
    }

    // Start Iterator
    iarray_iter_read_block_t *I2;
    iarray_iter_read_block_value_t val2;

    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I2, c_x, blockshape, &val2, true));

    iarray_iter_read_block_t *I3;
    iarray_iter_read_block_value_t val3;

    int64_t partsize_y = 0;
    switch (c_y->dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            partsize_y = c_y->catarr->psize * sizeof(double);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            partsize_y = c_y->catarr->psize * sizeof(float);
            break;
        default:
            break;
    }

    partsize_y += BLOSC_MAX_OVERHEAD;

    uint8_t *part_y = (uint8_t *) malloc(partsize_y);
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I3, c_y, blockshape, &val3, true));

    while (INA_SUCCEED(iarray_iter_read_block_has_next(I2)) && INA_SUCCEED(iarray_iter_read_block_has_next(I3))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I2, (void *) part_x, partsize_x));
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I3, (void *) part_y, partsize_y));

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                for (int64_t i = 0; i < val2.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((double *) val2.block_pointer)[i],
                                                   ((double *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((float *) val2.block_pointer)[i],
                                                   ((float *) val3.block_pointer)[i]);
                }
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    iarray_iter_read_block_free(&I2);
    iarray_iter_read_block_free(&I3);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));


    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);

    free(part_x);
    free(part_y);
    ina_mem_free(buf);

    return INA_SUCCESS;
}

INA_TEST_DATA(block_iterator_ext_part) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(block_iterator_ext_part) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(block_iterator_ext_part) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(block_iterator_ext_part, 2_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {5, 5};
    int64_t *pshape = NULL;
    int64_t blockshape[] = {3, 2};

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_part(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                         blockshape));
}

INA_TEST_FIXTURE(block_iterator_ext_part, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t pshape[] = {23, 32, 35};
    int64_t *blockshape = pshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_part(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                         blockshape));
}

INA_TEST_FIXTURE(block_iterator_ext_part, 4_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t pshape[] = {11, 8, 12, 21};
    int64_t *blockshape = pshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_part(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                         blockshape));
}

INA_TEST_FIXTURE(block_iterator_ext_part, 5_f_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 5;
    int64_t shape[] = {40, 26, 35, 23, 21};
    int64_t *pshape = NULL;
    int64_t blockshape[] = {12, 12, 12, 12, 12};

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_part(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                         blockshape));
}

INA_TEST_FIXTURE(block_iterator_ext_part, 6_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 6;
    int64_t shape[] = {12, 13, 21, 19, 13, 15};
    int64_t *pshape = NULL;
    int64_t blockshape[] = {2, 3, 5, 4, 3, 2};

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_part(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                         blockshape));
}

INA_TEST_FIXTURE(block_iterator_ext_part, 7_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 7;
    int64_t shape[] = {10, 8, 6, 7, 13, 9, 10};
    int64_t pshape[] = {2, 3, 4, 4, 2, 4, 5};
    int64_t *blockshape = pshape;

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_ext_part(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                         blockshape));
}

static ina_rc_t test_block_iterator_not_empty(iarray_context_t *ctx, iarray_data_type_t dtype,
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
        if (pshape)
            xdtshape.pshape[i] = pshape[i];
        size *= shape[i];
    }

    iarray_store_properties_t xstore;
    xstore.backend = pshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    xstore.enforce_frame = false;
    xstore.filename = NULL;


    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, 0, size, 1, &xstore, 0, &c_x));

    // Test write iterator
    iarray_iter_write_block_t *I;
    iarray_iter_write_block_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_new(ctx, &I, c_x, blockshape, &val, false));

    while (INA_SUCCEED(iarray_iter_write_block_has_next(I))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_write_block_next(I, NULL, 0));

        int64_t nelem = 0;
        int64_t inc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            nelem += val.elem_index[i] * inc;
            inc *= c_x->dtshape->shape[i];
        }
        if(dtype == IARRAY_DATA_TYPE_DOUBLE) {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((double *) val.block_pointer)[i] = (double) nelem + i;
            }
        } else {
            for (int64_t i = 0; i < val.block_size; ++i) {
                ((float *) val.block_pointer)[i] = (float) nelem  + i;
            }
        }
    }

    iarray_iter_write_block_free(&I);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));


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
    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buf, (size_t)c_x->catarr->size * type_size, &xstore, 0, &c_y));

    if (ndim == 2) {
        INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_x));
    }

    // Test read iterator
    iarray_iter_read_block_t *I2;
    iarray_iter_read_block_value_t val2;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I2, c_x, blockshape, &val2, false));

    iarray_iter_read_block_t *I3;
    iarray_iter_read_block_value_t val3;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_new(ctx, &I3, c_y, blockshape, &val3, false));

    while (INA_SUCCEED(iarray_iter_read_block_has_next(I2)) && INA_SUCCEED(iarray_iter_read_block_has_next(I3))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I2, NULL, 0));
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_block_next(I3, NULL, 0));

        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                for (int64_t i = 0; i < val2.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((double *) val2.block_pointer)[i],
                                                   ((double *) val3.block_pointer)[i]);
                }
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                for (int64_t i = 0; i < val3.block_size; ++i) {
                    INA_TEST_ASSERT_EQUAL_FLOATING(((float *) val2.block_pointer)[i],
                                                   ((float *) val3.block_pointer)[i]);
                }
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    iarray_iter_read_block_free(&I2);
    iarray_iter_read_block_free(&I3);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));


    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);

    ina_mem_free(buf);

    return INA_SUCCESS;
}


INA_TEST_DATA(block_iterator_not_empty) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(block_iterator_not_empty) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(block_iterator_not_empty) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(block_iterator_not_empty, 2_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {5, 5};
    int64_t *pshape = NULL;
    int64_t blockshape[] = {3, 7};

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                blockshape));
}

INA_TEST_FIXTURE(block_iterator_not_empty, 3_f_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 2;
    int64_t shape[] = {10, 10};
    int64_t *pshape = NULL;
    int64_t blockshape[] = {6, 8};

    INA_TEST_ASSERT_SUCCEED(test_block_iterator_not_empty(data->ctx, dtype, type_size, ndim, shape, pshape,
                                                blockshape));
}
