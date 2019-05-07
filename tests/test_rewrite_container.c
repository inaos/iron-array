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

static ina_rc_t test_rewrite_cont(iarray_context_t *ctx, iarray_data_type_t dtype,
                                  int32_t type_size, int8_t ndim, const int64_t *shape,
                                  const int64_t *pshape, const int64_t *blockshape, bool rewrite)
{
    INA_UNUSED(type_size);
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

    int niter = 1;
    if (rewrite) {
        niter++;
    }

    for (int j = 0; j < niter; ++j) {
        // Start Iterator
        iarray_iter_write_block_t *I;
        iarray_iter_write_block_value_t val;
        ina_rc_t err = iarray_iter_write_block_new(ctx, &I, c_x, blockshape, &val);
        if (rewrite && (j == 1)) {
            if (err != 0) { // We need the iterator to return an error
                return INA_SUCCESS;
            }
        }
        while (iarray_iter_write_block_has_next(I)) {
            iarray_iter_write_block_next(I);

            int64_t nelem = 0;
            int64_t inc = 1;
            for (int i = ndim - 1; i >= 0; --i) {
                nelem += val.elem_index[i] * inc;
                inc *= c_x->dtshape->shape[i];
            }
            if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
                for (int64_t i = 0; i < val.block_size; ++i) {
                    ((double *) val.pointer)[i] = (double) nelem + i;
                }
            } else {
                for (int64_t i = 0; i < val.block_size; ++i) {
                    ((float *) val.pointer)[i] = (float) nelem + i;
                }
            }
        }

        iarray_iter_write_block_free(I);
    }

    return INA_SUCCESS;
}

INA_TEST_DATA(rewrite_cont) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(rewrite_cont) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(rewrite_cont) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(rewrite_cont, 2_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {5, 5};
    int64_t pshape[] = {0, 0};
    int64_t blockshape[] = {3, 2};

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, pshape,
                                              blockshape, false));
}


INA_TEST_FIXTURE(rewrite_cont, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t pshape[] = {23, 32, 35};
    int64_t *blockshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, pshape,
                                              blockshape, true));
}


INA_TEST_FIXTURE(rewrite_cont, 4_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 4;
    int64_t shape[] = {30, 64, 50, 43};
    int64_t pshape[] = {11, 8, 12, 21};
    int64_t *blockshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, pshape,
                                              blockshape, false));
}

INA_TEST_FIXTURE(rewrite_cont, 5_f_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 5;
    int64_t shape[] = {40, 26, 35, 23, 21};
    int64_t pshape[] = {0, 0, 0, 0, 0};
    int64_t blockshape[] = {12, 12, 12, 12, 12};

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, pshape,
                                              blockshape, true));
}

INA_TEST_FIXTURE(rewrite_cont, 6_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 6;
    int64_t shape[] = {12, 13, 21, 19, 13, 15};
    int64_t pshape[] = {0, 0, 0, 0, 0, 0};
    int64_t blockshape[] = {2, 3, 5, 4, 3, 2};

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, pshape,
                                              blockshape, false));
}

INA_TEST_FIXTURE(rewrite_cont, 7_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 7;
    int64_t shape[] = {10, 8, 6, 7, 13, 9, 10};
    int64_t pshape[] = {2, 3, 1, 3, 2, 4, 5};
    int64_t *blockshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_rewrite_cont(data->ctx, dtype, type_size, ndim, shape, pshape,
                                              blockshape, true));
}
