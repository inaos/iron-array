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


static ina_rc_t test_cont_sframe(iarray_context_t *ctx, iarray_data_type_t dtype,
                                 int32_t type_size, int8_t ndim, const int64_t *shape,
                                 const int64_t *pshape)
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
    INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &xdtshape, size, 0, 1, NULL, 0, &c_x));

    uint8_t* sframe;
    int64_t len;
    bool shared;
    if (c_x->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        INA_TEST_ASSERT_FAILED(iarray_get_sframe(c_x, &sframe, &len, &shared));
        return INA_SUCCESS;
    }
    INA_TEST_ASSERT_SUCCEED(iarray_get_sframe(c_x, &sframe, &len, &shared));

    INA_TEST_ASSERT(shared == false);
    INA_TEST_ASSERT(len < size * type_size);

    return INA_SUCCESS;
}

INA_TEST_DATA(container_sframe) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(container_sframe) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(container_sframe) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(container_sframe, 1_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 1;
    int64_t shape[] = {10000};
    int64_t pshape[] = {1000};

    INA_TEST_ASSERT_SUCCEED(test_cont_sframe(data->ctx, dtype, type_size, ndim, shape, pshape));
}


INA_TEST_FIXTURE(container_sframe, 2_d_p) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {5, 5};
    int64_t pshape[] = {0, 0};

    INA_TEST_ASSERT_SUCCEED(test_cont_sframe(data->ctx, dtype, type_size, ndim, shape, pshape));
}

INA_TEST_FIXTURE(container_sframe, 3_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    int8_t ndim = 3;
    int64_t shape[] = {120, 131, 155};
    int64_t pshape[] = {23, 32, 35};

    INA_TEST_ASSERT_SUCCEED(test_cont_sframe(data->ctx, dtype, type_size, ndim, shape, pshape));
}
