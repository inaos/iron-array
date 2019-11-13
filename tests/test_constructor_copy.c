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

static ina_rc_t test_copy(iarray_context_t *ctx, iarray_data_type_t dtype, int8_t ndim,
                           const int64_t *shape, const int64_t *pshape, double start,
                           double stop, int64_t *stop_view, bool src_view, bool dest_view)
{
    int typesize;
    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        typesize = sizeof(double);
    } else {
        typesize = sizeof(float);
    }

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

    double step = (stop - start) / size;

    iarray_container_t *c_x;
    iarray_container_t *c_aux;

    printf("Start procedure\n");
    if (src_view) {
        printf("Src is view\n");
        INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, start, stop, step, NULL, 0, &c_aux));
        printf("Arange done\n");
        int64_t start_view[IARRAY_DIMENSION_MAX];
        for (int i = 0; i < ndim; ++i) {
            start_view[i] = 0;
        }
        printf("Start get slice\n");
        INA_TEST_ASSERT_SUCCEED(iarray_get_slice(ctx, c_aux, start_view, stop_view, stop_view, NULL, 0, true, &c_x));
        printf("Start squeeze\n");
        INA_TEST_ASSERT_SUCCEED(iarray_squeeze(ctx, c_x));
    } else {
        printf("Src is not view\n");
        INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, start, stop, step, NULL, 0, &c_x));
        printf("Finish arange\n");
    }

    iarray_container_t *c_y;
    printf("Start copy\n");
    INA_TEST_ASSERT_SUCCEED(iarray_copy(ctx, c_x, dest_view, NULL, 0, &c_y));

    // Assert iterator reading it
    printf("Start assertion\n");
    double tol;
    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        tol = 1e-14;
    } else {
        tol = 1e-6;
    }
    iarray_container_almost_equal(c_x, c_y, tol);

    printf("Stat free\n");
    if (src_view) {
        iarray_container_free(ctx, &c_aux);
    }
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_x);

    return INA_SUCCESS;
}

INA_TEST_DATA(constructor_copy) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_copy) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_copy) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

/*
INA_TEST_FIXTURE(constructor_copy, 1_f_p_n_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 1;
    int64_t shape[] = {1000};
    int64_t pshape[] = {0};
    int64_t stop_view[] = {431};
    double start = 0;
    double stop = 1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, false, false));
}

INA_TEST_FIXTURE(constructor_copy, 2_f_p_v_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {10, 200};
    int64_t pshape[] = {0, 0};
    int64_t stop_view[] = {1, 121};
    double start = - 0.1;
    double stop = - 0.2;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, true, false));
}

INA_TEST_FIXTURE(constructor_copy, 3_f_p_n_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 3;
    int64_t shape[] = {10, 20, 10};
    int64_t pshape[] = {0, 0, 0};
    int64_t stop_view[] = {2, 5, 6};
    double start = 1;
    double stop = 25;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, false, true));
}

INA_TEST_FIXTURE(constructor_copy, 4_f_p_v_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 4;
    int64_t shape[] = {10, 1, 1, 33};
    int64_t pshape[] = {0, 0, 0, 0};
    int64_t stop_view[] = {5, 1, 1, 12};
    double start = - 5;
    double stop = 101010;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, true, true));
}


INA_TEST_FIXTURE(constructor_copy, 5_d_p_n_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 5;
    int64_t shape[] = {2, 3, 4, 5, 6};
    int64_t pshape[] = {0, 0, 0, 0, 0};
    int64_t stop_view[] = {2, 2, 2, 2, 2};
    double start = - 0.1;
    double stop = - 0.25;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, false, false));
}

INA_TEST_FIXTURE(constructor_copy, 6_d_p_v_n) {
iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

int8_t ndim = 2;
int64_t shape[] = {6, 3, 6, 3, 6, 3};
int64_t pshape[] = {0, 0, 0, 0, 0, 0};
int64_t stop_view[] = {4, 3, 2, 3, 4, 3};

double start = 1000;
double stop = 2000;

INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, true, false));
}

INA_TEST_FIXTURE(constructor_copy, 7_d_p_n_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {2, 4, 6, 8, 6, 4, 2};
    int64_t pshape[] = {0, 0, 0, 0, 0, 0, 0};
    int64_t stop_view[] = {2, 3, 5, 2, 2, 2};

    double start = 0;
    double stop = 0.000001;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, false, true));
}

INA_TEST_FIXTURE(constructor_copy, 8_d_p_v_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 8;
    int64_t shape[] = {2, 9, 3, 8, 4, 7, 5, 6};
    int64_t pshape[] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t stop_view[] = {2, 2, 2, 2, 2, 2, 2, 2};
    double start = -1;
    double stop = 1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, true, true));
}

*/

INA_TEST_FIXTURE(constructor_copy, 8_f_n_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 8;
    int64_t shape[] = {5, 4, 7, 5, 4, 6, 2, 3};
    int64_t pshape[] = {2, 3, 4, 2, 2, 4, 1, 2};
    int64_t stop_view[] = {2, 2, 2, 2, 2, 2, 2, 2};
    double start = 0;
    double stop = 1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, false, false));
}

/*
INA_TEST_FIXTURE(constructor_copy, 7_f_v_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 7;
    int64_t shape[] = {7, 4, 8, 4, 5, 8, 4};
    int64_t pshape[] = {3, 3, 3, 3, 3, 3, 3};
    int64_t stop_view[] = {2, 2, 2, 3, 2, 2, 2};

    double start = 0;
    double stop = 5;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, true, false));
}

INA_TEST_FIXTURE(constructor_copy, 6_f_n_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 6;
    int64_t shape[] = {5, 7, 10, 12, 13, 6};
    int64_t pshape[] = {4, 4, 5, 11, 12, 4};
    int64_t stop_view[] = {2, 1, 4, 5, 6};
    double start = -0.112;
    double stop = 10102;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, false, true));
}

INA_TEST_FIXTURE(constructor_copy, 5_f_v_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 5;
    int64_t shape[] = {31, 21, 11, 5, 11};
    int64_t pshape[] = {21, 10, 3, 3, 8};
    int64_t stop_view[] = {10, 11, 3, 2, 4};

    double start = 1;
    double stop = -1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, true, true));
}

INA_TEST_FIXTURE(constructor_copy, 4_d_n_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 4;
    int64_t shape[] = {12, 31, 54, 12};
    int64_t pshape[] = {8, 8, 8, 3};
    int64_t stop_view[] = {2, 3, 23, 5};

    double start = 0.1;
    double stop = 0.9;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, false, false));
}

INA_TEST_FIXTURE(constructor_copy, 3_d_v_n) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 3;
    int64_t shape[] = {31, 45, 23};
    int64_t pshape[] = {21, 17, 11};
    int64_t stop_view[] = {5, 5, 4};

    double start = 0.00001;
    double stop = 0.00002;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, true, false));
}

INA_TEST_FIXTURE(constructor_copy, 2_d_n_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {54, 66};
    int64_t pshape[] = {21, 17};
    int64_t stop_view[] = {22, 31};

    double start = 3123;
    double stop = 45654;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, false, true));
}

INA_TEST_FIXTURE(constructor_copy, 1_d_v_v) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 1;
    int64_t shape[] = {445};
    int64_t pshape[] = {21};
    int64_t stop_view[] = {121};
    double start = -0.1;
    double stop = 0.1;

    INA_TEST_ASSERT_SUCCEED(test_copy(data->ctx, dtype, ndim, shape, pshape, start, stop, stop_view, true, true));
}
*/