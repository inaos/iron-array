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

static ina_rc_t test_get_ncores(int max_ncores)
{
    int ncores;
    INA_TEST_ASSERT_SUCCEED(iarray_get_ncores(&ncores, max_ncores));

    if (max_ncores == 0) {
        INA_TEST_ASSERT(ncores > max_ncores);
    }
    else if (max_ncores == 1) {
        INA_TEST_ASSERT(ncores == 1);
    }
    else {
        // The number of detected cores should always be > 1 on modern CPUs
        // (except maybe on some exotic CI platforms)
        INA_TEST_ASSERT(ncores > 1);
        INA_TEST_ASSERT(ncores <= max_ncores);
    }

    return INA_SUCCESS;
}

INA_TEST_DATA(get_ncores) {
    int max_ncores;    // to avoid warnings
};

INA_TEST_SETUP(get_ncores)
{
    iarray_init();
    data->max_ncores = 0;  // to avoid warnings
}

INA_TEST_TEARDOWN(get_ncores)
{
    iarray_destroy();
    data->max_ncores = 0;  // to avoid warnings
}

INA_TEST_FIXTURE(get_ncores, max_0)
{
    INA_TEST_ASSERT_SUCCEED(test_get_ncores(data->max_ncores));
}

INA_TEST_FIXTURE(get_ncores, max_1)
{
    data->max_ncores = 1;
    INA_TEST_ASSERT_SUCCEED(test_get_ncores(data->max_ncores));
}

INA_TEST_FIXTURE(get_ncores, max_8)
{
    data->max_ncores = 8;
    INA_TEST_ASSERT_SUCCEED(test_get_ncores(data->max_ncores));
}
