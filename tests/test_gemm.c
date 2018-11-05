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


#define NTHREADS 1

#define KB  1024
#define MB  (1024*KB)
#define GB  (1024*MB)

/* Compute and fill X values in a buffer */
void fill_buf(double *x, int nitems) {

    /* Fill with even values between 0 and 10 */
    double incx = 10. / nitems;

    for (int i = 0; i < nitems; i++) {
        x[i] = incx * i;
    }
}


int simple_matmul(size_t n, double const *a, double const *b, double *c)
{
    size_t i, j, k;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            double t = 0.0;
            for (k = 0; k < n; ++k)
                t += a[i * n + k] * b[k * n + j];
            c[i * n + j] += t;
        }
    }
    return 0;
}


bool test_mat_equal(double *c1, double *c2, int size) {
    for (int nelem = 0; nelem < size; nelem++) {
        double vdiff = fabs((c1[nelem] - c2[nelem]) / c1[nelem]);
        if (vdiff > 1e-6) {
            printf("%f, %f\n", c1[nelem], c2[nelem]);
            printf("Values differ in (%d nelem) (diff: %f)\n", nelem, vdiff);
            return false;
        }
    }
    return true;
}

/* Check that two super-chunks with the same partitions are equal */
int test_dgemm(caterva_array *cta_x, caterva_array *cta_y, caterva_array *cta_out, double *res) {

    size_t P = cta_x->cshape[7];

    size_t M = cta_x->eshape[6];
    size_t K = cta_x->eshape[7];
    size_t N = cta_y->eshape[7];

    size_t p_size = P * P;
    size_t typesize = cta_x->sc->typesize;

    double *x_block = (double *)malloc(p_size * typesize);
    double *y_block = (double *)malloc(p_size * typesize);
    double *out_block = (double *)malloc(p_size * typesize);

    int x_i, y_i;

    for (size_t m = 0; m < M / P; m++)
    {
        for (size_t n = 0; n < N / P; n++) {
            memset(out_block, 0, p_size * typesize);

            for (size_t k = 0; k < K / P; k++)
            {
                x_i = (m * K / P + k);
                y_i = (k * N / P + n);

                blosc2_schunk_decompress_chunk(cta_x->sc, x_i, x_block, p_size * typesize);
                blosc2_schunk_decompress_chunk(cta_y->sc, y_i, y_block, p_size * typesize);

                simple_matmul(P, x_block, y_block, out_block);
            }
            blosc2_schunk_append_buffer(cta_out->sc, &out_block[0], p_size * typesize);
        }
    }

    double *buf_out = (double *)malloc(cta_out->size * cta_out->sc->typesize);

    caterva_to_buffer(cta_out, buf_out);

    if (!test_mat_equal(buf_out, res, cta_out->size)) {
        return -1;
    }

    return 1;
}

INA_TEST_DATA(e) {
    int tests_run;

    blosc2_cparams cparams;
    blosc2_dparams dparams;

};

INA_TEST_SETUP(e) {

    blosc_init();

    data->cparams = BLOSC_CPARAMS_DEFAULTS;
    data->dparams = BLOSC_DPARAMS_DEFAULTS;

    data->cparams.typesize = sizeof(double);
    data->cparams.nthreads = NTHREADS;
    data->dparams.nthreads = NTHREADS;

}

INA_TEST_TEARDOWN() {

    blosc_destroy();
}

INA_TEST_FIXTURE(e, mul1) {
    // Create a super-chunk container for input (X values)

    size_t M = 1000;
    size_t N = 1000;
    size_t K = 1000;
    size_t P = 100;

    caterva_pparams pparams_x;
    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        pparams_x.shape[i] = 1;
        pparams_x.cshape[i] = 1;
    }

    pparams_x.shape[CATERVA_MAXDIM - 1] = M;  // FIXME: 1's at the beginning should be removed
    pparams_x.shape[CATERVA_MAXDIM - 2] = K;  // FIXME: 1's at the beginning should be removed
    pparams_x.cshape[CATERVA_MAXDIM - 1] = P;  // FIXME: 1's at the beginning should be removed
    pparams_x.cshape[CATERVA_MAXDIM - 2] = P;  // FIXME: 1's at the beginning should be removed

    pparams_x.ndims = 2;

    blosc2_frame fr_x = BLOSC_EMPTY_FRAME;

    caterva_array *cta_x = caterva_new_array(data->cparams, data->dparams, &fr_x, pparams_x);

    double *buffer_x = ina_mem_alloc(sizeof(double) * M * K);
    fill_buf(buffer_x, M * K);

    caterva_from_buffer(cta_x, buffer_x);

    caterva_pparams pparams_y;
    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        pparams_y.shape[i] = 1;
        pparams_y.cshape[i] = 1;
    }

    pparams_y.shape[CATERVA_MAXDIM - 1] = K;  // FIXME: 1's at the beginning should be removed
    pparams_y.shape[CATERVA_MAXDIM - 2] = N;  // FIXME: 1's at the beginning should be removed
    pparams_y.cshape[CATERVA_MAXDIM - 1] = P;  // FIXME: 1's at the beginning should be removed
    pparams_y.cshape[CATERVA_MAXDIM - 2] = P;  // FIXME: 1's at the beginning should be removed

    pparams_y.ndims = 2;

    blosc2_frame fr_y = BLOSC_EMPTY_FRAME;

    caterva_array *cta_y = caterva_new_array(data->cparams, data->dparams, &fr_y, pparams_y);

    double *buffer_y = ina_mem_alloc(sizeof(double) * K * N);
    fill_buf(buffer_y, K * N);

    caterva_from_buffer(cta_y, buffer_y);

    caterva_pparams pparams_out;
    for (int i = 0; i < CATERVA_MAXDIM; i++) {
        pparams_out.shape[i] = 1;
        pparams_out.cshape[i] = 1;
    }

    pparams_out.shape[CATERVA_MAXDIM - 1] = M;  // FIXME: 1's at the beginning should be removed
    pparams_out.shape[CATERVA_MAXDIM - 2] = N;  // FIXME: 1's at the beginning should be removed
    pparams_out.cshape[CATERVA_MAXDIM - 1] = P;  // FIXME: 1's at the beginning should be removed
    pparams_out.cshape[CATERVA_MAXDIM - 2] = P;  // FIXME: 1's at the beginning should be removed

    pparams_out.ndims = 2;

    blosc2_frame fr_out = BLOSC_EMPTY_FRAME;

    caterva_array *cta_out = caterva_new_array(data->cparams, data->dparams, &fr_out, pparams_out);

    double *buffer_out = malloc(cta_out->size * cta_out->sc->typesize);

    simple_matmul(M, buffer_x, buffer_y, buffer_out);

    INA_TEST_ASSERT_TRUE(test_dgemm(cta_x, cta_y, cta_out, buffer_out));

    caterva_free_array(cta_x);
    caterva_free_array(cta_y);
    caterva_free_array(cta_out);

    ina_mem_free(buffer_x);
    ina_mem_free(buffer_y);
}
