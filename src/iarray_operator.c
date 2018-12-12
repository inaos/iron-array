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

INA_API(ina_rc_t) iarray_almost_equal_data(iarray_container_t *a, iarray_container_t *b, double tol) {
    if(a->dtshape->dtype != b->dtshape->dtype){
        return false;
    }
    if(a->catarr->size != b->catarr->size) {
        return false;
    }
    size_t size = a->catarr->size;

    uint8_t *buf_a = malloc(a->catarr->size * a->catarr->sc->typesize);
    caterva_to_buffer(a->catarr, buf_a);
    uint8_t *buf_b = malloc(b->catarr->size * b->catarr->sc->typesize);
    caterva_to_buffer(b->catarr, buf_b);

    if(a->dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE) {
        double *b_a = (double *)buf_a;
        double *b_b = (double *)buf_b;

        for (size_t i = 0; i < size; ++i) {
            double vdiff = fabs((b_a[i] - b_b[i]) / b_a[i]);
            if (vdiff > tol) {
                printf("%f, %f\n", b_a[i], b_b[i]);
                printf("Values differ in (%lu nelem) (diff: %f)\n", i, vdiff);
                free(buf_a);
                free(buf_b);
                return false;
            }
        }
        free(buf_a);
        free(buf_b);
        return true;
    }
    else if(a->dtshape->dtype == IARRAY_DATA_TYPE_FLOAT) {
        float *b_a = (float *)buf_a;
        float *b_b = (float *)buf_b;

        for (size_t i = 0; i < size; ++i) {
            double vdiff = fabs((double)(b_a[i] - b_b[i]) / b_a[i]);
            if (vdiff > tol) {
                printf("%f, %f\n", b_a[i], b_b[i]);
                printf("Values differ in (%lu nelem) (diff: %f)\n", i, vdiff);
                free(buf_a);
                free(buf_b);
                return false;
            }
        }
        free(buf_a);
        free(buf_b);
        return true;
    }
    printf("Data type is not supported");
    free(buf_a);
    free(buf_b);
    return false;
}


static ina_rc_t _iarray_gemm(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *c) {

    caterva_update_shape(c->catarr, *c->shape);

    const int32_t P = (int32_t) a->catarr->pshape[0];
    uint64_t M = a->catarr->eshape[0];
    uint64_t K = a->catarr->eshape[1];
    uint64_t N = b->catarr->eshape[1];

    uint64_t p_size = (uint64_t) P * P * a->catarr->sc->typesize;
    int dtype = a->dtshape->dtype;

    uint8_t *a_block = malloc(p_size);
    uint8_t *b_block = malloc(p_size);
    uint8_t *c_block = malloc(p_size);

    iarray_itr_matmul_t *I;
    iarray_itr_matmul_new(ctx, a, b, &I);

    memset(c_block, 0, p_size);
    for (iarray_itr_matmul_init(I); !iarray_itr_matmul_finished(I); iarray_itr_matmul_next(I)) {

        int a_tam = blosc2_schunk_decompress_chunk(a->catarr->sc, (int)I->nchunk1, a_block, p_size);
        int b_tam = blosc2_schunk_decompress_chunk(b->catarr->sc, (int)I->nchunk2, b_block, p_size);

        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, P, P, P, 1.0, (double *)a_block, P, (double *)b_block, P, 1.0, (double *)c_block, P);
        }
        else if (dtype == IARRAY_DATA_TYPE_FLOAT) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, P, P, P, 1.0, (float *)a_block, P, (float *)b_block, P, 1.0, (float *)c_block, P);
        }
        if((I->cont + 1) % (K / P) == 0) {
            blosc2_schunk_append_buffer(c->catarr->sc, &c_block[0], p_size);
            memset(c_block, 0, p_size);
        }
    }

    free(a_block);
    free(b_block);
    free(c_block);

    return INA_SUCCESS;
}

static ina_rc_t _iarray_gemv(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *c) {

    caterva_update_shape(c->catarr, *c->shape);

    int32_t P = (int32_t) a->catarr->pshape[0];

    uint64_t M = a->catarr->eshape[0];
    uint64_t K = a->catarr->eshape[1];

    uint64_t p_size = (uint64_t) P * P * a->catarr->sc->typesize;
    uint64_t p_vsize = (uint64_t) P * a->catarr->sc->typesize;

    int dtype = a->dtshape->dtype;

    uint8_t *a_block = malloc(p_size);
    uint8_t *b_block = malloc(p_vsize);
    uint8_t *c_block = malloc(p_vsize);

    iarray_itr_matmul_t *I;
    iarray_itr_matmul_new(ctx, a, b, &I);

    memset(c_block, 0, p_vsize);
    for (iarray_itr_matmul_init(I); !iarray_itr_matmul_finished(I); iarray_itr_matmul_next(I)) {

        int a_tam = blosc2_schunk_decompress_chunk(a->catarr->sc, (int)I->nchunk1, a_block, p_size);
        int b_tam = blosc2_schunk_decompress_chunk(b->catarr->sc, (int)I->nchunk2, b_block, p_vsize);

        if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
            cblas_dgemv(CblasRowMajor, CblasNoTrans, P, P, 1.0, (double *) a_block, P, (double *) b_block, 1, 1.0, (double *) c_block, 1);
        }
        else if (dtype == IARRAY_DATA_TYPE_FLOAT) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, P, P, 1.0, (float *) a_block, P, (float *) b_block, 1, 1.0, (float *) c_block, 1);
        }

        if((I->cont + 1) % (K / P) == 0) {
            blosc2_schunk_append_buffer(c->catarr->sc, &c_block[0], p_vsize);
            memset(c_block, 0, p_vsize);
        }
    }
    free(a_block);
    free(b_block);
    free(c_block);

    return INA_SUCCESS;;
}

INA_API(ina_rc_t) iarray_matmul(iarray_context_t *ctx, iarray_container_t *a, iarray_container_t *b, iarray_container_t *c, int flag)
{
    /* FIXME: handle special shapes */
    if (a->dtshape->ndim != 2) {
        return INA_ERR_INVALID_ARGUMENT;
    }
    if (b->dtshape->ndim == 1) {
        return _iarray_gemv(ctx, a, b, c);
    }
    else if (b->dtshape->ndim == 2) {
        return _iarray_gemm(ctx, a, b, c);
    }
    else {
        return INA_ERR_INVALID_ARGUMENT;
    }
}
