/*
 * Copyright INAOS GmbH, Thalwil, 2019.
 * Copyright Francesc Alted, 2019.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of INAOS GmbH
 * and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <math.h>
#include <stdint.h>

// Min
void dmin(double *v, int64_t vlen, double *out) {
    double min = INFINITY;
    for (int i = 0; i < vlen; ++i) {
        if (v[i] < min)
            min = v[i];
    }

    *out = min;
}

void smin(float *v, int64_t vlen, float *out) {
    float min = INFINITY;
    for (int i = 0; i < vlen; ++i) {
        if (v[i] < min)
            min = v[i];
    }

    *out = min;
}

// Max
void dmax(double *v, int64_t vlen, double *out) {
    double max = -INFINITY;
    for (int i = 0; i < vlen; ++i) {
        if (v[i] > max)
            max = v[i];
    }

    *out = max;
}

void smax(float *v, int64_t vlen, float *out) {
    float max = -INFINITY;
    for (int i = 0; i < vlen; ++i) {
        if (v[i] > max)
            max = v[i];
    }

    *out = max;
}

// Sum
void dsum(double *v, int64_t vlen, double *out) {
    double sum = 0;
    for (int i = 0; i < vlen; ++i) {
        sum += v[i];
    }

    *out = sum;
}

void ssum(float *v, int64_t vlen, float *out) {
    float sum = 0;
    for (int i = 0; i < vlen; ++i) {
        sum += v[i];
    }

    *out = sum;
}

// Prod
void dprod(double *v, int64_t vlen, double *out) {
    double prod = 1;
    for (int i = 0; i < vlen; ++i) {
        prod *= v[i];
    }

    *out = prod;
}

void sprod(float *v, int64_t vlen, float *out) {
    float prod = 1;
    for (int i = 0; i < vlen; ++i) {
        prod *= v[i];
    }

    *out = prod;
}

// Mean
void dmean(double *v, int64_t vlen, double *out) {
    double mean = 0;
    for (int i = 0; i < vlen; ++i) {
        mean += v[i];
    }
    mean /= vlen;

    *out = mean;
}

void smean(float *v, int64_t vlen, float *out) {
    float mean = 0;
    for (int i = 0; i < vlen; ++i) {
        mean += v[i];
    }
    mean /= vlen;

    *out = mean;
}

// STD
void dstd(double *v, int64_t vlen, double *out) {
    double mean = 0;
    for (int i = 0; i < vlen; ++i) {
        mean += v[i];
    }
    mean /= vlen;

    double std = 0;
    for (int i = 0; i < vlen; ++i) {
        std += pow((v[i] - mean),  2);
    }
    std /= vlen;
    std = sqrt(std);

    *out = std;
}

void sstd(float *v, int64_t vlen, float *out) {
    float mean = 0;
    for (int i = 0; i < vlen; ++i) {
        mean += v[i];
    }
    mean /= vlen;

    float std = 0;
    for (int i = 0; i < vlen; ++i) {
        std += powf((v[i] - mean),  2);
    }
    std /= vlen;
    std = sqrtf(std);

    *out = std;
}
