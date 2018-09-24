//
// Created by Francesc Alted on 11/09/2018.
//

#include <stdio.h>
#include "blosc.h"
#include "iarray.h"

/*
  Example program demonstrating how to execute find roots in a super-chunk.

  To compile this program:

  $ gcc -O3 find_roots_schunk.c -o find_roots_schunk -lblosc

  To run:

  $ ./find_roots_schunk
  ...

*/


const float KB = (float)1024.;
const float MB = 1024 * KB;
const float GB = 1024 * MB;


const int NCHUNKS = 500;
const int CHUNKSIZE = 200 * 1000;  // fits well in modern L3 caches
const int NTHREADS = 4;


// Fill X values in regular array
int fill_x(double *x) {
  double incx = 10. / (NCHUNKS * CHUNKSIZE);

  /* Fill even values between 0 and 10 */
  for (int i = 0; i < NCHUNKS * CHUNKSIZE; i++) {
    x[i] = incx * i;
  }
  return 0;
}

// Compute and fill X values in a buffer
void fill_buffer(double *x, int nchunk) {
  double incx = 10. / (NCHUNKS * CHUNKSIZE);

  for (int i = 0; i < CHUNKSIZE; i++) {
    x[i] = incx * (nchunk * CHUNKSIZE + i);
  }
}

void fill_sc_x(blosc2_schunk *sc_x, const size_t isize) {
  double buffer_x[CHUNKSIZE];

  /* Fill with even values between 0 and 10 */
  for (int nchunk = 0; nchunk < NCHUNKS; nchunk++) {
    fill_buffer(buffer_x, nchunk);
    blosc2_schunk_append_buffer(sc_x, buffer_x, isize);
  }
}

double poly(const double x) {
  return (x - 1.35) * (x - 4.45) * (x - 8.5);
}

// Compute and fill Y values in regular array
void compute_y(const double *x, double *y) {
  for (int i = 0; i < NCHUNKS * CHUNKSIZE; i++) {
    y[i] = poly(x[i]);
  }
}

// Compute and fill Y values in a buffer
void fill_buffer_y(const double *x, double *y) {
  for (int i = 0; i < CHUNKSIZE; i++) {
    y[i] = poly(x[i]);
  }
}

void find_root(const double *x, const double *y,
               const double prev_value) {
  double pv = prev_value;
  int last_root_idx = -1;

  for (int i = 0; i < CHUNKSIZE; i++) {
    double yi = y[i];
    if (((yi > 0) - (yi < 0)) != ((pv > 0) - (pv < 0))) {
      if (last_root_idx != (i - 1)) {
        printf("%.16g, ", x[i]);
        last_root_idx = i;  // avoid the last point (ULP effects)
      }
    }
    pv = yi;
  }
}


int find_roots(void) {
  const size_t isize = CHUNKSIZE * sizeof(double);
  double buffer_x[CHUNKSIZE];
  double buffer_y[CHUNKSIZE];
  int dsize;
  blosc2_cparams cparams = BLOSC_CPARAMS_DEFAULTS;
  blosc2_dparams dparams = BLOSC_DPARAMS_DEFAULTS;
  blosc2_schunk *sc_x, *sc_y;
  int nchunk;
  blosc_timestamp_t last, current;
  double ttotal;
  double prev_value;

  /* Create a super-chunk container for input (X values) */
  cparams.typesize = sizeof(double);
  cparams.compcode = BLOSC_LZ4;
  cparams.clevel = 5;
  cparams.filters[0] = BLOSC_TRUNC_PREC;
  cparams.filters_meta[0] = 23;  // treat doubles as floats
  cparams.nthreads = NTHREADS;
  dparams.nthreads = NTHREADS;

  // Fill the plain x operand
  static double x[NCHUNKS * CHUNKSIZE];
  blosc_set_timestamp(&last);
  fill_x(x);
  blosc_set_timestamp(&current);
  ttotal = blosc_elapsed_secs(last, current);
  printf("Time for filling X values: %.3g s, %.1f MB/s\n",
         ttotal, sizeof(x) / (ttotal * MB));

  // Create and fill a super-chunk for the x operand
  sc_x = blosc2_new_schunk(cparams, dparams, NULL);
  blosc_set_timestamp(&last);
  fill_sc_x(sc_x, isize);
  blosc_set_timestamp(&current);
  ttotal = blosc_elapsed_secs(last, current);
  printf("Time for filling X values (compressed): %.3g s, %.1f MB/s\n",
         ttotal, sc_x->nbytes / (ttotal * MB));
  printf("Compression for X values: %.1f MB -> %.1f MB (%.1fx)\n",
         sc_x->nbytes / MB, sc_x->cbytes / MB,
         (1. * sc_x->nbytes) / sc_x->cbytes);

  // Compute the plain y vector
  static double y[NCHUNKS * CHUNKSIZE];
  blosc_set_timestamp(&last);
  compute_y(x, y);
  blosc_set_timestamp(&current);
  ttotal = blosc_elapsed_secs(last, current);
  printf("Time for computing and filling Y values: %.3g s, %.1f MB/s\n",
         ttotal, sizeof(x) / (ttotal * MB));

  // Create a super-chunk container and compute y values
  sc_y = blosc2_new_schunk(cparams, dparams, NULL);
  blosc_set_timestamp(&last);
  for (nchunk = 0; nchunk < NCHUNKS; nchunk++) {
    dsize = blosc2_schunk_decompress_chunk(sc_x, nchunk, buffer_x, isize);
    if (dsize < 0) {
      printf("Decompression error.  Error code: %d\n", dsize);
      return dsize;
    }
    fill_buffer_y(buffer_x, buffer_y);
    blosc2_schunk_append_buffer(sc_y, buffer_y, isize);
  }
  blosc_set_timestamp(&current);
  ttotal = blosc_elapsed_secs(last, current);
  printf("Time for computing and filling Y values (compressed): %.3g s, %.1f MB/s\n",
         ttotal, sc_y->nbytes / (ttotal * MB));
  printf("Compression for Y values: %.1f MB -> %.1f MB (%.1fx)\n",
         sc_y->nbytes / MB, sc_y->cbytes / MB,
         (1. * sc_y->nbytes) / sc_y->cbytes);

  // Find the roots of the polynomial
  printf("Roots found at: ");
  blosc_set_timestamp(&last);
  prev_value = y[0];
  for (nchunk = 0; nchunk < NCHUNKS; nchunk++) {
    find_root(x + nchunk * CHUNKSIZE, y + nchunk * CHUNKSIZE, prev_value);
    prev_value = y[(nchunk + 1) * CHUNKSIZE - 1];
  }
  blosc_set_timestamp(&current);
  ttotal = blosc_elapsed_secs(last, current);
  printf("\n");
  printf("Find root time:  %.3g s, %.1f MB/s\n",
         ttotal, 2. * sizeof(x) / (ttotal * MB));    // 2 super-chunks involved

  // Find the roots of the polynomial
  printf("Roots found at: ");
  blosc_set_timestamp(&last);
  prev_value = buffer_y[0];
  for (nchunk = 0; nchunk < NCHUNKS; nchunk++) {
    dsize = blosc2_schunk_decompress_chunk(sc_y, nchunk, (void *) buffer_y, isize);
    if (dsize < 0) {
      printf("Decompression error.  Error code: %d\n", dsize);
      return dsize;
    }
    dsize = blosc2_schunk_decompress_chunk(sc_x, nchunk, (void *) buffer_x, isize);
    if (dsize < 0) {
      printf("Decompression error.  Error code: %d\n", dsize);
      return dsize;
    }
    find_root(buffer_x, buffer_y, prev_value);
    prev_value = buffer_y[CHUNKSIZE - 1];
  }
  blosc_set_timestamp(&current);
  ttotal = blosc_elapsed_secs(last, current);
  printf("\n");
  printf("Find root time (compressed):  %.3g s, %.1f MB/s\n",
         ttotal, (sc_x->nbytes + sc_y->nbytes) / (ttotal * MB));    // 2 super-chunks involved

  // Free resources
  blosc2_free_schunk(sc_x);
  blosc2_free_schunk(sc_y);
  return 0;
}


int main() {
  printf("Blosc version info: %s (%s)\n",
         BLOSC_VERSION_STRING, BLOSC_VERSION_DATE);

  blosc_init();

  find_roots();

  blosc_destroy();

  return 0;
}
