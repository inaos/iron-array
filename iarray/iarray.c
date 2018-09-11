//
// Created by Francesc Alted on 11/09/2018.
//

#include <stdio.h>
#include "blosc.h"
#include "iarray.h"

/*
  Example program demonstrating how to execute an expression with super-chunks as operands.

  To compile this program:

  $ gcc -O3 iarray.c -o iarray -lblosc

  To run:

  $ ./iarray
  ...

*/


const float KB = (float)1024.;
const float MB = 1024 * KB;
const float GB = 1024 * MB;


const int NCHUNKS = 500;
const int CHUNKSIZE = 200 * 1000;  // fits well in modern L3 caches
const int NTHREADS = 4;
const size_t isize = CHUNKSIZE * sizeof(double);


void process_data(const double *x, double *y) {

  for (int i = 0; i < CHUNKSIZE; i++) {
    double xi = x[i];
    //y[i] = ((.25 * xi + .75) * xi - 1.5) * xi - 2;
    y[i] = (xi - 1.35) * (xi - 4.45) * (xi - 8.5);
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

void fill_buffer(double *x, int nchunk) {
  double incx = 10. / (NCHUNKS * CHUNKSIZE);

  for (int i = 0; i < CHUNKSIZE; i++) {
    x[i] = incx * (nchunk * CHUNKSIZE + i);
  }
}

// Fill x values in super-chunk
int fill_x(blosc2_schunk* sc_x) {
  long nbytes = 0;
  blosc_timestamp_t last, current;
  double ttotal;
  double buffer_x[CHUNKSIZE];

  /* Now fill the buffer with even values between 0 and 10 */
  blosc_set_timestamp(&last);
  for (int nchunk = 0; nchunk < NCHUNKS; nchunk++) {
    fill_buffer(buffer_x, nchunk);
    blosc2_schunk_append_buffer(sc_x, buffer_x, isize);
    nbytes += isize;
  }
  blosc_set_timestamp(&current);
  ttotal = blosc_elapsed_secs(last, current);
  printf("Creation time for x values: %.3g s, %.1f MB/s\n",
         ttotal, nbytes / (ttotal * MB));
  printf("Compression for x values: %.1f MB -> %.1f MB (%.1fx)\n",
         sc_x->nbytes / MB, sc_x->cbytes / MB,
         (1. * sc_x->nbytes) / sc_x->cbytes);
  return 0;
}


int compute_vectors(void) {
  double buffer_x[CHUNKSIZE];
  double buffer_y[CHUNKSIZE];
  int dsize;
  long nbytes = 0;
  blosc2_cparams cparams = BLOSC_CPARAMS_DEFAULTS;
  blosc2_dparams dparams = BLOSC_DPARAMS_DEFAULTS;
  blosc2_schunk *sc_x, *sc_y;
  int nchunk;
  blosc_timestamp_t last, current;
  double ttotal;
  double prev_value;

  /* Create a super-chunk container for input (X values) */
  cparams.typesize = sizeof(double);
  cparams.compcode = BLOSC_BLOSCLZ;
  cparams.clevel = 5;
  cparams.filters[0] = BLOSC_TRUNC_PREC;
  cparams.filters_meta[0] = 23;  // treat doubles as floats
  cparams.nthreads = NTHREADS;
  dparams.nthreads = NTHREADS;

  //  Create and fill a super-chunk for the x operand
  sc_x = blosc2_new_schunk(cparams, dparams, NULL);
  if (fill_x(sc_x) < 0) {
    fprintf(stderr, "Error filling the x values in super-chunk");
  }

  /* Create a super-chunk container for output (y values) */
  sc_y = blosc2_new_schunk(cparams, dparams, NULL);

  /* Retrieve the chunks and compute the polynomial in another super-chunk */
  blosc_set_timestamp(&last);
  for (nchunk = 0; nchunk < NCHUNKS; nchunk++) {
    dsize = blosc2_schunk_decompress_chunk(sc_x, nchunk, buffer_x, isize);
    if (dsize < 0) {
      printf("Decompression error.  Error code: %d\n", dsize);
      return dsize;
    }
    process_data(buffer_x, buffer_y);
    blosc2_schunk_append_buffer(sc_y, buffer_y, isize);
  }
  blosc_set_timestamp(&current);
  ttotal = blosc_elapsed_secs(last, current);
  printf("Computing Y polynomial: %.3g s, %.1f MB/s\n",
         ttotal,
         2. * nbytes / (ttotal * MB));    // 2 super-chunks involved
  printf("Compression for Y values: %.1f MB -> %.1f MB (%.1fx)\n",
         sc_y->nbytes / MB, sc_y->cbytes / MB,
         (1. * sc_y->nbytes) / sc_y->cbytes);

  /* Find the roots of the polynomial */
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
  printf("Find root time:  %.3g s, %.1f MB/s\n",
         ttotal, 2. * nbytes / (ttotal * MB));    // 2 super-chunks involved

  /* Free resources */
  /* Destroy the super-chunk */
  blosc2_free_schunk(sc_x);
  blosc2_free_schunk(sc_y);
  return 0;
}


int main() {
  printf("Blosc version info: %s (%s)\n",
         BLOSC_VERSION_STRING, BLOSC_VERSION_DATE);

  /* Initialize the Blosc compressor */
  blosc_init();

  compute_vectors();

  /* Destroy the Blosc environment */
  blosc_destroy();

  return 0;
}
