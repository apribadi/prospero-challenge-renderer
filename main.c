#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "render.h"
#include "prospero.c"

#define XMIN -1.0f
#define XMAX +1.0f
#define YMIN -1.0f
#define YMAX +1.0f
#define ITERATION_COUNT 500
#define WARMUP true
#define NUM_RESOLUTIONS 4

static uint8_t IMAGE_512[512][512];

static uint8_t IMAGE_1024[1024][1024];

static uint8_t IMAGE_2048[2048][2048];

static uint8_t IMAGE_4096[4096][4096];

static size_t RESOLUTIONS[NUM_RESOLUTIONS] = {
  512,
  1024,
  2048,
  4096,
};

static uint8_t * IMAGES[NUM_RESOLUTIONS] = {
  (uint8_t *) IMAGE_512,
  (uint8_t *) IMAGE_1024,
  (uint8_t *) IMAGE_2048,
  (uint8_t *) IMAGE_4096,
};

static char * FILE_NAMES[NUM_RESOLUTIONS] = {
  "prospero-512.pgm",
  "prospero-1024.pgm",
  "prospero-2048.pgm",
  "prospero-4096.pgm",
};

int main(int, char **) {
  for (size_t k = 0; k < NUM_RESOLUTIONS; k ++) {
    if (WARMUP) {
      for (size_t i = 0; i < ITERATION_COUNT; i ++) {
        render(
            PROSPERO_GEOMETRY,
            PROSPERO_CODE_LEN,
            PROSPERO_CODE,
            XMIN,
            XMAX,
            YMIN,
            YMAX,
            RESOLUTIONS[k],
            (uint8_t (*)[RESOLUTIONS[k]]) IMAGES[k]
          );
      }
    }

    // which clock ???
    // non-portable
    uint64_t start = clock_gettime_nsec_np(CLOCK_REALTIME);

    for (size_t i = 0; i < ITERATION_COUNT; i ++) {
      render(
          PROSPERO_GEOMETRY,
          PROSPERO_CODE_LEN,
          PROSPERO_CODE,
          XMIN,
          XMAX,
          YMIN,
          YMAX,
          RESOLUTIONS[k],
          (uint8_t (*)[RESOLUTIONS[k]]) IMAGES[k]
        );
    }

    uint64_t stop = clock_gettime_nsec_np(CLOCK_REALTIME);

    printf(
        "rendered %dx%d image %d times at %.3f ms per frame ...\n",
        (int) RESOLUTIONS[k],
        (int) RESOLUTIONS[k],
        ITERATION_COUNT,
        (double) (stop - start) / 1000000.0 / ITERATION_COUNT
      );
  }

  for (size_t k = 0; k < NUM_RESOLUTIONS; k ++) {
    render(
        PROSPERO_GEOMETRY,
        PROSPERO_CODE_LEN,
        PROSPERO_CODE,
        XMIN,
        XMAX,
        YMIN,
        YMAX,
        RESOLUTIONS[k],
        (uint8_t (*)[RESOLUTIONS[k]]) IMAGES[k]
      );

    FILE * out = fopen(FILE_NAMES[k], "w");
    if (! out) return 1;
    if (fprintf(out, "P5\n") < 0) return 1;
    if (fprintf(out, "%d\n", (int) RESOLUTIONS[k]) < 0) return 1;
    if (fprintf(out, "%d\n", (int) RESOLUTIONS[k]) < 0) return 1;
    if (fprintf(out, "255\n") < 0) return 1;
    if (fwrite(IMAGES[k], RESOLUTIONS[k] * RESOLUTIONS[k], 1, out) != 1) return 1;
    if (fclose(out) != 0) return 1;

    printf("wrote file %s ...\n", FILE_NAMES[k]);
  }

  return 0;
}
