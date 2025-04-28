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
#define ITERATION_COUNT 100
#define WARMUP true
#define MIN_BENCH_RESOLUTION 512
#define MAX_BENCH_RESOLUTION 4096
#define FILE_RESOLUTION 1024

static uint8_t BENCH_IMAGE[MAX_BENCH_RESOLUTION * MAX_BENCH_RESOLUTION];

static uint8_t FILE_IMAGE[FILE_RESOLUTION][FILE_RESOLUTION];

int main(int, char **) {
  for (size_t r = MIN_BENCH_RESOLUTION; r <= MAX_BENCH_RESOLUTION; r *= 2) {
    if (WARMUP) {
      for (size_t i = 0; i < ITERATION_COUNT; i ++) {
        render(
            sizeof(PROSPERO) / sizeof(Inst),
            PROSPERO,
            XMIN,
            XMAX,
            YMIN,
            YMAX,
            r,
            (uint8_t (*)[r]) BENCH_IMAGE
          );
      }
    }

    // which clock ???
    // non-portable
    uint64_t start = clock_gettime_nsec_np(CLOCK_REALTIME);

    for (size_t i = 0; i < ITERATION_COUNT; i ++) {
      render(
          sizeof(PROSPERO) / sizeof(Inst),
          PROSPERO,
          XMIN,
          XMAX,
          YMIN,
          YMAX,
          r,
          (uint8_t (*)[r]) BENCH_IMAGE
        );
    }

    uint64_t stop = clock_gettime_nsec_np(CLOCK_REALTIME);

    printf(
        "rendered %dx%d image %d times at %.2f ms per frame ...\n",
        (int) r,
        (int) r,
        ITERATION_COUNT,
        (double) (stop - start) / 1000000.0 / ITERATION_COUNT
      );
  }

  render(
      sizeof(PROSPERO) / sizeof(Inst),
      PROSPERO,
      XMIN,
      XMAX,
      YMIN,
      YMAX,
      FILE_RESOLUTION,
      FILE_IMAGE
    );

  FILE * out = fopen("prospero.pgm", "w");
  if (! out) return 1;
  if (fprintf(out, "P5\n") < 0) return 1;
  if (fprintf(out, "%d\n", FILE_RESOLUTION) < 0) return 1;
  if (fprintf(out, "%d\n", FILE_RESOLUTION) < 0) return 1;
  if (fprintf(out, "255\n") < 0) return 1;
  if (fwrite(FILE_IMAGE, sizeof(FILE_IMAGE), 1, out) != 1) return 1;
  if (fclose(out) != 0) return 1;

  printf("wrote file prospero.pgm ...\n");

  return 0;
}
