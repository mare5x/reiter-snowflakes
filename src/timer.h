#ifndef TIMER_H
#define TIMER_H

#include "omp.h"
#include "constants.h"

#if defined(USE_CUDA) || defined(USE_CUDA_MULTI)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define MAX_MAIN 1024
#define MAX_ITER ITERS + 3


enum TIMER_SCOPE {
    MAIN_SCOPE, ITER_SCOPE, N_TIMER_SCOPE
};
const char* TIMER_SCOPE_STR[] = {
    "main", "iter"
};

enum MAIN_TIMER_FIELD {
    FIELD_TOTAL,
    FIELD_INIT,
    N_MAIN_FIELD,
};
const char* MAIN_TIMER_FIELD_STR[] = {
    "total", "init"
};

enum ITER_TIMER_FIELD {
    FIELD_UPDATE,
    FIELD_IMAGE_DRAW,
    FIELD_IMAGE_WRITE,
    N_ITER_FIELDS,
};
const char* ITER_TIMER_FIELD_STR[] = {
    "update", "image_draw", "image_write"
};


double main_timer_data[MAX_MAIN][N_MAIN_FIELD];
double iter_timer_data[MAX_ITER][N_ITER_FIELDS];
double* timer_data[] = { &main_timer_data[0][0], &iter_timer_data[0][0] };
const int n_fields[] = { N_MAIN_FIELD, N_ITER_FIELDS };
const char** field_str[] = { MAIN_TIMER_FIELD_STR, ITER_TIMER_FIELD_STR };
int counters[N_TIMER_SCOPE];


void timer_start_main(enum MAIN_TIMER_FIELD field) {
    main_timer_data[counters[MAIN_SCOPE]][field] = omp_get_wtime();
}

void timer_end_main(enum MAIN_TIMER_FIELD field) {
    #ifdef USE_CUDA
    cudaDeviceSynchronize();
    #endif
    #ifdef USE_CUDA_MULTI
    for (int device = 0; device < USE_CUDA_MULTI; ++device) {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
    }
    #endif
    double tic = main_timer_data[counters[MAIN_SCOPE]][field];
    main_timer_data[counters[MAIN_SCOPE]][field] = omp_get_wtime() - tic;
}

void timer_step_main() {
    counters[MAIN_SCOPE] += 1;
}

void timer_start_iter(enum ITER_TIMER_FIELD field) {
    iter_timer_data[counters[ITER_SCOPE]][field] = omp_get_wtime();
}

void timer_end_iter(enum ITER_TIMER_FIELD field) {
    #ifdef USE_CUDA
    cudaDeviceSynchronize();
    #endif
    #ifdef USE_CUDA_MULTI
    for (int device = 0; device < USE_CUDA_MULTI; ++device) {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
    }
    #endif
    double tic = iter_timer_data[counters[ITER_SCOPE]][field];
    iter_timer_data[counters[ITER_SCOPE]][field] = omp_get_wtime() - tic;
}

void timer_step_iter() {
    counters[ITER_SCOPE] += 1;
}

void timer_print_avg(enum TIMER_SCOPE scope) {
    static double avg[32];
    static double nonzero[32];
    const double* data = timer_data[scope];
    memset(avg, 0, sizeof(avg));
    memset(nonzero, 0, sizeof(nonzero));
    for (int i = 0; i < counters[scope]; ++i) {
        for (int j = 0; j < n_fields[scope]; ++j) {
            double x = data[i * n_fields[scope] + j];
            if (x > 0) {
                avg[j] += x;
                nonzero[j] += 1;
            }
        }
    }
    printf("%4d: ", counters[scope] + 1);
    for (int j = 0; j < n_fields[scope]; ++j) {
        if (j) printf(" ");
        if (nonzero[j] > 0)
            printf("%f", avg[j] / nonzero[j]);
        else
            printf("%f", 0.0);
    }
    printf("\n");
    fflush(stdout);
}

void timer_print_csv(FILE* stream, enum TIMER_SCOPE scope) {
    double* data = timer_data[scope];
    for (int j = 0; j < n_fields[scope]; ++j) {
        fprintf(stream, "%s", field_str[scope][j]);
        if (j < n_fields[scope] - 1) fprintf(stream, ",");
    }
    fprintf(stream, "\n");
    for (int i = 0; i < counters[scope]; ++i) {
        for (int j = 0; j < n_fields[scope]; ++j) {
            fprintf(stream, "%f", data[i * n_fields[scope] + j]);
            if (j < n_fields[scope] - 1) fprintf(stream, ",");
        }
        fprintf(stream, "\n");
    }
}

void timer_write_csv(enum TIMER_SCOPE scope) {
    char filename[256];
    sprintf(filename, "%s/%s.log", IMG_DIR, TIMER_SCOPE_STR[scope]);
    FILE* f = fopen(filename, "w");
    timer_print_csv(f, scope);
    fclose(f);
    printf("%s\n", filename);
}

#endif  // TIMER_H