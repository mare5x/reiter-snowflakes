// CUDA version using shared memory.

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define USE_CUDA
#include "constants.h"
#include "timer.h"


// "odd-q" vertical layout for hexagonal grid based on:
// https://www.redblobgames.com/grids/hexagons-v1/
typedef struct {
    int q, r;
} Hex;

// This must be `__constant__` for performance!
__device__ __constant__ Hex oddq_directions[2][6] = {
   { {+1,  0}, {+1, -1}, { 0, -1},
     {-1, -1}, {-1,  0}, { 0, +1} },
   { {+1, +1}, {+1,  0}, { 0, -1},
     {-1,  0}, {-1, +1}, { 0, +1} }
};

__host__ __device__ static inline 
float get_s(const float* grid, Hex hex) {
    return grid[COLS * hex.r + hex.q];
}

__host__ __device__ static inline 
float __shared__get_s(const float* sgrid, int srows, int scols, Hex shex) {
    return sgrid[scols * shex.r + shex.q];
}

__host__ __device__ static inline
void set_s(float* grid, Hex hex, float s) {
    grid[COLS * hex.r + hex.q] = s;
}

__host__ __device__ static inline 
bool is_on_edge(Hex hex) {
    return hex.q == 0 || hex.q == COLS - 1 || hex.r == 0 || hex.r == ROWS - 1;
}

__device__ static
bool has_frozen_neighbor(const float* grid, Hex hex) {
    // Assume inner hex.
    int parity = hex.q & 1;
    for (int i = 0; i < 6; ++i) {
        Hex dir = oddq_directions[parity][i];
        Hex neighbor = { hex.q + dir.q, hex.r + dir.r };
        if (!is_on_edge(neighbor) && get_s(grid, neighbor) >= 1.0f) {
            return true;
        }
    }
    return false;
}

__device__ static
bool __shared__has_frozen_neighbor(const float* sgrid, int srows, int scols, Hex hex, Hex shex) {
    // Assume inner hex.
    int parity = hex.q & 1;
    for (int i = 0; i < 6; ++i) {
        Hex dir = oddq_directions[parity][i];
        Hex neighbor = { hex.q + dir.q, hex.r + dir.r };
        Hex sneighbor = { shex.q + dir.q, shex.r + dir.r };
        if (!is_on_edge(neighbor) && __shared__get_s(sgrid, srows, scols, sneighbor) >= 1.0f) {
            return true;
        }
    }
    return false;
}

__device__ static
bool __shared__is_receptive(const float* sgrid, int srows, int scols, Hex hex, Hex shex) {
    if (is_on_edge(hex)) return false;
    if (__shared__get_s(sgrid, srows, scols, shex) >= 1.0f) return true;  // frozen
    return __shared__has_frozen_neighbor(sgrid, srows, scols, hex, shex);
}

// s = **u** + v
__device__ static inline 
float __shared__get_u_part(const float* sgrid, int srows, int scols, Hex hex, Hex shex) {
    return __shared__is_receptive(sgrid, srows, scols, hex, shex)
        ? 0.0f : __shared__get_s(sgrid, srows, scols, shex);
}

__device__ static
float __shared__sum_neighbors_u(const float* sgrid, int srows, int scols, Hex hex, Hex shex) {
    // Assume inner hex.
    float sum = 0.0f;
    int parity = hex.q & 1;
    for (int i = 0; i < 6; ++i) {
        Hex dir = oddq_directions[parity][i];
        Hex neighbor = { hex.q + dir.q, hex.r + dir.r };
        Hex sneighbor = { shex.q + dir.q, shex.r + dir.r };
        sum += __shared__get_u_part(sgrid, srows, scols, neighbor, sneighbor);
    }
    return sum;
}

__device__ static inline
int clamp(int x, int a, int b) {
  const int t = x < a ? a : x;
  return t > b ? b : t;
}

// Initialize grid with `s_0(z)`.
__global__ static
void init_grid_kernel(float* d_grid) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (q < COLS && r < ROWS) {
        float s = (q == COLS / 2 && r == ROWS / 2) ? 1.0f : BETA;
        set_s(d_grid, (Hex){q, r}, s);
    }
}

__global__ static
void update_state_kernel(const float* grid, float* grid_new) {
    // Notes about shared memory:
    // - functions for the shared memory grid are prefixed with __shared__func
    // - we can't reuse the normal code because for shared memory we need
    //   to keep track of both the hex position in shared memory and the global
    //   hex position to be able to determine if a hex is on the global grid edge.
    // - we copy a few cells more than is strictly necessary to shared memory
    //   for the sake of code simplicity
    // - we need two layers of neighbors because we need to check if a neighbor 
    //   has a frozen neighbor
    // - we need to be careful when checking column parity
    const int sborder = 2;  
    const int scols = BLOCK_SIZE + sborder * 2;
    const int srows = BLOCK_SIZE + sborder * 2;
    __shared__ float sgrid[scols * srows];

    const int gq = blockIdx.x * blockDim.x - sborder;
    const int gr = blockIdx.y * blockDim.y - sborder;
    int si = threadIdx.y * blockDim.x + threadIdx.x;
    while (si < scols * srows) {
        const int sq = clamp(si % scols + gq, 0, COLS - 1);
        const int sr = clamp(si / scols + gr, 0, ROWS - 1);
        sgrid[si] = get_s(grid, { sq, sr });
        si += blockDim.x * blockDim.y;
    }
    __syncthreads();

    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= 1 && q < COLS - 1 && r >= 1 && r < ROWS - 1) {
        const Hex hex = { q, r };
        const Hex shex = { (int)threadIdx.x + sborder, (int)threadIdx.y + sborder };
        float s = __shared__get_s(sgrid, srows, scols, shex);
        s += ALPHA / 2.0f * (__shared__sum_neighbors_u(sgrid, srows, scols, hex, shex) / 6.0f 
                           - __shared__get_u_part(sgrid, srows, scols, hex, shex));
        s += GAMMA * __shared__is_receptive(sgrid, srows, scols, hex, shex);
        set_s(grid_new, hex, s);
    }
}


// Code from: https://www.redblobgames.com/grids/hexagons/more-pixel-to-hex.html#saevax
// TODO this can be optimized
__device__ static
void px_to_hex(float x, float y, float edge, int* q, int* r) {
    const float diameter = edge * sqrtf(3);
    const float radius = diameter / 2;

    float t1 = x / edge;
    float t2 = y / diameter;

    *r = (int)floorf((floorf(y / radius) + floorf(t2 - t1) + 2) / 3);
    *q = (int)floorf((floorf(t1 - t2) + floorf(t1 + t2) + 2) / 3);
}

// Hexagon grid to grayscale image.
__global__ static
void image_kernel(const float* grid, unsigned char* img, float hex_edge) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    // TODO Grid-stride loop to support arbitrary grid size
    if (px >= IMG_W || py >= IMG_H) return;
    
    // Center coordinates for px_to_hex
    float x = (float)px - (float)IMG_W / 2.0f;
    float y = (float)py - (float)IMG_H / 2.0f + hex_edge / 2.0f;
    
    int q, r;
    px_to_hex(x, y, hex_edge, &q, &r);
    r = r + (q - (q & 1)) / 2;  // Axial to offset coordinates
    q += COLS / 2;
    r += ROWS / 2;

    unsigned char color = 0;
    if (r >= 0 && r < ROWS && q >= 0 && q < COLS) {
        Hex hex = { q, r };
        float s = get_s(grid, hex);
        if (is_on_edge(hex)) color = 50;
        else if (s >= 1.0f) color = 255;
        else if (has_frozen_neighbor(grid, hex)) {
            color = (unsigned char)(128 + s * (200 - 128));
        } else {
            color = (unsigned char)(50 + s * (128 - 50));
        }   
    }
    img[py * IMG_W + px] = color;
}

static void write_image(
    char const* filename, 
    float* d_grid, 
    unsigned char* d_img, 
    unsigned char* h_img, 
    size_t img_bytes
) {
    timer_start_iter(FIELD_IMAGE_DRAW);
    // Fit hexagons to image dimensions:
    const float hex_edge_x = (float)(IMG_W - 2 * PADDING) / (0.5f + 1.5f * (float)COLS);
    const float hex_edge_y = (float)(IMG_H - 2 * PADDING) / (sqrtf(3) * ((float)ROWS + 0.5f));
    const float hex_edge = fminf(hex_edge_x, hex_edge_y);
    // Useful relationships:
    // const float hex_width = 2 * hex_edge;
    // const float hex_height = sqrt(3) / 2 * hex_width;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((IMG_W - 1) / block_size.x + 1,
                   (IMG_H - 1) / block_size.y + 1);
    image_kernel<<<grid_size, block_size>>>(d_grid, d_img, hex_edge);
    
    cudaMemcpy(h_img, d_img, img_bytes, cudaMemcpyDeviceToHost);
    timer_end_iter(FIELD_IMAGE_DRAW);

    timer_start_iter(FIELD_IMAGE_WRITE);
    stbi_write_png(filename, IMG_W, IMG_H, 1, h_img, IMG_W);
    timer_end_iter(FIELD_IMAGE_WRITE);
}

int main(int argc, char** argv) {
    timer_start_main(FIELD_TOTAL);
    timer_start_main(FIELD_INIT);

    // N.B. There is no need for a host grid (only for debugging).
    // All data is on the device, except when writing an image to disk.
    size_t grid_bytes = ROWS * COLS * sizeof(float);
    float *d_grid1, *d_grid2;
    cudaMalloc(&d_grid1, grid_bytes);
    cudaMalloc(&d_grid2, grid_bytes);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((COLS - 1) / block_size.x + 1,
                   (ROWS - 1) / block_size.y + 1);
    init_grid_kernel<<<grid_size, block_size>>>(d_grid1);
    cudaMemcpy(d_grid2, d_grid1, grid_bytes, cudaMemcpyDeviceToDevice);

    // A single image buffer is stored per host/device.
    size_t img_bytes = 1 * IMG_W * IMG_H;  // grayscale image
    unsigned char* h_img = (unsigned char*)malloc(img_bytes);
    unsigned char* d_img;
    cudaMalloc(&d_img, img_bytes);

    timer_end_main(FIELD_INIT);

    char filename[256];
    sprintf(filename, "%s/0.png", IMG_DIR);
    write_image(filename, d_grid1, d_img, h_img, img_bytes);

    for (int iter = 1; iter <= ITERS; ++iter) {
        timer_start_iter(FIELD_UPDATE);
        update_state_kernel<<<grid_size, block_size>>>(d_grid1, d_grid2);
        float* tmp = d_grid1;  
        d_grid1 = d_grid2;
        d_grid2 = tmp;
        timer_end_iter(FIELD_UPDATE);

        if (iter % IMG_ITERS == 0) {
            sprintf(filename, "%s/%d.png", IMG_DIR, iter);
            write_image(filename, d_grid1, d_img, h_img, img_bytes);
            timer_print_avg(ITER_SCOPE);
        }
        timer_step_iter();
    }
    timer_end_main(FIELD_TOTAL);
    timer_step_main();

    cudaFree(d_grid1);
    cudaFree(d_grid2);
    cudaFree(d_img);
    free(h_img);

    timer_print_csv(stdout, MAIN_SCOPE);
    timer_write_csv(MAIN_SCOPE);
    timer_write_csv(ITER_SCOPE);

    return 0;
}