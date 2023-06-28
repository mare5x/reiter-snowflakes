// CPU multi-threading using OpenMP.
// Just adds a few `#pragma omp ...` statements to the serial code.

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "omp.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "constants.h"
#include "timer.h"


// "odd-q" vertical layout for hexagonal grid based on:
// https://www.redblobgames.com/grids/hexagons-v1/
typedef struct {
    int q, r;
} Hex;

const Hex oddq_directions[2][6] = {
   { {+1,  0}, {+1, -1}, { 0, -1},
     {-1, -1}, {-1,  0}, { 0, +1} },
   { {+1, +1}, {+1,  0}, { 0, -1},
     {-1,  0}, {-1, +1}, { 0, +1} }
};

static inline float get_s(float* grid, Hex hex) {
    return grid[COLS * hex.r + hex.q];
}

static inline void set_s(float* grid, Hex hex, float s) {
    grid[COLS * hex.r + hex.q] = s;
}

static inline bool is_on_edge(Hex hex) {
    return hex.q == 0 || hex.q == COLS - 1 || hex.r == 0 || hex.r == ROWS - 1;
}

bool has_frozen_neighbor(float* grid, Hex hex) {
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

bool is_receptive(float* grid, Hex hex) {
    if (is_on_edge(hex)) return false;
    if (get_s(grid, hex) >= 1.0f) return true;  // frozen
    return has_frozen_neighbor(grid, hex);
}

// s = **u** + v
static inline float get_u_part(float* grid, Hex hex) {
    if (is_receptive(grid, hex)) {
        return 0.0f;
    } else {
        return get_s(grid, hex);
    }
}

float sum_neighbors_u(float* grid, Hex hex) {
    // Assume inner hex.
    float sum = 0.0f;
    int parity = hex.q & 1;
    for (int i = 0; i < 6; ++i) {
        Hex dir = oddq_directions[parity][i];
        Hex neighbor = { hex.q + dir.q, hex.r + dir.r };
        sum += get_u_part(grid, neighbor);
    }
    return sum;
}

// Initialize grid with `s_0(z)`.
float* init_grid(void) {
    float* grid = (float*)malloc(ROWS * COLS * sizeof(float));

    for (int r = 0; r < ROWS; ++r) {
        for (int q = 0; q < COLS; ++q) {
            set_s(grid, (Hex){q, r}, BETA);
        }
    }
    set_s(grid, (Hex){COLS / 2, ROWS / 2}, 1.0f);

    return grid;
}

void update_state(float* grid, float* grid_new) {
    // N.B. skip edge cells
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int r = 1; r < ROWS - 1; ++r) {
        for (int q = 1; q < COLS - 1; ++q) {
            Hex hex = { q, r };
            float s = get_s(grid, hex);
            s += ALPHA / 2.0f * (sum_neighbors_u(grid, hex) / 6.0f - get_u_part(grid, hex));
            s += GAMMA * is_receptive(grid, hex);
            set_s(grid_new, hex, s);
        }
    }
}   

// Code from: https://www.redblobgames.com/grids/hexagons/more-pixel-to-hex.html#saevax
// TODO this can be optimized
static
void px_to_hex(const float x, const float y, const float edge, int* q, int* r)
{
    const float diameter = edge * sqrtf(3);
    const float radius = diameter / 2;

    float t1 = x / edge;
    float t2 = y / diameter;

    *r = (int)floorf((floorf(y / radius) + floorf(t2 - t1) + 2) / 3);
    *q = (int)floorf((floorf(t1 - t2) + floorf(t1 + t2) + 2) / 3);
}

void write_image(
    char const* filename, 
    int w, 
    int h, 
    float* grid, 
    int rows, 
    int cols
) {
    timer_start_iter(FIELD_IMAGE_DRAW);
    unsigned char* img = (unsigned char*)malloc(w * h);

    // Fit hexagons to image dimensions:
    const float hex_edge_x = (float)(IMG_W - 2 * PADDING) / (0.5f + 1.5f * (float)COLS);
    const float hex_edge_y = (float)(IMG_H - 2 * PADDING) / (sqrtf(3) * ((float)ROWS + 0.5f));
    const float hex_edge = fminf(hex_edge_x, hex_edge_y);
    // Useful relationships:
    // const float hex_width = 2 * hex_edge;
    // const float hex_height = sqrt(3) / 2 * hex_width;

    #pragma omp parallel for collapse(2) schedule(guided)
    for (int py = 0; py < h; ++py) {
        for (int px = 0; px < w; ++px) {
            // Center coordinates for px_to_hex
            float x = (float)px - (float)w / 2;
            float y = (float)py - (float)h / 2 + hex_edge / 2;

            int q, r;
            px_to_hex(x, y, hex_edge, &q, &r);
            r = r + (q - (q & 1)) / 2;  // Axial to offset coordinates
            q += cols / 2;
            r += rows / 2;
         
            if (r >= 0 && r < rows && q >= 0 && q < cols) {
                unsigned char color;

                Hex hex = { q, r };
                float s = get_s(grid, hex);

                if (is_on_edge(hex)) color = 50;
                else if (s >= 1.0f) color = 255;
                else if (has_frozen_neighbor(grid, hex)) {
                    color = (unsigned char)(128 + s * (200 - 128));
                } else {
                    color = (unsigned char)(50 + s * (128 - 50));
                }
                
                img[py * w + px] = color;
            } else {
                img[py * w + px] = 0;
            }
        }
    }
    timer_end_iter(FIELD_IMAGE_DRAW);

    timer_start_iter(FIELD_IMAGE_WRITE);
    stbi_write_png(filename, w, h, 1, img, w);
    timer_end_iter(FIELD_IMAGE_WRITE);
    free(img);
}

int main(int argc, char** argv) {
    timer_start_main(FIELD_TOTAL);

    timer_start_main(FIELD_INIT);
    float* grid1 = init_grid();
    float* grid2 = init_grid();
    timer_end_main(FIELD_INIT);

    char filename[256];
    sprintf(filename, "%s/%d.png", IMG_DIR, 0);
    write_image(filename, IMG_W, IMG_H, grid1, ROWS, COLS);

    for (int iter = 1; iter <= ITERS; ++iter) {
        timer_start_iter(FIELD_UPDATE);
        update_state(grid1, grid2);
        float* tmp = grid1;
        grid1 = grid2;
        grid2 = tmp;
        timer_end_iter(FIELD_UPDATE);

        if (iter % IMG_ITERS == 0) {
            sprintf(filename, "%s/%d.png", IMG_DIR, iter);
            write_image(filename, IMG_W, IMG_H, grid1, ROWS, COLS);
            timer_print_avg(ITER_SCOPE);
        }
        timer_step_iter();
    }

    free(grid1);
    free(grid2);
    
    timer_end_main(FIELD_TOTAL);
    timer_step_main();

    timer_print_csv(stdout, MAIN_SCOPE);
    timer_write_csv(MAIN_SCOPE);
    timer_write_csv(ITER_SCOPE);

    return 0;
}