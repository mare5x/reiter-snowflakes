// N.B. The parameters are `#define`ed so that we can change them 
// using the `-D` compiler flag.
#ifndef CONSTANTS_H
#define CONSTANTS_H

// Hexagon grid size:
#ifndef ROWS
    #define ROWS 501
#endif
#ifndef COLS
    #define COLS 501
#endif

#ifndef BORDER_SIZE
    #define BORDER_SIZE 4
#endif

#ifndef ROW_WISE
    #define ROW_WISE 1
#endif

// Simulation parameters:
#ifndef ITERS
    #define ITERS 8000
#endif
#ifndef ALPHA
    #define ALPHA 0.501f
#endif
#ifndef BETA
    #define BETA 0.4f
#endif
#ifndef GAMMA
    #define GAMMA 0.0001f
#endif

// Image output options:
#ifndef IMG_DIR
    #define IMG_DIR "out"     // Output image directory path (make sure it exists)
#endif
#ifndef IMG_ITERS
    #define IMG_ITERS 500     // Save every this many iterations
#endif
#ifndef IMG_W
    #define IMG_W 1024
#endif
#ifndef IMG_H
    #define IMG_H 1024
#endif
#ifndef PADDING
    #define PADDING 8         // Pad each side [px] for a prettier image
#endif

// CUDA parameters:
#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 16
#endif

#endif  // CONSTANTS_H
