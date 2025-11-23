/* Minimal C-compatible graphics.h shim for TURBO_C folder
   This header declares only the symbols used by the example program.
   It avoids C++ headers so the file can be compiled with gcc.

   NOTE: This only provides declarations. The actual implementations
   must be provided by the BGI library (libbgi.a) available in MinGW's lib.
*/

#ifndef TURBO_C_GRAPHICS_H
#define TURBO_C_GRAPHICS_H

#ifdef __cplusplus
extern "C"
{
#endif

/* Common constants */
#define DETECT 0
#define grOk 0

/* Color constants (subset) */
#define BLACK 0
#define BLUE 1
#define GREEN 2
#define CYAN 3
#define RED 4
#define MAGENTA 5
#define BROWN 6
#define LIGHTGRAY 7
#define DARKGRAY 8
#define LIGHTBLUE 9
#define LIGHTGREEN 10
#define LIGHTCYAN 11
#define LIGHTRED 12
#define LIGHTMAGENTA 13
#define YELLOW 14
#define WHITE 15

/* Fill pattern constants for setfillstyle() */
#define EMPTY_FILL 0      /* No fill */
#define SOLID_FILL 1      /* Solid fill */
#define LINE_FILL 2       /* Horizontal lines */
#define LTSLASH_FILL 3    /* Light forward diagonal lines */
#define SLASH_FILL 4      /* Heavy forward diagonal lines */
#define BKSLASH_FILL 5    /* Heavy backward diagonal lines */
#define LTBKSLASH_FILL 6  /* Light backward diagonal lines */
#define HATCH_FILL 7      /* Horizontal and vertical crosshatch */
#define XHATCH_FILL 8     /* Diagonal crosshatch */
#define INTERLEAVE_FILL 9 /* Interleaving pattern */
#define WIDE_DOT_FILL 10  /* Widely spaced dots */
#define CLOSE_DOT_FILL 11 /* Closely spaced dots */
#define USER_FILL 12      /* User-defined fill pattern */

    /* Function prototypes (minimal) */
    int initgraph(int *graphdriver, int *graphmode, const char *path);
    int graphresult(void);
    void closegraph(void);
    int getmaxy(void);
    void outtextxy(int x, int y, const char *text);
    void setcolor(int color);
    void circle(int x, int y, int radius);
    void putpixel(int x, int y, int color);
    int getmaxx(void);
    void cleardevice(void);
    void line(int x, int y, int x2, int y2);
    void rectangle(int top, int left, int right, int bottom);
    void setfillstyle(int pattern, int color);
    void floodfill(int x, int y, int color);
    void drawpoly(int num_points, int *polypoints);
    void setbkcolor(int color);
    void arc(int x, int y, int startangle, int endangle, int radius);
    void bar(int top, int left, int right, int bottom);
    void ellipse(int x, int y, int start_angle, int end_angle, int x_radius, int y_radius);

#ifdef __cplusplus
}
#endif

#endif /* TURBO_C_GRAPHICS_H */
