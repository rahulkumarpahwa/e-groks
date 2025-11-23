#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <conio.h>
#include "graphics.h"

#define PI 3.14159265359

/* ============================================
   BOILER PLATE - Basic C Program Structure
   ============================================ */
void boilerplate_main() {
    printf("=== Boiler Plate of C Language ===\n");
    printf("This is a basic C program structure.\n");
}

/* ============================================
   DDA LINE ALGORITHM
   ============================================ */
void dda_line(int x0, int y0, int x1, int y1) {
    int dx = x1 - x0;
    int dy = y1 - y0;
    int steps = abs(dx) > abs(dy) ? abs(dx) : abs(dy);
    
    if (steps == 0) {
        putpixel(x0, y0, WHITE);
        return;
    }
    
    float x_inc = (float)dx / steps;
    float y_inc = (float)dy / steps;
    float x = x0;
    float y = y0;
    
    for (int i = 0; i <= steps; i++) {
        putpixel((int)round(x), (int)round(y), WHITE);
        x += x_inc;
        y += y_inc;
    }
}

void dda_line_demo() {
    int gdriver = DETECT, gmode;
    initgraph(&gdriver, &gmode, "");
    
    if (graphresult() != grOk) {
        printf("Graphics initialization failed\n");
        return;
    }
    
    int max_x = getmaxx();
    int max_y = getmaxy();
    cleardevice();
    
    setcolor(YELLOW);
    outtextxy(max_x/2 - 100, 20, "DDA Line Drawing Algorithm");
    
    setcolor(WHITE);
    dda_line(50, 50, 300, 300);
    
    setcolor(RED);
    dda_line(400, 100, 600, 300);
    
    setcolor(YELLOW);
    outtextxy(20, max_y - 40, "Press any key to exit...");
    
    getch();
    closegraph();
}

/* ============================================
   BRESENHAM LINE ALGORITHM
   ============================================ */
void bresenham_line(int x0, int y0, int x1, int y1) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    int x = x0;
    int y = y0;
    
    while (1) {
        putpixel(x, y, WHITE);
        
        if (x == x1 && y == y1) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

void bresenham_line_demo() {
    int gdriver = DETECT, gmode;
    initgraph(&gdriver, &gmode, "");
    
    if (graphresult() != grOk) {
        printf("Graphics initialization failed\n");
        return;
    }
    
    int max_x = getmaxx();
    int max_y = getmaxy();
    cleardevice();
    
    setcolor(YELLOW);
    outtextxy(max_x / 2 - 120, 20, "Bresenham Line Drawing Algorithm");
    
    setcolor(WHITE);
    bresenham_line(50, 50, 300, 300);
    
    setcolor(RED);
    bresenham_line(400, 100, 600, 300);
    
    setcolor(YELLOW);
    outtextxy(20, max_y - 40, "Press any key to exit...");
    
    getch();
    closegraph();
}

/* ============================================
   DIRECT METHOD LINE ALGORITHM
   ============================================ */
void direct_line_demo() {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "");
    
    int x0, y0, x1, y1;
    printf("Enter x0 y0 x1 y1: ");
    scanf("%d %d %d %d", &x0, &y0, &x1, &y1);
    
    int dx = x1 - x0;
    int dy = y1 - y0;
    
    if (dx == 0) {
        int ys = (y0 < y1) ? y0 : y1;
        int ye = (y0 < y1) ? y1 : y0;
        for (int y = ys; y <= ye; ++y)
            putpixel(x0, y, WHITE);
    } else if (dy == 0) {
        int xs = (x0 < x1) ? x0 : x1;
        int xe = (x0 < x1) ? x1 : x0;
        for (int x = xs; x <= xe; ++x)
            putpixel(x, y0, WHITE);
    } else {
        double m = (double)dy / (double)dx;
        if (fabs(m) <= 1.0) {
            int xs = (x0 < x1) ? x0 : x1;
            int xe = (x0 < x1) ? x1 : x0;
            int base = (x0 < x1) ? x0 : x1;
            int otherY = (x0 < x1) ? y0 : y1;
            for (int x = xs; x <= xe; ++x) {
                double y = otherY + m * (x - base);
                putpixel(x, (int)round(y), WHITE);
            }
        }
    }
    
    getch();
    closegraph();
}

/* ============================================
   INCREMENTAL LINE ALGORITHM
   ============================================ */
void incremental_line_demo() {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "");
    
    int x0, y0, x1, y1;
    printf("Enter x0 y0 x1 y1: ");
    scanf("%d %d %d %d", &x0, &y0, &x1, &y1);
    
    int dx = x1 - x0;
    int dy = y1 - y0;
    int steps = abs(dx) > abs(dy) ? abs(dx) : abs(dy);
    
    if (steps == 0) {
        putpixel(x0, y0, WHITE);
    } else {
        double xincr = (double)dx / (double)steps;
        double yincr = (double)dy / (double)steps;
        double x = x0, y = y0;
        for (int i = 0; i <= steps; ++i) {
            putpixel((int)round(x), (int)round(y), WHITE);
            x += xincr;
            y += yincr;
        }
    }
    
    getch();
    closegraph();
}

/* ============================================
   MIDPOINT CIRCLE ALGORITHM
   ============================================ */
void midpoint_circle(int xc, int yc, int r) {
    int x = 0;
    int y = r;
    int d = 3 - 2 * r;
    
    while (x <= y) {
        putpixel(xc + x, yc + y, WHITE);
        putpixel(xc - x, yc + y, WHITE);
        putpixel(xc + x, yc - y, WHITE);
        putpixel(xc - x, yc - y, WHITE);
        putpixel(xc + y, yc + x, WHITE);
        putpixel(xc - y, yc + x, WHITE);
        putpixel(xc + y, yc - x, WHITE);
        putpixel(xc - y, yc - x, WHITE);
        
        if (d < 0) d = d + 4 * x + 6;
        else {
            d = d + 4 * (x - y) + 10;
            y--;
        }
        x++;
    }
}

void midpoint_circle_demo() {
    int gdriver = DETECT, gmode;
    initgraph(&gdriver, &gmode, "");
    
    if (graphresult() != grOk) {
        printf("Graphics initialization failed\n");
        return;
    }
    
    int max_x = getmaxx();
    int max_y = getmaxy();
    cleardevice();
    
    setcolor(YELLOW);
    outtextxy(max_x / 2 - 100, 20, "Midpoint Circle Drawing Algorithm");
    
    setcolor(WHITE);
    midpoint_circle(max_x / 2, max_y / 2, 50);
    
    setcolor(RED);
    midpoint_circle(max_x / 2, max_y / 2, 100);
    
    setcolor(YELLOW);
    outtextxy(20, max_y - 40, "Press any key to exit...");
    
    getch();
    closegraph();
}

/* ============================================
   BRESENHAM CIRCLE ALGORITHM
   ============================================ */
void bresenham_circle(int xc, int yc, int r) {
    int x = 0;
    int y = r;
    int d = 3 - 2 * r;
    
    while (x <= y) {
        putpixel(xc + x, yc + y, WHITE);
        putpixel(xc - x, yc + y, WHITE);
        putpixel(xc + x, yc - y, WHITE);
        putpixel(xc - x, yc - y, WHITE);
        putpixel(xc + y, yc + x, WHITE);
        putpixel(xc - y, yc + x, WHITE);
        putpixel(xc + y, yc - x, WHITE);
        putpixel(xc - y, yc - x, WHITE);
        
        if (d < 0) d = d + 4 * x + 6;
        else {
            d = d + 4 * (x - y) + 10;
            y--;
        }
        x++;
    }
}

void bresenham_circle_demo() {
    int gdriver = DETECT, gmode;
    initgraph(&gdriver, &gmode, "");
    
    if (graphresult() != grOk) {
        printf("Graphics initialization failed\n");
        return;
    }
    
    int max_x = getmaxx();
    int max_y = getmaxy();
    cleardevice();
    
    setcolor(YELLOW);
    outtextxy(max_x/2 - 120, 20, "Bresenham Circle Drawing Algorithm");
    
    setcolor(WHITE);
    bresenham_circle(150, 200, 50);
    
    setcolor(RED);
    bresenham_circle(150, 200, 100);
    
    setcolor(YELLOW);
    outtextxy(20, max_y - 40, "Press any key to exit...");
    
    getch();
    closegraph();
}

/* ============================================
   COHEN-SUTHERLAND LINE CLIPPING
   ============================================ */
#define INSIDE 0
#define LEFT 1
#define RIGHT 2
#define BOTTOM 4
#define TOP 8

typedef struct { int x1, y1, x2, y2; } Line;
typedef struct { int xmin, ymin, xmax, ymax; } Window;

int computeCode(int x, int y, Window window) {
    int code = INSIDE;
    if (x < window.xmin) code |= LEFT;
    else if (x > window.xmax) code |= RIGHT;
    if (y < window.ymin) code |= BOTTOM;
    else if (y > window.ymax) code |= TOP;
    return code;
}

int cohen_sutherland(Line *line, Window window) {
    int code1 = computeCode(line->x1, line->y1, window);
    int code2 = computeCode(line->x2, line->y2, window);
    int accept = 0;
    
    while (1) {
        if ((code1 | code2) == 0) {
            accept = 1;
            break;
        } else if ((code1 & code2) != 0) {
            break;
        } else {
            int code_out = code1 ? code1 : code2;
            int x, y;
            
            if (code_out & TOP) {
                x = line->x1 + (line->x2 - line->x1) * (window.ymax - line->y1) / (line->y2 - line->y1);
                y = window.ymax;
            } else if (code_out & BOTTOM) {
                x = line->x1 + (line->x2 - line->x1) * (window.ymin - line->y1) / (line->y2 - line->y1);
                y = window.ymin;
            } else if (code_out & RIGHT) {
                y = line->y1 + (line->y2 - line->y1) * (window.xmax - line->x1) / (line->x2 - line->x1);
                x = window.xmax;
            } else {
                y = line->y1 + (line->y2 - line->y1) * (window.xmin - line->x1) / (line->x2 - line->x1);
                x = window.xmin;
            }
            
            if (code_out == code1) {
                line->x1 = x;
                line->y1 = y;
                code1 = computeCode(line->x1, line->y1, window);
            } else {
                line->x2 = x;
                line->y2 = y;
                code2 = computeCode(line->x2, line->y2, window);
            }
        }
    }
    return accept;
}

void cohen_sutherland_demo() {
    int gdriver = DETECT, gmode;
    initgraph(&gdriver, &gmode, "");
    
    if (graphresult() != grOk) {
        printf("Graphics initialization failed\n");
        return;
    }
    
    int max_x = getmaxx();
    int max_y = getmaxy();
    cleardevice();
    
    Window window = {150, 100, 500, 400};
    
    setcolor(YELLOW);
    outtextxy(max_x/2 - 150, 20, "Cohen-Sutherland Line Clipping");
    
    setcolor(GREEN);
    rectangle(window.xmin, window.ymin, window.xmax, window.ymax);
    
    setcolor(RED);
    Line lines[] = {{50,150,300,300}, {100,50,400,450}, {550,50,600,400}};
    int num_lines = 3;
    
    for (int i = 0; i < num_lines; i++)
        line(lines[i].x1, lines[i].y1, lines[i].x2, lines[i].y2);
    
    getch();
    closegraph();
}

/* ============================================
   TOWER OF HANOI
   ============================================ */
void hanoi(int n, char from, char to, char aux) {
    if (n == 0) return;
    hanoi(n - 1, from, aux, to);
    printf("Move disk %d from %c to %c\n", n, from, to);
    hanoi(n - 1, aux, to, from);
}

void tower_of_hanoi_demo() {
    int n;
    printf("Enter number of disks for Tower of Hanoi: ");
    if (scanf("%d", &n) != 1) return;
    printf("Solution:\n");
    hanoi(n, 'A', 'C', 'B');
}

/* ============================================
   POINT IN CLIPPING BOUNDARY
   ============================================ */
int point_inside_rect(int x, int y, int xmin, int ymin, int xmax, int ymax) {
    return (x >= xmin && x <= xmax && y >= ymin && y <= ymax);
}

void point_in_clip_demo() {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "");
    
    int xmin = 150, ymin = 100, xmax = 500, ymax = 400;
    rectangle(xmin, ymin, xmax, ymax);
    
    setcolor(WHITE);
    outtextxy(20, 20, "Enter point coordinates to test if inside rect");
    
    int x, y;
    printf("Enter point x y: ");
    if (scanf("%d %d", &x, &y) != 2) {
        closegraph();
        return;
    }
    
    if (point_inside_rect(x, y, xmin, ymin, xmax, ymax)) {
        putpixel(x, y, GREEN);
        printf("Point is inside\n");
    } else {
        putpixel(x, y, RED);
        printf("Point is outside\n");
    }
    
    getch();
    closegraph();
}

/* ============================================
   POINT ROTATION
   ============================================ */
void rotate_point(int xc, int yc, int x, int y, double angle_deg, int *rx, int *ry) {
    double angle = angle_deg * PI / 180.0;
    double s = sin(angle), c = cos(angle);
    double tx = x - xc;
    double ty = y - yc;
    double nx = tx * c - ty * s;
    double ny = tx * s + ty * c;
    *rx = (int)round(nx + xc);
    *ry = (int)round(ny + yc);
}

void point_rotation_demo() {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "");
    
    int xc, yc, x, y;
    double angle;
    printf("Enter center xc yc and point x y and angle (deg): ");
    if (scanf("%d %d %d %d %lf", &xc, &yc, &x, &y, &angle) != 5) {
        closegraph();
        return;
    }
    
    int rx, ry;
    rotate_point(xc, yc, x, y, angle, &rx, &ry);
    
    setcolor(WHITE);
    putpixel(x, y, RED);
    putpixel(rx, ry, GREEN);
    line(xc, yc, x, y);
    line(xc, yc, rx, ry);
    
    getch();
    closegraph();
}

/* ============================================
   POINT SCALING
   ============================================ */
void scale_point(int xc, int yc, int x, int y, double sx, double sy, int *sx_out, int *sy_out) {
    double tx = x - xc;
    double ty = y - yc;
    double nx = tx * sx;
    double ny = ty * sy;
    *sx_out = (int)round(nx + xc);
    *sy_out = (int)round(ny + yc);
}

void point_scaling_demo() {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "");
    
    int xc, yc, x, y;
    double sx, sy;
    printf("Enter center xc yc and point x y and scale sx sy: ");
    if (scanf("%d %d %d %d %lf %lf", &xc, &yc, &x, &y, &sx, &sy) != 6) {
        closegraph();
        return;
    }
    
    int rx, ry;
    scale_point(xc, yc, x, y, sx, sy, &rx, &ry);
    
    setcolor(WHITE);
    putpixel(x, y, RED);
    putpixel(rx, ry, GREEN);
    line(xc, yc, x, y);
    line(xc, yc, rx, ry);
    
    getch();
    closegraph();
}

/* ============================================
   POINT TRANSLATION
   ============================================ */
void translate_point(int x, int y, int tx, int ty, int *rx, int *ry) {
    *rx = x + tx;
    *ry = y + ty;
}

void point_translation_demo() {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "");
    
    int x, y, tx, ty;
    printf("Enter point x y and translation tx ty: ");
    if (scanf("%d %d %d %d", &x, &y, &tx, &ty) != 4) {
        closegraph();
        return;
    }
    
    int rx, ry;
    translate_point(x, y, tx, ty, &rx, &ry);
    
    putpixel(x, y, RED);
    putpixel(rx, ry, GREEN);
    line(x, y, rx, ry);
    
    getch();
    closegraph();
}

/* ============================================
   TRIANGLE ROTATION
   ============================================ */
void rotate_point_d(double xc, double yc, double x, double y, double angle_deg, double *rx, double *ry) {
    double angle = angle_deg * PI / 180.0;
    double s = sin(angle), c = cos(angle);
    double tx = x - xc, ty = y - yc;
    double nx = tx * c - ty * s, ny = tx * s + ty * c;
    *rx = nx + xc;
    *ry = ny + yc;
}

void triangle_rotation_demo() {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "");
    
    double x1, y1, x2, y2, x3, y3, xc, yc, angle;
    printf("Enter triangle x1 y1 x2 y2 x3 y3 and center xc yc and angle: ");
    if (scanf("%lf %lf %lf %lf %lf %lf %lf %lf %lf", &x1, &y1, &x2, &y2, &x3, &y3, &xc, &yc, &angle) != 9) {
        closegraph();
        return;
    }
    
    double rx1, ry1, rx2, ry2, rx3, ry3;
    rotate_point_d(xc, yc, x1, y1, angle, &rx1, &ry1);
    rotate_point_d(xc, yc, x2, y2, angle, &rx2, &ry2);
    rotate_point_d(xc, yc, x3, y3, angle, &rx3, &ry3);
    
    setcolor(WHITE);
    line((int)round(x1), (int)round(y1), (int)round(x2), (int)round(y2));
    line((int)round(x2), (int)round(y2), (int)round(x3), (int)round(y3));
    line((int)round(x3), (int)round(y3), (int)round(x1), (int)round(y1));
    
    setcolor(GREEN);
    line((int)round(rx1), (int)round(ry1), (int)round(rx2), (int)round(ry2));
    line((int)round(rx2), (int)round(ry2), (int)round(rx3), (int)round(ry3));
    line((int)round(rx3), (int)round(ry3), (int)round(rx1), (int)round(ry1));
    
    getch();
    closegraph();
}

/* ============================================
   TRIANGLE TRANSLATION
   ============================================ */
void translate_point_i(int x, int y, int tx, int ty, int *rx, int *ry) {
    *rx = x + tx;
    *ry = y + ty;
}

void triangle_translation_demo() {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "");
    
    int x1, y1, x2, y2, x3, y3, tx, ty;
    printf("Enter triangle x1 y1 x2 y2 x3 y3 and translation tx ty: ");
    if (scanf("%d %d %d %d %d %d %d %d", &x1, &y1, &x2, &y2, &x3, &y3, &tx, &ty) != 8) {
        closegraph();
        return;
    }
    
    int rx1, ry1, rx2, ry2, rx3, ry3;
    translate_point_i(x1, y1, tx, ty, &rx1, &ry1);
    translate_point_i(x2, y2, tx, ty, &rx2, &ry2);
    translate_point_i(x3, y3, tx, ty, &rx3, &ry3);
    
    setcolor(WHITE);
    line(x1, y1, x2, y2);
    line(x2, y2, x3, y3);
    line(x3, y3, x1, y1);
    
    setcolor(GREEN);
    line(rx1, ry1, rx2, ry2);
    line(rx2, ry2, rx3, ry3);
    line(rx3, ry3, rx1, ry1);
    
    getch();
    closegraph();
}

/* ============================================
   LINE INTERSECTION
   ============================================ */
int compute_intersection(double x1, double y1, double x2, double y2,
                         double x3, double y3, double x4, double y4,
                         double *ix, double *iy) {
    double a1 = y2 - y1;
    double b1 = x1 - x2;
    double c1 = a1 * x1 + b1 * y1;
    double a2 = y4 - y3;
    double b2 = x3 - x4;
    double c2 = a2 * x3 + b2 * y3;
    double det = a1 * b2 - a2 * b1;
    
    if (fabs(det) < 1e-9) return 0;
    
    *ix = (b2 * c1 - b1 * c2) / det;
    *iy = (a1 * c2 - a2 * c1) / det;
    return 1;
}

void line_intersection_demo() {
    int x1, y1, x2, y2, x3, y3, x4, y4;
    printf("Enter A(x1 y1) B(x2 y2) C(x3 y3) D(x4 y4): \n");
    if (scanf("%d %d %d %d %d %d %d %d", &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4) != 8)
        return;
    
    double ix, iy;
    int ok = compute_intersection(x1, y1, x2, y2, x3, y3, x4, y4, &ix, &iy);
    
    if (!ok) {
        printf("Lines are parallel or coincident\n");
        return;
    }
    
    printf("Intersection at: (%f, %f)\n", ix, iy);
    
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "");
    
    line(x1, y1, x2, y2);
    line(x3, y3, x4, y4);
    putpixel((int)round(ix), (int)round(iy), GREEN);
    
    getch();
    closegraph();
}

/* ============================================
   BUBBLE SHOT VISUALIZATION
   ============================================ */
typedef struct {
    int x, y, r, dy, color;
} Bubble;

void bubble_shot_demo() {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "");
    
    int maxx = getmaxx(), maxy = getmaxy();
    srand(time(NULL));
    
    const int N = 8;
    Bubble b[N];
    
    for (int i = 0; i < N; i++) {
        b[i].r = 20 + rand() % 20;
        b[i].x = 50 + i * 80;
        b[i].y = 50 + rand() % 100;
        b[i].dy = 2 + rand() % 4;
        b[i].color = rand() % 15 + 1;
    }
    
    while (!kbhit()) {
        cleardevice();
        
        for (int i = 0; i < N; i++) {
            setcolor(b[i].color);
            setfillstyle(SOLID_FILL, b[i].color);
            circle(b[i].x, b[i].y, b[i].r);
            floodfill(b[i].x, b[i].y, b[i].color);
            
            b[i].y += b[i].dy;
            
            if (b[i].y + b[i].r >= maxy - 20) {
                b[i].y = maxy - 20 - b[i].r;
                b[i].dy = -b[i].dy;
            }
            if (b[i].y - b[i].r <= 20) {
                b[i].y = 20 + b[i].r;
                b[i].dy = -b[i].dy;
            }
        }
        delay(50);
    }
    
    getch();
    closegraph();
}

/* ============================================
   BASIC GRAPHICS STRUCTURE
   ============================================ */
void basic_graphics_demo() {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "");
    
    if (graphresult() != grOk) {
        printf("Graphics initialization failed\n");
        return;
    }
    
    setbkcolor(BLACK);
    cleardevice();
    
    setcolor(WHITE);
    outtextxy(50, 50, "Basic graphics.h program structure");
    outtextxy(50, 70, "Press any key to exit...");
    
    getch();
    closegraph();
}

/* ============================================
   DRAWING PRIMITIVES
   ============================================ */
void drawing_primitives_demo() {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "");
    
    int maxx = getmaxx(), maxy = getmaxy();
    cleardevice();
    setbkcolor(BLACK);
    cleardevice();
    
    setcolor(WHITE);
    outtextxy(20, 20, "Drawing Primitives Demo");
    
    setcolor(RED);
    circle(150, 120, 50);
    
    setcolor(GREEN);
    rectangle(300, 80, 450, 160);
    
    setcolor(CYAN);
    line(50, 220, 500, 220);
    putpixel(60, 230, WHITE);
    
    setcolor(YELLOW);
    arc(150, 320, 0, 180, 50);
    
    setcolor(MAGENTA);
    bar(300, 260, 380, 320);
    
    setcolor(LIGHTBLUE);
    ellipse(520, 120, 0, 360, 60, 30);
    
    int poly[] = {520, 240, 580, 300, 460, 300};
    drawpoly(3, poly);
    
    setcolor(WHITE);
    outtextxy(20, maxy - 60, "Press any key to fill shapes and exit");
    
    getch();
    
    setfillstyle(SOLID_FILL, RED);
    floodfill(150, 120, RED);
    
    setfillstyle(SOLID_FILL, GREEN);
    floodfill(310, 90, GREEN);
    
    setfillstyle(SOLID_FILL, LIGHTBLUE);
    floodfill(520, 120, LIGHTBLUE);
    
    getch();
    closegraph();
}

/* ============================================
   2D TRANSFORMATIONS
   ============================================ */
typedef struct {
    float x, y;
} Point;

typedef struct {
    Point vertices[10];
    int num_vertices;
} Polygon;

void draw_polygon(Polygon poly, int color) {
    setcolor(color);
    for (int i = 0; i < poly.num_vertices; i++) {
        int x1 = poly.vertices[i].x;
        int y1 = poly.vertices[i].y;
        int x2 = poly.vertices[(i + 1) % poly.num_vertices].x;
        int y2 = poly.vertices[(i + 1) % poly.num_vertices].y;
        line(x1, y1, x2, y2);
    }
}

Polygon translate(Polygon poly, float tx, float ty) {
    Polygon translated = poly;
    for (int i = 0; i < poly.num_vertices; i++) {
        translated.vertices[i].x = poly.vertices[i].x + tx;
        translated.vertices[i].y = poly.vertices[i].y + ty;
    }
    return translated;
}

Polygon scale(Polygon poly, float sx, float sy, float cx, float cy) {
    Polygon scaled = poly;
    for (int i = 0; i < poly.num_vertices; i++) {
        scaled.vertices[i].x = cx + (poly.vertices[i].x - cx) * sx;
        scaled.vertices[i].y = cy + (poly.vertices[i].y - cy) * sy;
    }
    return scaled;
}

Polygon rotate(Polygon poly, float angle, float cx, float cy) {
    Polygon rotated = poly;
    float rad = angle * PI / 180.0;
    float cos_a = cos(rad);
    float sin_a = sin(rad);
    
    for (int i = 0; i < poly.num_vertices; i++) {
        float x = poly.vertices[i].x - cx;
        float y = poly.vertices[i].y - cy;
        rotated.vertices[i].x = cx + (x * cos_a - y * sin_a);
        rotated.vertices[i].y = cy + (x * sin_a + y * cos_a);
    }
    return rotated;
}

Polygon reflect_x(Polygon poly, float cy) {
    Polygon reflected = poly;
    for (int i = 0; i < poly.num_vertices; i++) {
        reflected.vertices[i].x = poly.vertices[i].x;
        reflected.vertices[i].y = 2 * cy - poly.vertices[i].y;
    }
    return reflected;
}

Polygon reflect_y(Polygon poly, float cx) {
    Polygon reflected = poly;
    for (int i = 0; i < poly.num_vertices; i++) {
        reflected.vertices[i].x = 2 * cx - poly.vertices[i].x;
        reflected.vertices[i].y = poly.vertices[i].y;
    }
    return reflected;
}

Polygon shear_x(Polygon poly, float shear_factor) {
    Polygon sheared = poly;
    for (int i = 0; i < poly.num_vertices; i++) {
        sheared.vertices[i].x = poly.vertices[i].x + shear_factor * poly.vertices[i].y;
        sheared.vertices[i].y = poly.vertices[i].y;
    }
    return sheared;
}

void transformations_2d_demo() {
    int gdriver = DETECT, gmode;
    initgraph(&gdriver, &gmode, "");
    
    if (graphresult() != grOk) {
        printf("Graphics initialization failed\n");
        return;
    }
    
    int max_x = getmaxx();
    int max_y = getmaxy();
    cleardevice();
    
    setcolor(YELLOW);
    outtextxy(max_x / 2 - 150, 10, "2D Geometric Transformations");
    
    Polygon square;
    square.num_vertices = 4;
    square.vertices[0].x = 50;
    square.vertices[0].y = 100;
    square.vertices[1].x = 100;
    square.vertices[1].y = 100;
    square.vertices[2].x = 100;
    square.vertices[2].y = 150;
    square.vertices[3].x = 50;
    square.vertices[3].y = 150;
    
    setcolor(LIGHTCYAN);
    outtextxy(20, 50, "1. TRANSLATION");
    draw_polygon(square, WHITE);
    
    Polygon translated = translate(square, 80, 0);
    draw_polygon(translated, GREEN);
    
    setcolor(YELLOW);
    outtextxy(20, max_y - 40, "Press any key to exit...");
    
    getch();
    closegraph();
}

/* ============================================
   MAIN MENU
   ============================================ */
void show_menu() {
    printf("\n=== Computer Graphics Programs ===\n");
    printf("1. Boiler Plate\n");
    printf("2. DDA Line Algorithm\n");
    printf("3. Bresenham Line Algorithm\n");
    printf("4. Direct Method Line Algorithm\n");
    printf("5. Incremental Line Algorithm\n");
    printf("6. Midpoint Circle Algorithm\n");
    printf("7. Bresenham Circle Algorithm\n");
    printf("8. Cohen-Sutherland Line Clipping\n");
    printf("9. Tower of Hanoi\n");
    printf("10. Point In Clipping Boundary\n");
    printf("11. Point Rotation\n");
    printf("12. Point Scaling\n");
    printf("13. Point Translation\n");
    printf("14. Triangle Rotation\n");
    printf("15. Triangle Translation\n");
    printf("16. Line Intersection\n");
    printf("17. Bubble Shot Visualization\n");
    printf("18. Basic Graphics Structure\n");
    printf("19. Drawing Primitives\n");
    printf("20. 2D Transformations\n");
    printf("0. Exit\n");
    printf("Enter your choice: ");
}

int main() {
    int choice;
    
    while (1) {
        show_menu();
        scanf("%d", &choice);
        
        switch (choice) {
            case 1:
                boilerplate_main();
                break;
            case 2:
                dda_line_demo();
                break;
            case 3:
                bresenham_line_demo();
                break;
            case 4:
                direct_line_demo();
                break;
            case 5:
                incremental_line_demo();
                break;
            case 6:
                midpoint_circle_demo();
                break;
            case 7:
                bresenham_circle_demo();
                break;
            case 8:
                cohen_sutherland_demo();
                break;
            case 9:
                tower_of_hanoi_demo();
                break;
            case 10:
                point_in_clip_demo();
                break;
            case 11:
                point_rotation_demo();
                break;
            case 12:
                point_scaling_demo();
                break;
            case 13:
                point_translation_demo();
                break;
            case 14:
                triangle_rotation_demo();
                break;
            case 15:
                triangle_translation_demo();
                break;
            case 16:
                line_intersection_demo();
                break;
            case 17:
                bubble_shot_demo();
                break;
            case 18:
                basic_graphics_demo();
                break;
            case 19:
                drawing_primitives_demo();
                break;
            case 20:
                transformations_2d_demo();
                break;
            case 0:
                printf("Goodbye!\n");
                return 0;
            default:
                printf("Invalid choice! Try again.\n");
        }
    }
    
    return 0;
}