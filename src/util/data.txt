## command to compile:

`gcc -o a.exe <filename>.c -L"C:/mingw/lib" -lbgi -lgdi32 -lcomdlg32 -luuid -loleaut32 -lole32 -lstdc++`

## run:

`./a.exe`

## List Of Programs:

List of program for practical of graphics:

1. Line drawing algo DDA + bresenham.
2. Direct method line algo
3. Incremental line algo
4. Circle drawing algos (midpoint and bresenham)
5. Clipping (Cohen-Sutherland) transformation
6. Tower of Hanoi
7. Bubble shot visualisation
8. Function to check if a point is inside or outside clipping boundary
9. Rotation
10. Scaling
11. Translation
12. Triangle rotate
13. Triangle translate
14. Function to calculate intersection of two lines (A-B and C-D)
15. Saving, loading and printing images
16. Basic `graphics.h` program structure
17. Drawing primitives with `graphics.h` such as `circle()`, `rectangle()`, `line()`, `putpixel()`, `arc()`, `bar()`, `ellipse()`, `drawpolygon()`, `outtextxy()`, `setfillstyle()`, `floodfill()` and `setcolor()`, `setbkcolor()`.

## Note:

- make sure to include the '#include"graphics.h"' in every c program.
- add the #include<conio.h> for the getch() method.
- In case, any function which is used in the program is not declared in the custom graphics.h then we can declare it there.
- the two files: 1. winbgim.h is putted in MinGW/include/ folder and 2. libbgi.a is putted in MinGW/lib/ folder and third the custom grahics.h file is created here to get the filed compiled and executed.
