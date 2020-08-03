/*
 * Graphical buttons draw code
 */
#include "buttonUtils/drawPrim.cpp"          // Draws a primitive

void drawClock(
      GLfloat     gx,
      GLfloat     gy,
      GLfloat     hour,
      GLfloat     minute,
      GLfloat     scale,
      GLfloat     w2h,
      GLfloat*    faceColor,
      GLfloat*    detailColor,
      drawCall*   clockButton
      );
#include "buttonUtils/drawClock.cpp"         // Draws the master switch (clock in center of display)

void drawArrow(
      GLfloat     gx, 
      GLfloat     gy,
      GLfloat     ao,
      GLfloat     scale,
      GLfloat     w2h,
      GLfloat*    faceColor,
      GLfloat*    extraColor,
      GLfloat*    detailColor,
      drawCall*   arrowButton
      );
#include "buttonUtils/drawArrow.cpp"         // Draws a generic arrow that can be oriented in different directions

#include "buttonUtils/drawBulbButtons.cpp"   // Draws the Color-setting bottons that encircle/straddle the master switch
#include "buttonUtils/drawGranChanger.cpp"   // Draws the Granularity Rocker on the color picker screen
#include "buttonUtils/drawHueRing.cpp"       // Draws the ring of colored dots on the color picker
#include "buttonUtils/drawColrTri.cpp"       // Draws the triangle of colored dots for the color picker

void drawConfirm(
      GLfloat     gx,
      GLfloat     gy,
      GLfloat     scale,
      GLfloat     w2h,
      GLfloat*    faceColor,
      GLfloat*    extraColor,
      GLfloat*    detailColor,
      drawCall*   confirmButton
      );
#include "buttonUtils/drawConfirm.cpp"       // Draws a checkmark button
//#include "buttonUtils/primDrawTest.cpp"      // used for testing primitive draw code

void drawMenu(
      GLfloat     gx, 
      GLfloat     gy,               // Menu Position
      GLfloat     scale,            // Menu Size
      GLfloat     direction,        // Direction, in degrees about the unit circle, the menu slides out to
      GLfloat     deployed,         // 0.0=closed, 1.0=completely open
      GLfloat     floatingIndex,    // index of the selected element, used for scroll bar
      GLfloat     scrollCursor,     // animation cursor for element motion during scrolling (-1.0 to 1.0)
      GLuint      numElements,      // number of elements
      GLuint      menuType,         // 0=carousel w/ rollover, 1=linear strip w/ terminals, 2=value slider w/ min/max
      GLuint      numListings,      // number of elements to display at once
      GLuint      selectedElement,  // Index of the current selected element
      GLboolean   drawIndex,        // whether or not to draw the index over the number of elements
      GLboolean   selectFromScroll, // whether or not elements are selected by scrolling to them
      GLfloat*    elementCoords,    // Relative coordinates of Menu elements
      GLfloat     w2h,              // width to height ratio
      GLfloat*    faceColor,        // Main color for the body of the menu
      GLfloat*    detailColor,      // scroll bar, arrow colors
      drawCall*   MenuIndex,        // drawCall object for drawing menu index
      drawCall*   MenuOpen,         // drawCall object for drawing the menu open
      drawCall*   MenuClosed        // drawCall object for drawing the menu closed
      );

#include "buttonUtils/drawMenu.cpp"

void drawImageSquare(
      GLfloat     gx, 
      GLfloat     gy,
      GLfloat     ao,
      GLfloat     scale,
      GLfloat     w2h,
      GLfloat*    color,
      drawCall*   image
      );
#include "buttonUtils/drawImageSquare.cpp"
