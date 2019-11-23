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
#include "buttonUtils/drawConfirm.cpp"       // Draws a checkmark button
//#include "buttonUtils/primDrawTest.cpp"      // used for testing primitive draw code

