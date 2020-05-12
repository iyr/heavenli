/*
 * Defines a drop menu for 4 or more listings
 */

unsigned int drawMenuOverflow(
      float          direction,     // Direction, in degrees, the menu slides out to
      float          deployed,      // 0.0=closed, 1.0=completely open
      float          floatingIndex, // index of the selected element, used for scroll bar
      float          scrollCursor,  // element animation cursor for scrolling: -1.0 to 1.0
      unsigned int   numElements,   // number of elements
      unsigned int   menuLayout,      // 0=carousel w/ rollover, 1=terminated linear strip
      unsigned int   circleSegments,// number of polygon segments
      bool           drawIndex,     // whether or not to draw the index over the number of elements
      float*         elementCoords, // Relative coordinates of Menu elements
      float          w2h,           // width to height ratio
      float*         faceColor,     // Main color for the body of the menu
      float*         detailColor,   // scroll bar, 
      std::vector<float> &verts,       // Input Vector of x,y coordinates
      std::vector<float> &colrs        // Input Vector of r,g,b values
      ){
   unsigned int numListings = 3;
   float mx=0.0f, // Origin of Menu
         my=0.0f, // Origin of Menu
         tmo;     // local coordinate offset
   float arrowRad = 0.05f*pow(deployed, 2.0f);

   tmo = 5.75f+(float)drawIndex;

   // Menu Body
   definePill(
         mx, 
         my,
         mx + tmo*deployed,
         my,
         1.0,
         circleSegments,
         faceColor,
         verts,
         colrs
         );

   // Element Diamonds
   if (  (menuLayout == 1 and floatingIndex < 1.0f)
         or 
         (menuLayout == 1 and floatingIndex >= numElements-2.0f)
         or
         abs(scrollCursor) == 0.0f
         ){
      float tms = 1.0f;
      float tmr;
      for (unsigned int i = 0; i < numListings+1; i++) {
         tmr = (2.0f + (float)i*1.75f + (float)drawIndex)*deployed;
         if (i == numListings)
            tms = 0.0f;
         defineEllipse(
               mx + (2.0f + (float)i*1.75f + (float)drawIndex)*deployed,
               my,
               0.1f*tms,
               0.1f*tms,
               2,
               detailColor,
               verts,
               colrs
               );
         elementCoords[i*3+0] = mx+tmr*cos(degToRad(direction));
         elementCoords[i*3+1] = my+tmr*sin(degToRad(direction));
         elementCoords[i*3+2] = tms*deployed;
      }
   } else {
      float tms = 1.0f;
      float tmr, tma;
      for (unsigned int i = 0; i < numListings+1; i++) {

         if (i == 0) {
            tma = -3.0f*pow(scrollCursor, 2.0f) + 4.0f*scrollCursor;
            tms = scrollCursor;
            tmr = (2.0f + ((float)i - tma + 1.0f)*1.75f + (float)drawIndex)*deployed;
         } else if (i == numListings) {
            tms -= abs(scrollCursor);
            tma = -3.0f*pow(1.0f-scrollCursor, 2.0f) + 4.0f*(1.0f-scrollCursor);
            tmr = (2.0f + ((float)i + tma - 2.0f)*1.75f + (float)drawIndex)*deployed;
         } else {
            tms = 1.0f;
            tmr = (2.0f + ((float)i + scrollCursor - 1.0f)*1.75f + (float)drawIndex)*deployed;
         }
         defineEllipse(
               mx+tmr,
               my,
               0.1f*tms,
               0.1f*tms,
               2,
               detailColor,
               verts,
               colrs
               );
         elementCoords[i*3+0] = mx+tmr*cos(degToRad(direction));
         elementCoords[i*3+1] = my+tmr*sin(degToRad(direction));
         elementCoords[i*3+2] = tms*deployed;
      }
   }

   // Shift Selection arrow based on menu type
   if (menuLayout == 0)
      tmo = 3.75f;
   else 
   if (menuLayout == 1) {
      tmo = 3.75f;
      if (floatingIndex < 1.0f) {
         tmo = 5.5f-1.75f*scrollCursor;
      } else if (floatingIndex > (float)numElements-2.0f) {
         tmo = 3.75f-1.75f*scrollCursor;
      } else {
         tmo = 3.75f;
      }

      if (scrollCursor == 0.0f) {
         if (floatingIndex == 1.0f) {
            tmo = 3.75f;
         } else if (floatingIndex == (float)numElements-1.0f) {
            tmo = 2.0f;
         }
      }
   }

   // Selection Arrow
   definePill(
         mx+tmo*deployed,
         my+0.85f*pow(deployed, 3.0f),
         mx+tmo*deployed-0.20f*deployed,
         my+1.00f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         detailColor,
         verts,
         colrs
         );
   definePill(
         mx+tmo*deployed,
         my+0.85f*pow(deployed, 3.0f),
         mx+tmo*deployed+0.20f*deployed,
         my+1.00f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         detailColor,
         verts,
         colrs
         );

   tmo = 6.5f;
   float tmt = 0.0f;
   if (menuLayout == 0)
      tmt = 0.25f*deployed;
   else 
   if (menuLayout == 1) {  // Flatten arrow if at end of linear strip
      if (floatingIndex < 1.0f) {
         tmt += 0.25f*scrollCursor;
      } else {
         tmt = 0.25f*deployed;
      }

      if (scrollCursor == 0.0f) {
         if (floatingIndex == 1.0f) {
            tmt = 0.25f*deployed;
         }
      }
   }

   // Distil Arrow
   definePill(
         mx+tmo*deployed,
         my,
         mx+tmo*deployed-tmt,
         my+0.75f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         detailColor,
         verts,
         colrs
         );
   definePill(
         mx+tmo*deployed,
         my,
         mx+tmo*deployed-tmt,
         my-0.75f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         detailColor,
         verts,
         colrs
         );

   tmo = 1.0f + arrowRad;

   if (menuLayout == 0)
      tmt = 0.25f*deployed;
   else 
   if (menuLayout == 1) {  // Flatten arrow if at end of linear strip
      if (floatingIndex > (float)numElements-2.0f) {
         tmt -= 0.25f*scrollCursor;
      } else {
         tmt = 0.25f*deployed;
      }

      if (scrollCursor == 0.0f) {
         if (floatingIndex == (float)numElements-1.0f) {
            tmt -= 0.25f*deployed;
         }
      }
   }

   // Proximal Arrow
   definePill(
         mx+tmo*deployed,
         my,
         mx+tmo*deployed+tmt,
         my+0.75f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         detailColor,
         verts,
         colrs
         );
   definePill(
         mx+tmo*deployed,
         my,
         mx+tmo*deployed+tmt,
         my-0.75f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         detailColor,
         verts,
         colrs
         );

   return verts.size()/2;
}

unsigned int drawMenuOverflow(
      float          direction,     // Direction, in degrees, the menu slides out to
      float          deployed,      // 0.0=closed, 1.0=completely open
      float          floatingIndex, // index of the selected element, used for scroll bar
      float          scrollCursor,  // element animation cursor for scrolling: -1.0 to 1.0
      unsigned int   numElements,   // number of elements
      unsigned int   menuLayout,      // 0=carousel w/ rollover, 1=terminated linear strip
      unsigned int   circleSegments,// number of polygon segments
      bool           drawIndex,     // whether or not to draw the index over the number of elements
      float*         elementCoords, // Relative coordinates of Menu elements
      float          w2h,           // width to height ratio
      unsigned int   index,
      float*         verts          // Input Vector of x,y coordinates
      ){
   unsigned int numListings = 3;
   unsigned int subIndex = index;

   float mx=0.0f,  // Origin of Menu
           my=0.0f,  // Origin of Menu
           tmo;     // local coordinate offset

   float arrowRad = 0.05f*pow(deployed, 2.0f);
   tmo = 5.75f+(float)drawIndex;

   // Menu Body
   subIndex = updatePillGeometry(
         mx, 
         my,
         mx + tmo*deployed,
         my,
         1.0,
         circleSegments,
         subIndex,
         verts
         );

   // Element Diamonds
   if (  (menuLayout == 1 and floatingIndex < 1.0f)
         or 
         (menuLayout == 1 and floatingIndex >= numElements-2.0f)
         or
         abs(scrollCursor) == 0.0f
         ){
      float tms = 1.0f;
      float tmr;
      for (unsigned int i = 0; i < numListings+1; i++) {
         tmr = (2.0f + (float)i*1.75f + (float)drawIndex)*deployed;
         if (i == numListings)
            tms = 0.0f;
         subIndex = updateEllipseGeometry(
               mx + (2.0f + (float)i*1.75f + (float)drawIndex)*deployed,
               my,
               0.1f*tms,
               0.1f*tms,
               2,
               subIndex,
               verts
               );
         elementCoords[i*3+0] = mx+tmr*cos(degToRad(direction));
         elementCoords[i*3+1] = my+tmr*sin(degToRad(direction));
         elementCoords[i*3+2] = tms*deployed;
      }
   } else {
      float tms = 1.0f;
      float tmr, tma;
      for (unsigned int i = 0; i < numListings+1; i++) {
         if (i == 0) {
            tma = -3.0f*pow(scrollCursor, 2.0f) + 4.0f*scrollCursor;
            tms = scrollCursor;
            tmr = (2.0f + ((float)i - tma + 1.0f)*1.75f + (float)drawIndex)*deployed;
         } else if (i == numListings) {
            tms -= abs(scrollCursor);
            tma = -3.0f*pow(1.0f-scrollCursor, 2.0f) + 4.0f*(1.0f-scrollCursor);
            tmr = (2.0f + ((float)i + tma - 2.0f)*1.75f + (float)drawIndex)*deployed;
         } else {
            tms = 1.0f;
            tmr = (2.0f + ((float)i+scrollCursor-1.0f)*1.75f + (float)drawIndex)*deployed;
         }
         subIndex = updateEllipseGeometry(
               mx+tmr,
               my,
               0.1f*tms,
               0.1f*tms,
               2,
               subIndex,
               verts
               );
         elementCoords[i*3+0] = mx+tmr*cos(degToRad(direction));
         elementCoords[i*3+1] = my+tmr*sin(degToRad(direction));
         elementCoords[i*3+2] = tms*deployed;
      }
   }

   // Shift Selection arrow based on menu type
   if (menuLayout == 0)
      tmo = 3.75f;
   else 
   if (menuLayout == 1) {
      tmo = 3.75f;
      if (floatingIndex < 1.0f) {
         tmo = 5.5f-1.75f*scrollCursor;
      } else if (floatingIndex > (float)numElements-2.0f) {
         tmo = 3.75f-1.75f*scrollCursor;
      } else {
         tmo = 3.75f;
      }

      if (scrollCursor == 0.0f) {
         if (floatingIndex == 1.0f) {
            tmo = 3.75f;
         } else if (floatingIndex == (float)numElements-1.0f) {
            tmo = 2.0f;
         }
      }
   }

   // Selection Arrow
   subIndex = updatePillGeometry(
         mx+tmo*deployed,
         my+0.85f*pow(deployed, 3.0f),
         mx+tmo*deployed-0.20f*deployed,
         my+1.00f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         subIndex,
         verts
         );

   subIndex = updatePillGeometry(
         mx+tmo*deployed,
         my+0.85f*pow(deployed, 3.0f),
         mx+tmo*deployed+0.20f*deployed,
         my+1.00f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         subIndex,
         verts
         );

   tmo = 6.5f;
   
   float tmt = 0.0f;
   if (menuLayout == 0)
      tmt = 0.25f*deployed;
   else 
   if (menuLayout == 1) {  // Flatten arrow if at end of linear strip
      if (floatingIndex < 1.0f) {
         tmt += 0.25f*scrollCursor;
      } else {
         tmt = 0.25f*deployed;
      }

      if (scrollCursor == 0.0f) {
         if (floatingIndex == 1.0f) {
            tmt = 0.25f*deployed;
         }
      }
   }

   // Distil Arrow
   subIndex = updatePillGeometry(
         mx+tmo*deployed,
         my,
         mx+tmo*deployed-tmt,
         my+0.75f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         subIndex,
         verts
         );
   subIndex = updatePillGeometry(
         mx+tmo*deployed,
         my,
         mx+tmo*deployed-tmt,
         my-0.75f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         subIndex,
         verts
         );

   tmo = 1.0f + arrowRad;
   if (menuLayout == 0)
      tmt = 0.25f*deployed;
   else 
   if (menuLayout == 1) {  // Flatten arrow if at end of linear strip
      if (floatingIndex > (float)numElements-2.0f) {
         tmt -= 0.25f*scrollCursor;
      } else {
         tmt = 0.25f*deployed;
      }

      if (scrollCursor == 0.0f) {
         if (floatingIndex == (float)numElements-1.0f) {
            tmt -= 0.25f*deployed;
         }
      }
   }

   // Proximal Arrow
   subIndex = updatePillGeometry(
         mx+tmo*deployed,
         my,
         mx+tmo*deployed+tmt,
         my+0.75f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         subIndex,
         verts
         );
   subIndex = updatePillGeometry(
         mx+tmo*deployed,
         my,
         mx+tmo*deployed+tmt,
         my-0.75f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         subIndex,
         verts
         );

   return subIndex;
}

unsigned int drawMenuOverflow(
      unsigned int   circleSegments,// number of polygon segments
      float*         faceColor,     // Main color for the body of the menu
      float*         detailColor,   // scroll bar, 
      unsigned int   index,
      float*         colrs        // Input Vector of r,g,b values
      ){
   unsigned int subIndex = index;
   unsigned int numListings = 3;

   // Menu Body
   subIndex = updatePillColor(
         circleSegments,
         faceColor,
         subIndex,
         colrs
         );

   // Element Diamonds
   for (unsigned int i = 0; i < numListings+1; i++) {
      subIndex = updateEllipseColor(
            2,
            detailColor,
            subIndex,
            colrs
            );
   }

   // Selection Arrow
   subIndex = updatePillColor(
         circleSegments/5,
         detailColor,
         subIndex,
         colrs
         );

   subIndex = updatePillColor(
         circleSegments/5,
         detailColor,
         subIndex,
         colrs
         );

   // Distil Arrow
   subIndex = updatePillColor(
         circleSegments/5,
         detailColor,
         subIndex,
         colrs
         );
   subIndex = updatePillColor(
         circleSegments/5,
         detailColor,
         subIndex,
         colrs
         );

   // Proximal Arrow
   subIndex = updatePillColor(
         circleSegments/5,
         detailColor,
         subIndex,
         colrs
         );
   subIndex = updatePillColor(
         circleSegments/5,
         detailColor,
         subIndex,
         colrs
         );

   return subIndex;
}
