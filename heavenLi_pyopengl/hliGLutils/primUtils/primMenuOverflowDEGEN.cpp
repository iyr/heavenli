
/*
 * Defines a drop menu for 4 or more listings
 */
unsigned int defineMenuOverflow(
      float          direction,        // Direction, in degrees, the menu slides out to
      float          deployed,         // 0.0=closed, 1.0=completely open
      float          floatingIndex,    // index of the selected element, used for scroll bar
      float          scrollCursor,     // element animation cursor for scrolling: -1.0 to 1.0
      unsigned int   numElements,      // number of elements
      unsigned int   menuLayout,       // 0=carousel w/ rollover, 1=terminated linear strip
      unsigned int   circleSegments,   // number of polygon segments
      unsigned int   numListings,      // number of elements to display at once
      bool           drawIndex,        // whether or not to draw the index over the number of elements
      bool           selectFromScroll, // Whether or not elements are selected by scrolling to them
      float*         elementCoords,    // Relative coordinates of Menu elements
      float          w2h,              // width to height ratio
      float*         faceColor,        // Main color for the body of the menu
      float*         detailColor,      // scroll bar, 
      std::vector<float> &verts,       // Input Vector of x,y coordinates
      std::vector<float> &colrs        // Input Vector of r,g,b values
      ){

   float mx          = 0.0f,              // Origin of Menu x coord
         my          = 0.0f,                 // Origin of Menu y coord
         tmo         = 5.75f+(float)drawIndex,  // local coordinate offset
         arrowRad    = 0.05f*pow(deployed, 2.0f),  // arrow thickness
         endOffset   = 1.0f/(float)(numListings-1),   // distance from last element to end of menu
         diffElements = floor((float)numListings*0.5f),  // number of elements that straddle selected element
         elementSpacing = ((6.0f-endOffset)-(1.5f+endOffset))/(float)(numListings-1);  // distance between elements
   
   /*
    * Menu Body
    */
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
   
   /*
    * Element Diamonds
    */

   float* glCoords = new float[(numListings+1)*3];
   defineElementCoords(
      direction,
      deployed,
      floatingIndex,
      scrollCursor,
      numElements,
      menuLayout,
      numListings,
      selectFromScroll,
      glCoords,
      elementCoords
      );

   for (unsigned int i = 0; i < numListings+1; i++){
      defineEllipse(
            glCoords[i*3+0],
            glCoords[i*3+1],
            0.1f*glCoords[i*3+2],
            0.1f*glCoords[i*3+2],
            2,
            detailColor,
            verts,
            colrs
            );
   }
   
   /*
    * Shift Selection arrow based on menu type
    */
   if (menuLayout == 0)
      tmo = 3.75f;
   else 
   if (menuLayout == 1) {
      tmo = 3.75f;
      if (floatingIndex < diffElements) {
         // Resolve edge-case bug
         if (floor(floatingIndex) == floatingIndex)
            tmo = 3.75f+ceil(diffElements-0.0f-floatingIndex)*elementSpacing;
         else
            tmo = 3.75f+ceil(diffElements-1.0f-floatingIndex)*elementSpacing+elementSpacing*(1.0f-scrollCursor);

      } else if (floatingIndex > (float)numElements-1.0f-diffElements) {
         // Resolve edge-case bug
         if (floor(floatingIndex) == floatingIndex)
            tmo = 3.75f+ceil(numElements-diffElements-1.0f-floatingIndex)*elementSpacing;
         else
            tmo = 3.75f+ceil(numElements-diffElements-2.0f-floatingIndex)*elementSpacing+elementSpacing*(1.0f-scrollCursor);

      } else {
         tmo = 3.75f;
      }

      if (scrollCursor == 0.0f) {   // Resolve edge-case bug
         if (floatingIndex == diffElements) {
            tmo = 3.75f;
         }
      }
   }
   
   // Flip selection arrow, scrollbar based on direction of menu
   float mirror = 1.0f,
         theta0 = 270.0f,
         theta1 = 360.0f;
   if (  (  
            direction <= 135.0f
            &&
            direction >= -1.0f
         )
         ||
         (
            direction <= 361.0f
            &&
            direction >= 315.0f
         )
      ){
      mirror = 1.0f;
   } else {
      theta0 = 0.0f;
      theta1 = 90.0f;
      mirror = -1.0f;
   }

   /*
    * Scrollbar
    */

   // Scrollbar background
   defineArch(
         mx+6.25f*deployed,            // x-position
         mirror*(my-1.125f*deployed),  // y-position
         0.0f, 0.0f,                   // x,y inner radii
         theta0,                       // arch start in degrees
         theta1,                       // arch end in degrees
         0.125f,                       // arch outer radius
         circleSegments/5,
         faceColor,
         verts,
         colrs
         );

         theta0 -= 90.0f*mirror;
         theta1 -= 90.0f*mirror;

   defineArch(
         mx+1.25f*deployed,            // x-position
         mirror*(my-1.125f*deployed),  // y-position
         0.0f, 0.0f,                   // x,y inner radii
         theta0,                       // arch start in degrees
         theta1,                       // arch end in degrees
         0.125f,                       // arch outer radius
         circleSegments/5,
         faceColor,
         verts,
         colrs
         );

   float tmy1 = mirror*(my - 0.5f*deployed),  // y-position
         tmy2 = mirror*(my-1.125f*deployed),  // y-position
         tmyy = 0.0f;

   if (mirror < 0.0f) {
      tmyy = tmy1;
      tmy1 = tmy2;
      tmy2 = tmyy;
   }

   defineQuad2pt(
         (mx+1.25f*deployed-0.125f),   // x-position
         tmy1,                         // y-position
         (mx+6.25f*deployed+0.125f),   // x-position
         tmy2,                         // y-position
         faceColor,
         verts,
         colrs
         );

   tmy1 = mirror*(my-1.125f*deployed);  // y-position
   tmy2 = mirror*(my-1.250f*deployed);  // y-position
   if (mirror < 0.0f) {
      tmyy = tmy1;
      tmy1 = tmy2;
      tmy2 = tmyy;
   }

   defineQuad2pt(
         (mx+1.25f*deployed),          // x-position
         tmy1,                         // y-position
         (mx+6.25f*deployed),          // x-position
         tmy2,                         // y-position
         faceColor,
         verts,
         colrs
         );

   float diffSize, scrollbarOffset, normCursor = 0.0f;
   scrollbarOffset   = 2.5f*((float)numListings/(float)numElements);
   diffSize          = 2.5f - scrollbarOffset;

   // Animate scrollbar midpoint
   if (floatingIndex <= (float)(numElements-1)){
      normCursor     = rangeShift(floatingIndex, (float)(numElements-1), 0.0f, 3.75f-diffSize, 3.75f+diffSize);
   } else if (floatingIndex > (float)(numElements-1)) { // Handle transition from start/end for carousel menus
      normCursor     = rangeShift(scrollCursor, 0.0f, 1.0f, 3.75f-diffSize, 3.75f+diffSize);
   }    

   // Foreground (cursor) scrollbar
   definePill(
         constrain(mx+(normCursor+scrollbarOffset)*deployed, 1.25f, 6.25f),
         mirror*(my-1.125f*deployed),
         constrain(mx+(normCursor-scrollbarOffset)*deployed, 1.25f, 6.25f),
         mirror*(my-1.125f*deployed),
         0.05f*deployed,
         circleSegments/5,
         detailColor,
         verts,
         colrs
         );

   /*
    * Selection Arrow
    */
   definePill(
         mx+tmo*deployed,
         mirror*(my+1.00f*pow(deployed, 3.0f)),
         mx+tmo*deployed-0.20f*deployed,
         mirror*(my+1.15f*pow(deployed, 3.0f)),
         arrowRad*(float)selectFromScroll,
         circleSegments/5,
         detailColor,
         verts,
         colrs
         );
   definePill(
         mx+tmo*deployed,
         mirror*(my+1.00f*pow(deployed, 3.0f)),
         mx+tmo*deployed+0.20f*deployed,
         mirror*(my+1.15f*pow(deployed, 3.0f)),
         arrowRad*(float)selectFromScroll,
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
   
   /*
    * Distil Arrow
    */
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
   
   /*
    * Proximal Arrow
    */
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

   if (drawIndex) {
      float tmx = 6.75f*deployed+0.25f,   // text location, X
            tmy = -0.5f*deployed*mirror;  // text location, Y
      defineEllipse(
            tmx, tmy,
            0.45f, 0.45f,
            circleSegments/2,
            detailColor,
            verts,
            colrs
            );
   } else {
      defineEllipse(
            0.0f, 0.0f,
            0.0f, 0.0f,
            circleSegments/2,
            detailColor,
            verts,
            colrs
            );
   }

   return verts.size()/2;
}

unsigned int updateMenuOverflowGeometry(
      float          direction,        // Direction, in degrees, the menu slides out to
      float          deployed,         // 0.0=closed, 1.0=completely open
      float          floatingIndex,    // index of the selected element, used for scroll bar
      float          scrollCursor,     // element animation cursor for scrolling: -1.0 to 1.0
      unsigned int   numElements,      // number of elements
      unsigned int   menuLayout,       // 0=carousel w/ rollover, 1=terminated linear strip
      unsigned int   circleSegments,   // number of polygon segments
      unsigned int   numListings,      // number of elements to display at once
      bool           drawIndex,        // whether or not to draw the index over the number of elements
      bool           selectFromScroll, // whether or not elements are selected by scrolling to them
      float*         elementCoords,    // Relative coordinates of Menu elements
      float          w2h,              // width to height ratio
      unsigned int   index,            // Index of where to start writing to input array
      float*         verts             // Input Vector of x,y coordinates
      ){
   unsigned int subIndex = index;

   float mx          = 0.0f,              // Origin of Menu x coord
         my          = 0.0f,                 // Origin of Menu y coord
         tmo         = 5.75f+(float)drawIndex,  // local coordinate offset
         arrowRad    = 0.05f*pow(deployed, 2.0f),  // arrow thickness
         endOffset   = 1.0f/(float)(numListings-1),   // distance from last element to end of menu
         diffElements = floor((float)numListings*0.5f),  // number of elements that straddle selected element
         elementSpacing = (6.0f-1.5f-endOffset*2.0f)/(float)(numListings-1);  // distance between elements

   /*
    * Menu Body
    */
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
   
   /*
    * Element Diamonds
    */

   float* glCoords = new float[(numListings+1)*3];
   defineElementCoords(
      direction,
      deployed,
      floatingIndex,
      scrollCursor,
      numElements,
      menuLayout,
      numListings,
      selectFromScroll,
      glCoords,
      elementCoords
      );

   for (unsigned int i = 0; i < numListings+1; i++){
      subIndex = updateEllipseGeometry(
            glCoords[i*3+0],
            glCoords[i*3+1],
            0.1f*glCoords[i*3+2],
            0.1f*glCoords[i*3+2],
            2,
            subIndex,
            verts
            );
   }
   
   /*
    * Shift Selection arrow based on menu type
    */

   if (menuLayout == 0)
      tmo = 3.75f;
   else 
   if (menuLayout == 1) {
      tmo = 3.75f;
      if (floatingIndex < diffElements) {
         if (floor(floatingIndex) == floatingIndex)   // Resolve edge-case bug
            tmo = 3.75f+ceil(diffElements-floatingIndex)*elementSpacing;
         else
            tmo = 3.75f+elementSpacing*(1.0f-scrollCursor)+ceil(diffElements-1.000f-floatingIndex)*elementSpacing;
      } else if (floatingIndex > (float)numElements-1.0f-diffElements) {
         if (floor(floatingIndex) == floatingIndex)   // Resolve edge-case bug
            tmo = 3.75f+ceil(numElements-diffElements-1.0f-floatingIndex)*elementSpacing;
         else
            tmo = 3.75f+elementSpacing*(1.0f-scrollCursor)+ceil(numElements-diffElements-2.0f-floatingIndex)*elementSpacing;
      } else {
         tmo = 3.75f;
      }

      if (scrollCursor == 0.0f) {   // Resolve edge-case bug
         if (floatingIndex == diffElements) {
            tmo = 3.75f;
         }
      }
   }
   
   // Flip selection arrow, scrollbar based on direction of menu
   float mirror = 1.0f,
         theta0 = 270.0f,
         theta1 = 360.0f;
   if (  (  
            direction <= 135.0f
            &&
            direction >= -1.0f
         )
         ||
         (
            direction <= 361.0f
            &&
            direction >= 315.0f
         )
      ){
      mirror = 1.0f;
   } else {
      theta0 = 0.0f;
      theta1 = 90.0f;
      mirror = -1.0f;
   }

   /*
    * Scrollbar
    */

   // Scrollbar background

   subIndex = updateArchGeometry(
         mx+6.25f*deployed,            // x-position
         mirror*(my-1.125f*deployed),  // y-position
         0.0f, 0.0f,                   // x,y inner radii
         theta0,                       // arch start in degrees
         theta1,                       // arch end in degrees
         0.125f,                       // arch outer radius
         circleSegments/5,
         subIndex,
         verts
         );

         theta0 -= 90.0f*mirror;
         theta1 -= 90.0f*mirror;

   subIndex = updateArchGeometry(
         mx+1.25f*deployed,            // x-position
         mirror*(my-1.125f*deployed),  // y-position
         0.0f, 0.0f,                   // x,y inner radii
         theta0,                       // arch start in degrees
         theta1,                       // arch end in degrees
         0.125f,                       // arch outer radius
         circleSegments/5,
         subIndex,
         verts
         );
   
   float tmy1 = mirror*(my - 0.5f*deployed),  // y-position
         tmy2 = mirror*(my-1.125f*deployed),  // y-position
         tmyy = 0.0f;

   if (mirror < 0.0f) {
      tmyy = tmy1;
      tmy1 = tmy2;
      tmy2 = tmyy;
   }

   subIndex = updateQuad2ptGeometry(
         (mx+1.25f*deployed-0.125f),   // x-position
         tmy1,                         // y-position
         (mx+6.25f*deployed+0.125f),   // x-position
         tmy2,                         // y-position
         subIndex,
         verts
         );

   tmy1 = mirror*(my-1.125f*deployed);  // y-position
   tmy2 = mirror*(my-1.250f*deployed);  // y-position
   if (mirror < 0.0f) {
      tmyy = tmy1;
      tmy1 = tmy2;
      tmy2 = tmyy;
   }

   subIndex = updateQuad2ptGeometry(
         (mx+1.25f*deployed),          // x-position
         tmy1,                         // y-position
         (mx+6.25f*deployed),          // x-position
         tmy2,                         // y-position
         subIndex,
         verts
         );

   float diffSize, scrollbarOffset, normCursor = 0.0f;
   scrollbarOffset   = 2.5f*((float)numListings/(float)numElements);
   diffSize          = 2.5f - scrollbarOffset;

   // Animate scrollbar midpoint
   if (floatingIndex <= (float)(numElements-1)){
      normCursor     = rangeShift(floatingIndex, (float)(numElements-1), 0.0f, 3.75f-diffSize, 3.75f+diffSize);
   } else if (floatingIndex > (float)(numElements-1)) { // Handle transition from start/end for carousel menus
      normCursor     = rangeShift(scrollCursor, 0.0f, 1.0f, 3.75f-diffSize, 3.75f+diffSize);
   }    

   // Foreground (cursor) scrollbar
   subIndex = updatePillGeometry(
         constrain(mx+(normCursor+scrollbarOffset)*deployed, 1.25f, 6.25f),
         mirror*(my-1.125f*deployed),
         constrain(mx+(normCursor-scrollbarOffset)*deployed, 1.25f, 6.25f),
         mirror*(my-1.125f*deployed),
         0.05f*deployed,
         circleSegments/5,
         subIndex,
         verts
         );
   /*
    * Selection Arrow
    */
   subIndex = updatePillGeometry(
         mx+tmo*deployed,
         mirror*(my+1.00f*pow(deployed, 3.0f)),
         mx+tmo*deployed-0.20f*deployed,
         mirror*(my+1.15f*pow(deployed, 3.0f)),
         arrowRad*(float)selectFromScroll,
         circleSegments/5,
         subIndex,
         verts
         );

   subIndex = updatePillGeometry(
         mx+tmo*deployed,
         mirror*(my+1.00f*pow(deployed, 3.0f)),
         mx+tmo*deployed+0.20f*deployed,
         mirror*(my+1.15f*pow(deployed, 3.0f)),
         arrowRad*(float)selectFromScroll,
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
   
   /*
    * Distil Arrow
    */
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
   
   /*
    * Proximal Arrow
    */
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

   if (drawIndex) {
      float tmx = 6.75f*deployed+0.25f,   // text location, X
            tmy = -0.5f*deployed*mirror;  // text location, Y
      subIndex = updateEllipseGeometry(
            tmx, tmy,
            0.45f, 0.45f,
            circleSegments/2,
            subIndex,
            verts
            );
   } else {
      subIndex = updateEllipseGeometry(
            0.0f, 0.0f,
            0.0f, 0.0f,
            circleSegments/2,
            subIndex,
            verts
            );
   }

   return subIndex;
}

unsigned int updateMenuOverflowColors(
      bool           drawIndex,
      unsigned int   circleSegments,// number of polygon segments
      unsigned int   numListings,   // number of elements to display at once
      float*         faceColor,     // Main color for the body of the menu
      float*         detailColor,   // scroll bar, 
      unsigned int   index,         // Index of where to start writing to input array
      float*         colrs          // Input Vector of r,g,b values
      ){
   unsigned int subIndex = index;
   
   /*
    * Menu Body
    */
   subIndex = updatePillColor(
         circleSegments,
         faceColor,
         subIndex,
         colrs
         );
   
   /*
    * Element Diamonds
    */
   for (unsigned int i = 0; i < numListings+1; i++) {
      subIndex = updateEllipseColor(
            2,
            detailColor,
            subIndex,
            colrs
            );
   }
   
   /*
    * Scrollbar
    */

   // Background Scrollbar
   subIndex = updateArchColor(
         circleSegments/5,
         faceColor,
         subIndex,
         colrs
         );

   subIndex = updateArchColor(
         circleSegments/5,
         faceColor,
         subIndex,
         colrs
         );

   subIndex = updateQuadColor(
         faceColor,
         subIndex,
         colrs
         );

   subIndex = updateQuadColor(
         faceColor,
         subIndex,
         colrs
         );

   // Foreground (cursor) scrollbar
   subIndex = updatePillColor(
         circleSegments/5,
         detailColor,
         subIndex,
         colrs
         );
   /*
    * Selection Arrow
    */
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
   
   /*
    * Distil Arrow
    */
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
   
   /*
    * Proximal Arrow
    */
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
   if (drawIndex) {
      subIndex = updateEllipseColor(
            circleSegments/2,
            detailColor,
            subIndex,
            colrs
            );
   } else {
      subIndex = updateEllipseColor(
            circleSegments/2,
            detailColor,
            subIndex,
            colrs
            );
   }
   return subIndex;
}
