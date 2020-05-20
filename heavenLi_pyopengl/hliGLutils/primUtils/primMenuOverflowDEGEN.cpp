/*
 * Defines a drop menu for 4 or more listings
 */

unsigned int drawMenuOverflow(
      float          direction,     // Direction, in degrees, the menu slides out to
      float          deployed,      // 0.0=closed, 1.0=completely open
      float          floatingIndex, // index of the selected element, used for scroll bar
      float          scrollCursor,  // element animation cursor for scrolling: -1.0 to 1.0
      unsigned int   numElements,   // number of elements
      unsigned int   menuLayout,    // 0=carousel w/ rollover, 1=terminated linear strip
      unsigned int   circleSegments,// number of polygon segments
      unsigned int   numListings,   // number of elements to display at once
      bool           drawIndex,     // whether or not to draw the index over the number of elements
      float*         elementCoords, // Relative coordinates of Menu elements
      float          w2h,           // width to height ratio
      float*         faceColor,     // Main color for the body of the menu
      float*         detailColor,   // scroll bar, 
      std::vector<float> &verts,    // Input Vector of x,y coordinates
      std::vector<float> &colrs     // Input Vector of r,g,b values
      ){

   float mx          = 0.0f,              // Origin of Menu x coord
         my          = 0.0f,                 // Origin of Menu y coord
         tmo,                                   // local coordinate offset
         arrowRad    = 0.05f*pow(deployed, 2.0f),  // arrow thickness
         endOffset   = 1.0f/(float)(numListings-1),   // distance from last element to end of menu
         elementSpacing = ((6.0f-endOffset)-(1.5f+endOffset))/(float)(numListings-1);  // distance between elements

   tmo = 5.75f+(float)drawIndex;
   
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
   if (  (menuLayout == 1 and floatingIndex < 1.0f)
         or 
         (menuLayout == 1 and floatingIndex >= numElements-2.0f)
         or
         abs(scrollCursor) == 0.0f
         ){    // Avoid extra ops when not animating by drawing diamonds statically
      float tms = 1.0f;
      float tmr;
      for (unsigned int i = 0; i < numListings+1; i++) {
         tmr = ((1.5f+endOffset) + (float)i*elementSpacing + (float)drawIndex)*deployed;
         if (i == numListings)
            tms = 0.0f;
         defineEllipse(
               mx + ((1.5f+endOffset) + (float)i*elementSpacing + (float)drawIndex)*deployed,
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
   } else {    // Draw animated element diamonds
      float tms = 1.0f;
      float tmr, tma;
      for (unsigned int i = 0; i < numListings+1; i++) {

         if (i == 0) {  // Special animation curve for end-case listings
            tms = scrollCursor;
            tma = -3.0f*pow(scrollCursor, 2.0f) + 4.0f*scrollCursor;
            tmr = ((1.5f+endOffset) + ((float)i - tma + 1.0f)*elementSpacing + (float)drawIndex)*deployed;
         } else if (i == numListings) {   // Special animation curve for end-case listings
            tms -= abs(scrollCursor);
            tma = -3.0f*pow(1.0f-scrollCursor, 2.0f) + 4.0f*(1.0f-scrollCursor);
            tmr = ((1.5f+endOffset) + ((float)i + tma - 2.0f)*elementSpacing + (float)drawIndex)*deployed;
         } else {
            tms = 1.0f;
            tmr = ((1.5f+endOffset) + ((float)i + scrollCursor - 1.0f)*elementSpacing + (float)drawIndex)*deployed;
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
   
   /*
    * Shift Selection arrow based on menu type
    */
   if (menuLayout == 0)
      tmo = 3.75f;
   else 
   if (menuLayout == 1) {
      tmo = 3.75f;
      if (floatingIndex < 1.0f) {
         tmo = 5.5f-elementSpacing*scrollCursor;
      } else if (floatingIndex > (float)numElements-2.0f) {
         tmo = 3.75f-elementSpacing*scrollCursor;
      } else {
         tmo = 3.75f;
      }

      if (scrollCursor == 0.0f) {
         if (floatingIndex == 1.0f) {
            tmo = 3.75f;
         } else if (floatingIndex == (float)numElements-1.0f) {
            tmo = (1.5f+endOffset);
         }
      }
   }
   
   // Flip selection arrow, scrollbar based on direction of menu
   float mirror = 1.0f;
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
         270.0f,                       // arch start in degrees
         360.0f,                       // arch end in degrees
         0.125f,                       // arch outer radius
         circleSegments/5,
         faceColor,
         verts,
         colrs
         );

   defineArch(
         mx+1.25f*deployed,            // x-position
         mirror*(my-1.125f*deployed),  // y-position
         0.0f, 0.0f,                   // x,y inner radii
         180.0f,                       // arch start in degrees
         270.0f,                       // arch end in degrees
         0.125f,                       // arch outer radius
         circleSegments/5,
         faceColor,
         verts,
         colrs
         );

   defineQuad2pt(
         mx+1.25f*deployed-0.125f,     // x-position
         mirror*(my-1.125f*deployed),  // y-position
         mx+6.25f*deployed+0.125f,     // x-position
         mirror*(my - 0.5f*deployed),  // y-position
         faceColor,
         verts,
         colrs
         );

   defineQuad2pt(
         mx+1.25f*deployed,            // x-position
         mirror*(my-1.125f*deployed),  // y-position
         mx+6.25f*deployed,            // x-position
         mirror*(my-1.250f*deployed),  // y-position
         faceColor,
         verts,
         colrs
         );

   float diffSize, scrollbarOffset, normCursor;
   scrollbarOffset   = 2.5f*((float)numListings/(float)numElements);
   diffSize          = 2.5f - scrollbarOffset;

   // Animate scrollbar midpoint
   if (floatingIndex <= (float)(numElements-1)){
      normCursor     = rangeShift(floatingIndex, (float)(numElements-1), 0.0f, 3.75f-diffSize, 3.75f+diffSize);
   } else if (floatingIndex > (float)(numElements-1)) { // Handle transition from start/end for carousel menus
      normCursor     = rangeShift(scrollCursor, 0.0f, 1.0f, 3.75f-diffSize, 3.75f+diffSize);
   } else {
      normCursor     = 0.0f;
   }

   // Foreground (cursor) scrollbar
   definePill(
         constrain(mx+(normCursor+scrollbarOffset)*deployed, 1.25f, 6.25f),
         mirror*(my-1.125f*deployed),
         constrain(mx+(normCursor-scrollbarOffset)*deployed, 1.25f, 6.25f),
         mirror*(my-1.125f*deployed),
         0.05f,
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
         mirror*(my+0.85f*pow(deployed, 3.0f)),
         mx+tmo*deployed-0.20f*deployed,
         mirror*(my+1.00f*pow(deployed, 3.0f)),
         arrowRad,
         circleSegments/5,
         detailColor,
         verts,
         colrs
         );
   definePill(
         mx+tmo*deployed,
         mirror*(my+0.85f*pow(deployed, 3.0f)),
         mx+tmo*deployed+0.20f*deployed,
         mirror*(my+1.00f*pow(deployed, 3.0f)),
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

   return verts.size()/2;
}

unsigned int drawMenuOverflow(
      float          direction,     // Direction, in degrees, the menu slides out to
      float          deployed,      // 0.0=closed, 1.0=completely open
      float          floatingIndex, // index of the selected element, used for scroll bar
      float          scrollCursor,  // element animation cursor for scrolling: -1.0 to 1.0
      unsigned int   numElements,   // number of elements
      unsigned int   menuLayout,    // 0=carousel w/ rollover, 1=terminated linear strip
      unsigned int   circleSegments,// number of polygon segments
      unsigned int   numListings,   // number of elements to display at once
      bool           drawIndex,     // whether or not to draw the index over the number of elements
      float*         elementCoords, // Relative coordinates of Menu elements
      float          w2h,           // width to height ratio
      unsigned int   index,         // Index of where to start writing to input array
      float*         verts          // Input Vector of x,y coordinates
      ){
   unsigned int subIndex = index;

   float mx          = 0.0f,              // Origin of Menu x coord
         my          = 0.0f,                 // Origin of Menu y coord
         tmo,                                   // local coordinate offset
         arrowRad    = 0.05f*pow(deployed, 2.0f),  // arrow thickness
         endOffset   = 1.0f/(float)(numListings-1),   // distance from last element to end of menu
         diffElements = floor((float)numListings*0.5f),  // number of elements that straddle selected element
         elementSpacing = (6.0f-1.5f-endOffset*2.0f)/(float)(numListings-1);  // distance between elements
         //elementSpacing = ((6.0f-endOffset)-(1.5f+endOffset))/(float)(numListings-1);  // distance between elements

   tmo = 5.75f+(float)drawIndex;
   
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
   if (  (menuLayout == 1 and floatingIndex < diffElements)
         or 
         (menuLayout == 1 and floatingIndex >= (float)(numElements-1)-diffElements)
         or
         abs(scrollCursor) == 0.0f
         ){    // Avoid extra ops when not animating by drawing diamonds statically
      float tms = 1.0f;
      float tmr, tmb=0.0f;
      for (unsigned int i = 0; i < numListings+1; i++) {
         tms = 1.0f;

         tmb = 0.0f;
         tmr = ((1.5f+endOffset) + (float)i*elementSpacing + (float)drawIndex)*deployed;
         if (i == numListings) {
            tms = 0.0f;
         } 

         if (menuLayout == 0) {
            if (i == (unsigned int)round((float)numListings*0.5f)){
               tmr = ((1.5f+endOffset) + (float)i*elementSpacing + (float)drawIndex)*deployed;
               tms = 1.0f;
               tmb = 0.0f;
            } 
            if (i+1 == (unsigned int)round((float)numListings*0.5f)){
               tmr = ((1.5f+endOffset) + (float)i*elementSpacing + (float)drawIndex)*deployed;
               tms = 1.25f;
               tmb = 0.1f;
            }
         }

         if (menuLayout == 1) {
            // Handle selection of elements in middle
            if (  (
                  floatingIndex >= diffElements
                  and
                  floatingIndex <= (float)(numElements-1)-diffElements
                  )
                  ){
               if (i == (unsigned int)round((float)numListings*0.5f)){
                  tms = 1.0f;
                  tmb = 0.0f;
               } 
               if (i+1 == (unsigned int)round((float)numListings*0.5f)){
                  tms = 1.25f;
                  tmb = 0.1f;
               }
            }

            // Handle selection of elements on ends
            if (  (
                  floatingIndex < diffElements
                  or
                  floatingIndex >= (float)(numElements-1)-diffElements
                  )
               ) {

               // Handle selection of elements start of list
               if (i == numListings-1-(unsigned int)floor(floatingIndex)) {
                  if (floatingIndex == floor(floatingIndex)) { // Resolve edge-case bug
                     tms = 1.25f;
                     tmb = 0.1f;
                  } else {
                     tms = 1.0f+0.25f*(1.0f-scrollCursor);
                     tmb = 0.1f*(1.0f-scrollCursor);
                  }
               }
               if (i+1 == numListings-1-(unsigned int)floor(floatingIndex)) {
                  if (floatingIndex == floor(floatingIndex)) { // Resolve edge-case bug
                     tms = 1.0f;
                     tmb = 0.0f;
                  } else {
                     tms = 1.0f+0.25f*(scrollCursor);
                     tmb = 0.1f*(scrollCursor);
                  }
               }

               // Handle selection of elements at end of list
               if (i == numElements-1-(unsigned int)floor(floatingIndex)) {
                  if (floatingIndex == floor(floatingIndex)) { // Resolve edge-case bug
                     tms = 1.25f;
                     tmb = 0.1f;
                  } else {
                     tms = 1.0f+0.25f*(1.0f-scrollCursor);
                     tmb = 0.1f*(1.0f-scrollCursor);
                  }
               }
               if (i+1 == numElements-1-(unsigned int)floor(floatingIndex)) {
                  if (floatingIndex == floor(floatingIndex)) { // Resolve edge-case bug
                     tms = 1.0f;
                     tmb = 0.0f;
                  } else {
                     tms = 1.0f+0.25f*(scrollCursor);
                     tmb = 0.1f*(scrollCursor);
                  }
               }
            }
         }

         subIndex = updateEllipseGeometry(
               mx+tmr, 
               my+tmb,
               0.1f*tms,
               0.1f*tms,
               2,
               subIndex,
               verts
               );
         elementCoords[i*3+0] = mx+(tmr)*cos(degToRad(direction))-tmb*sin(degToRad(direction));
         elementCoords[i*3+1] = my+(tmr)*sin(degToRad(direction))-tmb*cos(degToRad(direction));
         elementCoords[i*3+2] = tms*deployed;
      }
   } else {    // Draw animated element diamonds
      float tms = 1.0f;
      float tmr, tma, tmb=0.0f;
      for (unsigned int i = 0; i < numListings+1; i++) {
         tmb = 0.0f;
         tms = 1.0f;
         tmr = ((1.5f+endOffset) + ((float)i+scrollCursor-1.0f)*elementSpacing + (float)drawIndex)*deployed;

         // Special animation curves for selection case listings
         if (i == (unsigned int)round((float)numListings*0.5f)){  
            tms = 1.0f+0.25f*(1.0f-scrollCursor);
            tmb = 0.1f*(1.0f-scrollCursor);
         } 
         if (i+1 == (unsigned int)round((float)numListings*0.5f)){
            tms = 1.0f+0.25f*(scrollCursor);
            tmb = 0.1f*(scrollCursor);
         } 

         // Special animation curves for end-case listings
         if (i == 0) {
            tma = -3.0f*pow(scrollCursor, 2.0f) + 4.0f*scrollCursor;
            tms = scrollCursor;
            tmr = ((1.5f+endOffset) + ((float)i - tma + 1.0f)*elementSpacing + (float)drawIndex)*deployed;
         } 
         if (i == numListings) {
            tms -= abs(scrollCursor);
            tma = -3.0f*pow(1.0f-scrollCursor, 2.0f) + 4.0f*(1.0f-scrollCursor);
            tmr = ((1.5f+endOffset) + ((float)i + tma - 2.0f)*elementSpacing + (float)drawIndex)*deployed;
         }

         subIndex = updateEllipseGeometry(
               mx+tmr,
               my+tmb,
               0.1f*tms,
               0.1f*tms,
               2,
               subIndex,
               verts
               );
         elementCoords[i*3+0] = mx+(tmr)*cos(degToRad(direction))-tmb*sin(degToRad(direction));
         elementCoords[i*3+1] = my+(tmr)*sin(degToRad(direction))-tmb*cos(degToRad(direction));
         elementCoords[i*3+2] = tms*deployed;
      }
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
   float mirror = 1.0f;
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
         270.0f,                       // arch start in degrees
         360.0f,                       // arch end in degrees
         0.125f,                       // arch outer radius
         circleSegments/5,
         subIndex,
         verts
         );

   subIndex = updateArchGeometry(
         mx+1.25f*deployed,            // x-position
         mirror*(my-1.125f*deployed),  // y-position
         0.0f, 0.0f,                   // x,y inner radii
         180.0f,                       // arch start in degrees
         270.0f,                       // arch end in degrees
         0.125f,                       // arch outer radius
         circleSegments/5,
         subIndex,
         verts
         );

   subIndex = updateQuad2ptGeometry(
         mx+1.25f*deployed-0.125f,     // x-position
         mirror*(my-1.125f*deployed),  // y-position
         mx+6.25f*deployed+0.125f,     // x-position
         mirror*(my - 0.5f*deployed),  // y-position
         subIndex,
         verts
         );

   subIndex = updateQuad2ptGeometry(
         mx+1.25f*deployed,            // x-position
         mirror*(my-1.125f*deployed),  // y-position
         mx+6.25f*deployed,            // x-position
         mirror*(my-1.250f*deployed),  // y-position
         subIndex,
         verts
         );

   float diffSize, scrollbarOffset, normCursor;
   scrollbarOffset   = 2.5f*((float)numListings/(float)numElements);
   diffSize          = 2.5f - scrollbarOffset;

   // Animate scrollbar midpoint
   if (floatingIndex <= (float)(numElements-1)){
      normCursor     = rangeShift(floatingIndex, (float)(numElements-1), 0.0f, 3.75f-diffSize, 3.75f+diffSize);
   } else if (floatingIndex > (float)(numElements-1)) { // Handle transition from start/end for carousel menus
      normCursor     = rangeShift(scrollCursor, 0.0f, 1.0f, 3.75f-diffSize, 3.75f+diffSize);
   } else {
      normCursor     = 0.0f;
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
         mirror*(my+0.85f*pow(deployed, 3.0f)),
         mx+tmo*deployed-0.20f*deployed,
         mirror*(my+1.00f*pow(deployed, 3.0f)),
         arrowRad,
         circleSegments/5,
         subIndex,
         verts
         );

   subIndex = updatePillGeometry(
         mx+tmo*deployed,
         mirror*(my+0.85f*pow(deployed, 3.0f)),
         mx+tmo*deployed+0.20f*deployed,
         mirror*(my+1.00f*pow(deployed, 3.0f)),
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

   return subIndex;
}

unsigned int drawMenuOverflow(
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

   return subIndex;
}
