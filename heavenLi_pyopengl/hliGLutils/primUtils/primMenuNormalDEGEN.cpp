/*
 * Defines a drop menu for 3 or fewer listings
 */

unsigned int defineMenuNormal(
      float          direction,        // Direction, in degrees, the menu slides out to
      float          deployed,         // 0.0=closed, 1.0=completely open
      float          floatingIndex,    // index of the selected element, used for scroll bar
      float          scrollCursor,     // element animation cursor for scrolling: -1.0 to 1.0
      unsigned int   numElements,      // number of elements
      unsigned int   circleSegments,   // number of polygon segments
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
         tmo         = 0.0f,                    // Overall menu length (not including endcap circle radius)
         arrowRad    = 0.05f*pow(deployed, 2.0f),  // arrow thickness
         endOffset   = 0.5f,                          // distance from last element to end of menu
         elementSpacing = ((6.0f-endOffset)-(1.5f+endOffset))/2.0f;  // distance between elements

   tmo = 0.25f + numElements*1.75f;

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
   float* glCoords = new float[(numElements+1)*3];
   defineElementCoords(
      direction,
      deployed,
      floatingIndex,
      scrollCursor,
      numElements,
      1,
      numElements,
      selectFromScroll,
      glCoords,
      elementCoords
      );

   for (unsigned int i = 0; i < numElements; i++){
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

   tmo = 3.75f;
   float diffElements = floor((float)numElements*0.5f); // number of elements that straddle selected element
   if (floatingIndex <= diffElements) {
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

   } 
   //else {
      //tmo = 3.75f;
   //}
  
   //if (scrollCursor == 0.0f) {   // Resolve edge-case bug
      //if (floatingIndex == diffElements) {
         //tmo = 3.75f;
      //}
   //}

   if (numElements < 4) tmo -= 1.75f;
   // Flip selection arrow
   float mirror = 1.0f;
         //theta0 = 270.0f,
         //theta1 = 360.0f;
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
      //theta0 = 0.0f;
      //theta1 = 90.0f;
      mirror = -1.0f;
   }

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

   /*
    * Proximal Bar
    */
   tmo = 1.0f + arrowRad;
   definePill(
         mx+tmo*deployed,
         my,
         mx+tmo*deployed,
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
         mx+tmo*deployed,
         my-0.75f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         detailColor,
         verts,
         colrs
         );
   return verts.size()/2;
}

unsigned int updateMenuNormalGeometry(
      float          direction,     // Direction, in degrees, the menu slides out to
      float          deployed,      // 0.0=closed, 1.0=completely open
      float          floatingIndex, // index of the selected element, used for scroll bar
      float          scrollCursor,  // element animation cursor for scrolling: -1.0 to 1.0
      unsigned int   numElements,   // number of elements
      unsigned int   circleSegments,// number of polygon segments
      bool           selectFromScroll,
      float*         elementCoords, // Relative coordinates of Menu elements
      float          w2h,           // width to height ratio
      unsigned int   index,         // Index of where to start writing to input array
      float*         verts          // Input Vector of x,y coordinates
      ){

   unsigned int subIndex = index;
   float mx          = 0.0f,              // Origin of Menu x coord
         my          = 0.0f,                 // Origin of Menu y coord
         tmo         = 0.0f,                    // Overall menu length (not including endcap circle radius)
         arrowRad    = 0.05f*pow(deployed, 2.0f),  // arrow thickness
         endOffset   = 0.5f,                          // distance from last element to end of menu
         elementSpacing = (6.0f-1.5f-endOffset*2.0f)/2.0f;  // distance between elements

   tmo = 0.25f + numElements*1.75f;

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

   float* glCoords = new float[(numElements+1)*3];
   defineElementCoords(
      direction,
      deployed,
      floatingIndex,
      scrollCursor,
      numElements,
      1,
      numElements,
      selectFromScroll,
      glCoords,
      elementCoords
      );

   for (unsigned int i = 0; i < numElements; i++){
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

   tmo = 3.75f;
   float diffElements = floor((float)numElements*0.5f); // number of elements that straddle selected element

   if (floatingIndex <= diffElements) {
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

   } 
   //else {
      //tmo = 3.75f;
   //}
  
   //if (scrollCursor == 0.0f) {   // Resolve edge-case bug
      //if (floatingIndex == diffElements) {
         //tmo = 3.75f;
      //}
   //}

   if (numElements < 4) tmo -= 1.75f;

   // Flip selection arrow, scrollbar based on direction of menu
   float mirror = 1.0f;
         //theta0 = 270.0f,
         //theta1 = 360.0f;
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
      //theta0 = 0.0f;
      //theta1 = 90.0f;
      mirror = -1.0f;
   }

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

   /*
    * Proximal Bar
    */
   tmo = 1.0f + arrowRad;
   subIndex = updatePillGeometry(
         mx+tmo*deployed,
         my,
         mx+tmo*deployed,
         my+0.75f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         subIndex,
         verts
         );
   subIndex = updatePillGeometry(
         mx+tmo*deployed,
         my,
         mx+tmo*deployed,
         my-0.75f*pow(deployed, 3.0f),
         arrowRad,
         circleSegments/5,
         subIndex,
         verts
         );
   return subIndex;
}

unsigned int updateMenuNormalColors(
      unsigned int   numElements,   // number of elements to display at once
      unsigned int   circleSegments,// number of polygon segments
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
   for (unsigned int i = 0; i < numElements; i++) {
      subIndex = updateEllipseColor(
            2,
            detailColor,
            subIndex,
            colrs
            );
   }

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
