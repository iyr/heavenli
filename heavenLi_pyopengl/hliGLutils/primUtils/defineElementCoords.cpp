// Helper function for defining the relative positon of menu elements

void defineElementCoords(
      float          direction,        // Direction menu slides out (degrees, unit circle)
      float          deployed,         // animation cursor for menu slide-out (0.0==closed)
      float          floatingIndex,    // index of currently selected index
      float          scrollCursor,     // animation cursor for scrolling
      unsigned int   numElements,      // number of list elements
      unsigned int   menuLayout,       // linear/circular
      unsigned int   numListings,      // number of elements to display at any given time
      bool           selectFromScroll, // whether or not elements are selected by merely scrolling to them
      float*         glCoords,         // base, unmodified coordinates
      float*         elementCoords     // element coordinates rotated by direction
      ){
   float mx          = 0.0f,              // Origin of Menu x coord
         my          = 0.0f,                 // Origin of Menu y coord
         endOffset   = 0.0f,
         diffElements = floor((float)numListings*0.5f),  // number of elements that straddle selected element
         elementSpacing = 0.0f;

   if (numListings < 1) numListings = 3;

   if (numElements < 3) endOffset = 0.5f;
   else                 endOffset = 1.0f/(float)(numListings-1);
   if (numElements < 3) elementSpacing = (6.0f-1.5f-endOffset*2.0f)/2.0f;
   else                 elementSpacing = (6.0f-1.5f-endOffset*2.0f)/(float)(numListings-1);
   if (numElements < 4) menuLayout = 1;

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

   // Avoid extra ops when not animating by drawing diamonds statically
   if (  (menuLayout == 1 && floatingIndex < diffElements)
         || 
         (menuLayout == 1 && floatingIndex >= (float)(numElements-1)-diffElements)
         ||
         abs(scrollCursor) == 0.0f
         ){    
      float tms = 1.0f;
      float tms2 = 1.0f;
      float tmr, tmb=0.0f;

      unsigned int limit = 0;
      if (numElements < 3) 
      {
         limit = numElements;
         tms2  = 1.0f;
      } else 
      {
         limit = numListings+1;
         tms2  = 3.0f/numListings;
      }

      for (unsigned int i = 0; i < limit; i++) {
         tms = 1.0f;
         tmb = 0.0f;
         tmr = ((1.5f+endOffset) + (float)i*elementSpacing)*deployed;

         if (menuLayout == 0) {
            if (i == (unsigned int)round((float)numListings*0.5f)){
               tmr = ((1.5f+endOffset) + (float)i*elementSpacing)*deployed;
               tms = 1.0f;
               tmb = 0.0f;
            } 
            if (i+1 == (unsigned int)round((float)numListings*0.5f)){
               tmr = ((1.5f+endOffset) + (float)i*elementSpacing)*deployed;
               tms = 1.0f+0.25f*(float)selectFromScroll*(float)selectFromScroll;
               tmb =       0.1f*(float)selectFromScroll;
            }
         }

         if (menuLayout == 1) {
            // Handle selection of elements in middle
            if (  (
                  floatingIndex >= diffElements
                  &&
                  floatingIndex <= (float)(numElements-1)-diffElements
                  )
                  ){
               if (i == (unsigned int)round((float)numListings*0.5f)){
                  tms = 1.0f;
                  tmb = 0.0f;
               } 
               if (i+1 == (unsigned int)round((float)numListings*0.5f)){
                  tms = 1.0f+0.25f*(float)selectFromScroll;
                  tmb =       0.1f*(float)selectFromScroll;
               }
            }
            // Handle selection of elements on ends
            if (  (
                  floatingIndex < diffElements
                  ||
                  floatingIndex > (float)(numElements-1)-diffElements
                  )
               ) {
               //printf("numListings: %i\nnumElements: %i\nfloatingIndex: %f\n", numListings, numElements, floatingIndex);

               // Handle selection of elements at start of list
               if (  i == numListings-1-(unsigned int)floor(floatingIndex)
                     &&
                     i >= floor(diffElements)                  // Resolve edge-case bug
                     ){
                  if (floatingIndex == floor(floatingIndex)) { // Resolve edge-case bug
                     tms = 1.0f+0.25f*(float)selectFromScroll;
                     tmb =       0.1f*(float)selectFromScroll;
                  } else {
                     tms = 1.0f+0.25f*(1.0f-scrollCursor)*(float)selectFromScroll;
                     tmb =       0.1f*(1.0f-scrollCursor)*(float)selectFromScroll;
                  }
               }
               if (  i+1 == numListings-1-(unsigned int)floor(floatingIndex)
                     &&
                     i >= floor(diffElements)                  // Resolve edge-case bug
                     ){
                  if (floatingIndex == floor(floatingIndex)) { // Resolve edge-case bug
                     tms = 1.0f;
                     tmb = 0.0f;
                  } else {
                     tms = 1.0f+0.25f*(scrollCursor)*(float)selectFromScroll;
                     tmb =       0.1f*(scrollCursor)*(float)selectFromScroll;
                  }
               }

               // Handle selection of elements at end of list
               if (  i == numElements-1-(unsigned int)floor(floatingIndex)
                     &&
                     i <= floor(diffElements)                  // Resolve edge-case bug
                     ){
                  if (floatingIndex == floor(floatingIndex)) { // Resolve edge-case bug
                     tms = 1.0f+0.25f*(float)selectFromScroll;
                     tmb =       0.1f*(float)selectFromScroll;
                  } else {
                     tms = 1.0f+0.25f*(1.0f-scrollCursor)*(float)selectFromScroll;
                     tmb =       0.1f*(1.0f-scrollCursor)*(float)selectFromScroll;
                  }
               }
               if (  i+1 == numElements-1-(unsigned int)floor(floatingIndex)
                     &&
                     i+1 <= floor(diffElements)                // Resolve edge-case bug
                     ){
                  if (floatingIndex == floor(floatingIndex)) { // Resolve edge-case bug
                     tms = 1.0f;
                     tmb = 0.0f;
                  } else {
                     tms = 1.0f+0.25f*(scrollCursor)*(float)selectFromScroll;
                     tmb =       0.1f*(scrollCursor)*(float)selectFromScroll;
                  }
               }
            }
         }
         if (i == numListings) tms = 0.0f;

         tms *= 0.75f*tms2;
         elementCoords[i*3+0] = (float)(mx+(tmr)*cos(degToRad(direction))+mirror*tmb*sin(degToRad(-direction)));
         elementCoords[i*3+1] = (float)(my+(tmr)*sin(degToRad(direction))+mirror*tmb*cos(degToRad(-direction)));
         elementCoords[i*3+2] = tms*deployed;

         glCoords[i*3+0] = mx+tmr;
         glCoords[i*3+1] = my+tmb;
         glCoords[i*3+2] = tms;
      }
   } 
   else  // Draw animated element diamonds 
   {    
      float tms = 1.0f;
      float tms2 = 1.0f;
      float tmr, tma, tmb=0.0f;
      unsigned int limit = 0;
      if (numElements < 3) 
      {
         limit = numElements;
         tms2  = 1.0f;
      } else 
      {
         limit = numListings+1;
         tms2  = 3.0f/numListings;
      }

      for (unsigned int i = 0; i < limit; i++) {
         tmb = 0.0f;
         tms = 1.0f;
         tmr = ((1.5f+endOffset) + ((float)i+scrollCursor-1.0f)*elementSpacing)*deployed;

         // Special animation curves for selection case listings
         if (i == (unsigned int)round((float)numListings*0.5f)){  
            tms = 1.0f+0.25f*(1.0f-scrollCursor)*(float)selectFromScroll;
            tmb =       0.1f*(1.0f-scrollCursor)*(float)selectFromScroll;
         } 
         if (i+1 == (unsigned int)round((float)numListings*0.5f)){
            tms = 1.0f+0.25f*(scrollCursor)*(float)selectFromScroll;
            tmb =       0.1f*(scrollCursor)*(float)selectFromScroll;
         } 

         // Special animation curves for end-case listings
         if (i == 0) {
            tma = -3.0f*pow(scrollCursor, 2.0f) + 4.0f*scrollCursor;
            tms = scrollCursor;
            tmr = ((1.5f+endOffset) + ((float)i - tma + 1.0f)*elementSpacing)*deployed;
         } 
         if (i == numListings) {
            tma = -3.0f*pow(1.0f-scrollCursor, 2.0f) + 4.0f*(1.0f-scrollCursor);
            tms -= abs(scrollCursor);
            tmr = ((1.5f+endOffset) + ((float)i + tma - 2.0f)*elementSpacing)*deployed;
         }

         tms *= 0.75f*tms2;

         elementCoords[i*3+0] = (float)(mx+(tmr)*cos(degToRad(direction))+mirror*tmb*sin(degToRad(-direction)));
         elementCoords[i*3+1] = (float)(my+(tmr)*sin(degToRad(direction))+mirror*tmb*cos(degToRad(-direction)));
         elementCoords[i*3+2] = tms*deployed;

         glCoords[i*3+0] = mx+tmr;
         glCoords[i*3+1] = my+tmb;
         glCoords[i*3+2] = tms;
      }
   }

   return;
}

