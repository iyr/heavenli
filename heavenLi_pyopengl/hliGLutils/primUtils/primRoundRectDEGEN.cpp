
using namespace std;

unsigned int defineRoundRect(
      float px,                  // X-Coordinate of first corner     
      float py,                  // Y-Coordinate of first corner     
      float qx,                  // X-Coordinate of second corner    
      float qy,                  // Y-Coordinate of second corner    
      float cr,                  // Radius of rounded corner         
      GLuint circleSegments,     // Number of polygons to draw corner arches
      float *color,              // Polygon Color                    
      std::vector<float> &verts, // Input Vector of x,y coordinates  
      std::vector<float> &colrs  // Input Vector of r,g,b,a values     
      ){

   defineArch(px, py, 0.0f, 0.0f,  90.0f, 180.0f, cr, circleSegments, color, verts, colrs);
   defineArch(px, qy, 0.0f, 0.0f, 180.0f, 270.0f, cr, circleSegments, color, verts, colrs);
   defineArch(qx, qy, 0.0f, 0.0f, 270.0f, 360.0f, cr, circleSegments, color, verts, colrs);
   defineArch(qx, py, 0.0f, 0.0f, 360.0f,  90.0f, cr, circleSegments, color, verts, colrs);

   defineQuad2pt(px-cr, py, px, qy, color, verts, colrs);
   defineQuad2pt(px, py+cr, qx, qy, color, verts, colrs);
   defineQuad2pt(qx+cr, qy, qx, py, color, verts, colrs);
   defineQuad2pt(qx, qy-cr, px, qy, color, verts, colrs);

   return verts.size()/2;
}

unsigned int updateRoundRect(
      float px,               // X-Coordinate of first corner     
      float py,               // Y-Coordinate of first corner     
      float qx,               // X-Coordinate of second corner    
      float qy,               // Y-Coordinate of second corner    
      float cr,               // Radius of rounded corner         
      GLuint circleSegments,  // Number of polygons to draw corner arches
      int   index,            // Index of where to start writing in input array
      float *verts            // Input Vector of x,y coordinates
      ){

   index = updateArchGeometry(px, py, 0.0f, 0.0f,  90.0f, 180.0f, cr, circleSegments, index, verts);
   index = updateArchGeometry(px, qy, 0.0f, 0.0f, 180.0f, 270.0f, cr, circleSegments, index, verts);
   index = updateArchGeometry(qx, qy, 0.0f, 0.0f, 270.0f, 360.0f, cr, circleSegments, index, verts);
   index = updateArchGeometry(qx, py, 0.0f, 0.0f, 360.0f,  90.0f, cr, circleSegments, index, verts);

   index = updateQuad2ptGeometry(px-cr, py, px, qy, index, verts);
   index = updateQuad2ptGeometry(px, py+cr, qx, qy, index, verts);
   index = updateQuad2ptGeometry(qx+cr, qy, qx, py, index, verts);
   index = updateQuad2ptGeometry(qx, qy-cr, px, qy, index, verts);

   return index;
}

unsigned int updateRoundRect(
      GLuint circleSegments,  // Number of polygons to draw corner arches
      float *color,           // Polygon Color                    
      int   index,            // Index of where to start writing in input array
      float *colrs            // Input Array of r,g,b,a values
      ){

   index = updateArchColor(circleSegments, color, index, colrs);
   index = updateArchColor(circleSegments, color, index, colrs);
   index = updateArchColor(circleSegments, color, index, colrs);
   index = updateArchColor(circleSegments, color, index, colrs);

   index = updateQuadColor(color, index, colrs);
   index = updateQuadColor(color, index, colrs);
   index = updateQuadColor(color, index, colrs);
   index = updateQuadColor(color, index, colrs);

   return index;
}
