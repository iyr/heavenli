/*
 * A struct for telling which vertex shader attributes
 * map to which data sets
 * (Vertex Attribute Strings)
 */

struct VertexAttributeStrings {
   const vector<string> AttribStrings 
   {
      "vertCoord",
      "vertColor",
      "vertNorml",
      "vertTexUV"
   };
   const string coordData = AttribStrings[0];
   const string colorData = AttribStrings[1];
   const string normlData = AttribStrings[2];
   const string texuvData = AttribStrings[3];

   //string coordData = "vertCoord";
   //string colorData = "vertColor";
   //string normlData = "vertNorml";
   //string texuvData = "vertTexUV";
} VAS;
