#version 100
attribute   vec4 vertCoord;
attribute   vec4 vertColor;
attribute   vec4 vertNorml;
attribute   vec2 vertTexUV;

uniform    mat4 MVP;

varying    vec4 color;
varying    vec2 texCoord;

void main() {
   color = vertColor;
   gl_Position = MVP * vertCoord;
   texCoord = vertTexUV;
}
