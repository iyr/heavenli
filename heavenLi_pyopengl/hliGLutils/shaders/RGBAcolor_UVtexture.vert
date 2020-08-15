#version 100
attribute  vec2 vertCoord;
attribute  vec2 vertTexUV;
attribute  vec4 vertColor;

uniform    mat4 MVP;

varying    vec4 color;
varying    vec2 texCoord;

void main() {
   color = vertColor;
   gl_Position = MVP * vec4(vertCoord, 1, 1);
   texCoord = vertTexUV;
}
