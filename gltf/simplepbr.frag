#version 130

in vec2 texcoord;

uniform struct {
  vec4 baseColor;
  float roughness;
  float metallic;
  float refractiveIndex;
} p3d_Material;

uniform sampler2D p3d_Texture0;

out vec4 color;

// Give texture slots names
#define p3d_TextureBaseColor p3d_Texture0

void main() {
  color = p3d_Material.baseColor * texture(p3d_TextureBaseColor, texcoord);
}
