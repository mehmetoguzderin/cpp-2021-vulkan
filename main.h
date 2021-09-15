#define LOCAL_SIZE 8u
#define TILE_SIZE 256u

#define CONSTANTS        \
  Constants {            \
    int offset[2];       \
    int wh[2];           \
    float clearColor[4]; \
  }

#ifdef SHADER_GLSL
#else
struct CONSTANTS;
#endif