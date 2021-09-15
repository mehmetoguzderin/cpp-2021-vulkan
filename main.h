#ifdef SHADER_GLSL
#define REFERENCE_VARIABLE(type, name) inout type name
#else
#define REFERENCE_VARIABLE(type, name) type& name
#endif

// Begin http://jcgt.org/published/0007/03/04/

mat3 quat2mat(vec4 q) {
  q *= 1.41421356;  // sqrt(2)
  return mat3(1.0 - q.y * q.y - q.z * q.z, q.x * q.y + q.w * q.z, q.x * q.z - q.w * q.y, q.x * q.y - q.w * q.z, 1.0 - q.x * q.x - q.z * q.z,
              q.y * q.z + q.w * q.x, q.x * q.z + q.w * q.y, q.y * q.z - q.w * q.x, 1.0 - q.x * q.x - q.y * q.y);
}

struct Ray {
  vec3 origin;
  vec3 direction;
};

struct Box {
  vec3 center;
  vec3 radius;
  vec3 invRadius;
  mat3 rotation;
};

float safeInverse(float x) {
  return (x == 0.0) ? 1e12 : (1.0 / x);
}

vec3 safeInverse(vec3 v) {
  return vec3(safeInverse(v.x), safeInverse(v.y), safeInverse(v.z));
}

float maxComponent(vec3 v) {
  return max(max(v.x, v.y), v.z);
}

// vec3 box.radius:       independent half-length along the X, Y, and Z axes
// mat3 box.rotation:     box-to-world rotation (orthonormal 3x3 matrix) transformation
// bool rayCanStartInBox: if true, assume the origin is never in a box. GLSL optimizes this at compile time
// bool oriented:         if false, ignore box.rotation
bool intersectBoxCommon(Box box,
                        Ray ray,
                        REFERENCE_VARIABLE(float, distance),
                        REFERENCE_VARIABLE(vec3, normal),
                        const bool rayCanStartInBox,
                        REFERENCE_VARIABLE(bool, oriented),
                        REFERENCE_VARIABLE(vec3, _invRayDirection)) {
  // Move to the box's reference frame. This is unavoidable and un-optimizable.
  ray.origin = box.rotation * (ray.origin - box.center);
  if (oriented) {
    ray.direction = ray.direction * box.rotation;
  }

  // This "rayCanStartInBox" branch is evaluated at compile time because `const` in GLSL
  // means compile-time constant. The multiplication by 1.0 will likewise be compiled out
  // when rayCanStartInBox = false.
  float winding;
  if (rayCanStartInBox) {
    // Winding direction: -1 if the ray starts inside of the box (i.e., and is leaving), +1 if it is starting outside of the box
    winding = (maxComponent(abs(ray.origin) * box.invRadius) < 1.0) ? -1.0 : 1.0;
  } else {
    winding = 1.0;
  }

  // We'll use the negated sign of the ray direction in several places, so precompute it.
  // The sign() instruction is fast...but surprisingly not so fast that storing the result
  // temporarily isn't an advantage.
  vec3 sgn = -sign(ray.direction);

  // Ray-plane intersection. For each pair of planes, choose the one that is front-facing
  // to the ray and compute the distance to it.
  vec3 distanceToPlane = box.radius * winding * sgn - ray.origin;
  if (oriented) {
    distanceToPlane /= ray.direction;
  } else {
    distanceToPlane *= _invRayDirection;
  }

  // Perform all three ray-box tests and cast to 0 or 1 on each axis.
  // Use a macro to eliminate the redundant code (no efficiency boost from doing so, of course!)
  // Could be written with
#define TEST(U, V, W)                                                                                                  \
  /* Is there a hit on this axis in front of the origin? Use multiplication instead of && for a small speedup */       \
  (distanceToPlane.U >= 0.0) && /* Is that hit within the face of the box? */                                          \
      all(lessThan(abs(vec2(ray.origin.V, ray.origin.W) + vec2(ray.direction.V, ray.direction.W) * distanceToPlane.U), \
                   vec2(box.radius.V, box.radius.W)))

  bvec3 test = bvec3(TEST(x, y, z), TEST(y, z, x), TEST(z, x, y));

  // CMOV chain that guarantees exactly one element of sgn is preserved and that the value has the right sign
  sgn = test.x ? vec3(sgn.x, 0.0, 0.0) : (test.y ? vec3(0.0, sgn.y, 0.0) : vec3(0.0, 0.0, test.z ? sgn.z : 0.0));
#undef TEST

  // At most one element of sgn is non-zero now. That element carries the negative sign of the
  // ray direction as well. Notice that we were able to drop storage of the test vector from registers,
  // because it will never be used again.

  // Mask the distance by the non-zero axis
  // Dot product is faster than this CMOV chain, but doesn't work when distanceToPlane contains nans or infs.
  //
  distance = (sgn.x != 0.0) ? distanceToPlane.x : ((sgn.y != 0.0) ? distanceToPlane.y : distanceToPlane.z);

  // Normal must face back along the ray. If you need
  // to know whether we're entering or leaving the box,
  // then just look at the value of winding. If you need
  // texture coordinates, then use box.invDirection * hitPoint.

  if (oriented) {
    normal = box.rotation * sgn;
  } else {
    normal = sgn;
  }

  return (sgn.x != 0) || (sgn.y != 0) || (sgn.z != 0);
}

// Just determines whether the ray hits the axis-aligned box.
// invRayDirection is guaranteed to be finite for all elements.
bool hitAABox(vec3 boxCenter, vec3 boxRadius, vec3 rayOrigin, vec3 rayDirection, vec3 invRayDirection) {
  rayOrigin -= boxCenter;
  vec3 distanceToPlane = (-boxRadius * sign(rayDirection) - rayOrigin) * invRayDirection;

#define TEST(U, V, W)                                                                                             \
  (float(distanceToPlane.U >= 0.0) * float(abs(rayOrigin.V + rayDirection.V * distanceToPlane.U) < boxRadius.V) * \
   float(abs(rayOrigin.W + rayDirection.W * distanceToPlane.U) < boxRadius.W))

  // If the ray is in the box or there is a hit along any axis, then there is a hit
  return bool(float(abs(rayOrigin.x) < boxRadius.x) * float(abs(rayOrigin.y) < boxRadius.y) * float(abs(rayOrigin.z) < boxRadius.z) +
              TEST(x, y, z) + TEST(y, z, x) + TEST(z, x, y));
#undef TEST
}

// There isn't really much application for ray-AABB where we don't check if the ray is in the box, so we
// just give a dummy implementation here to allow the test harness to compile.
bool outsideHitAABox(vec3 boxCenter, vec3 boxRadius, vec3 rayOrigin, vec3 rayDirection, vec3 invRayDirection) {
  return hitAABox(boxCenter, boxRadius, rayOrigin, rayDirection, invRayDirection);
}

// Ray is always outside of the box
bool outsideIntersectBox(Box box,
                         Ray ray,
                         REFERENCE_VARIABLE(float, distance),
                         REFERENCE_VARIABLE(vec3, normal),
                         REFERENCE_VARIABLE(bool, oriented),
                         REFERENCE_VARIABLE(vec3, _invRayDirection)) {
  return intersectBoxCommon(box, ray, distance, normal, false, oriented, _invRayDirection);
}

bool intersectBox(Box box,
                  Ray ray,
                  REFERENCE_VARIABLE(float, distance),
                  REFERENCE_VARIABLE(vec3, normal),
                  REFERENCE_VARIABLE(bool, oriented),
                  REFERENCE_VARIABLE(vec3, _invRayDirection)) {
  return intersectBoxCommon(box, ray, distance, normal, true, oriented, _invRayDirection);
}

// End http://jcgt.org/published/0007/03/04/

#define LOCAL_SIZE 8u
#define TILE_SIZE 384u

#define CONSTANTS    \
  Constants {        \
    ivec2 offset;    \
    ivec2 wh;        \
    vec4 clearColor; \
  }

#ifdef SHADER_GLSL
#else
struct CONSTANTS;
#endif