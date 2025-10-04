#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ glm::vec3 cosWeightedHemiSphereSample(glm::vec3 normal,
    thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(-1, 1);
    glm::vec2 sample = glm::vec2(u01(rng), u01(rng));
    float pi = 3.14159265359;

    // MALLY"S METHOD WITH DISK SAMPLE
    float x = 2.0f * sample[0] - 1.0f;
    float y = 2.0f * sample[1] - 1.0f;
    float r = 1.f;
    float a = 0.f;
    if (x == 0 && y == 0) {
        return normal;
    }
    if (abs(x) > abs(y)) {
        r *= x;
        a = (pi * 0.25) * (y / x);
    }
    else {
        r *= y;
        a = (pi * 0.5) - ((pi * 0.25) * (x / y));
    }


    glm::vec3 localSample = glm::vec3(cos(a) * r, sin(a) * r, sqrt(1 - (r * r)));

    glm::vec3 nt = (abs(normal.x) > 0.1f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    glm::vec3 tangent = glm::normalize(glm::cross(nt, normal));
    glm::vec3 bitangent = glm::cross(normal, tangent);

    return localSample.x * tangent + localSample.y * bitangent + localSample.z * normal;
}

__host__ __device__ glm::vec3 reflect(glm::vec3 normal, glm::vec3 wi) {
    glm::vec3 cross = glm::normalize(glm::cross(normal, wi));
    glm::vec3 planeNorm = glm::normalize(glm::cross(cross, normal));
    return wi - 2.0f * glm::dot(wi, planeNorm) * planeNorm;

}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above

    // CHANGE LATER TO BE BASED OFF OF MATERIAL TYPE

    Ray newRay = Ray();
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 newRayDir;
    if (u01(rng) >  1.f - m.specular.exponent) {
        //DIFFUSE
        newRayDir = calculateRandomDirectionInHemisphere(normal, rng);
    }
    else {
        //MIRROR
        newRayDir = reflect(normal, -glm::normalize(pathSegment.ray.direction));

    }
    newRay.direction = glm::normalize(newRayDir);
    newRay.origin = intersect + normal * 0.001f;
    pathSegment.ray = newRay;
    pathSegment.remainingBounces--;
}

__host__ __device__
float windowWeight(float r, float oRad, float iRad) {
    float normalizedDist = (r - iRad) / (oRad - iRad);
    normalizedDist = glm::clamp(normalizedDist, 0.0f, 1.0f);
    float shapedFalloff = glm::pow(1.0f - normalizedDist, 2.0f) * glm::smoothstep(0.0f, 0.1f, normalizedDist);
    return glm::abs(1.0f - normalizedDist);
}

__host__ __device__ glm::vec3 bhAccel(glm::vec3 r, float h2, float M, float w) {
    float rL = glm::length(r);
    float radL5 = rL * rL * rL * rL * rL;
    return (-3.0f * M * h2) * (1.0f / radL5) * r * w;
}

__host__ __device__ void rk4Step(glm::vec3& r, glm::vec3& v, float h2, float dt, float M, float w) {
    // K1
    glm::vec3 K1r = v;
    glm::vec3 K1v = bhAccel(r, h2, M, w);
    
    // K2
    glm::vec3 K2r = v + 0.5f * dt * K1v;
    glm::vec3 K2v = bhAccel(r + 0.5f * dt * K1r, h2, M, w);

    // K3
    glm::vec3 K3r = v + 0.5f * dt * K2v;
    glm::vec3 K3v = bhAccel(r + 0.5f * dt * K2r, h2, M, w);

    // K4
    glm::vec3 K4r = v +  dt * K3v;
    glm::vec3 K4v = bhAccel(r + dt * K3r, h2, M, w);

    r += (dt / 6.0f) * (K1r + 2.0f * K2r + 2.0f * K3r + K4r);
    v += (dt / 6.0f) * (K1v + 2.0f * K2v + 2.0f * K3v + K4v);
}

__host__ __device__ float chooseDt(glm::vec3 r, glm::vec3 v, float h2, float M, float w, float iRad, float oRad) {
    glm::vec3 a = bhAccel(r, h2, M, w);
    glm::vec3 vPerp = glm::normalize(v);
    glm::vec3 aPerp = a - glm::dot(a, vPerp) * vPerp;
    float dtMin = 0.001f * iRad;
    float dtMax = 0.1f * oRad;
    float eps = 0.2f;
    float amin = 1e-6f;
    float dt = (eps * glm::length(v)) / glm::max(glm::length(aPerp), amin);
    return glm::clamp(dt, dtMin, dtMax);
}


// 2D hash function for pseudorandom gradients
__host__ __device__ inline glm::vec2 random2(glm::vec2 p) {
    p = glm::vec2(glm::dot(p, glm::vec2(127.1f, 311.7f)),
        glm::dot(p, glm::vec2(269.5f, 183.3f)));
    return glm::fract(glm::sin(p) * 43758.5453123f) * 2.0f - 1.0f;
}

// 2D Perlin noise
__host__ __device__ float noise(glm::vec2 x) {
    // grid
    glm::vec2 p = glm::floor(x);
    glm::vec2 w = glm::fract(x);

    // quintic interpolant
    glm::vec2 u = w * w * w * (w * (w * 6.0f - 15.0f) + 10.0f);

    // gradients
    glm::vec2 ga = random2(p + glm::vec2(0.0f, 0.0f));
    glm::vec2 gb = random2(p + glm::vec2(1.0f, 0.0f));
    glm::vec2 gc = random2(p + glm::vec2(0.0f, 1.0f));
    glm::vec2 gd = random2(p + glm::vec2(1.0f, 1.0f));

    // projections
    float va = glm::dot(ga, w - glm::vec2(0.0f, 0.0f));
    float vb = glm::dot(gb, w - glm::vec2(1.0f, 0.0f));
    float vc = glm::dot(gc, w - glm::vec2(0.0f, 1.0f));
    float vd = glm::dot(gd, w - glm::vec2(1.0f, 1.0f));

    // interpolation
    return va +
        u.x * (vb - va) +
        u.y * (vc - va) +
        u.x * u.y * (va - vb - vc + vd);
}

__host__ __device__ glm::vec2 swirl(glm::vec2 p, float swirlFactor) {
    float r = glm::length(p);
    float theta = glm::atan(p.y, p.x);
    theta += swirlFactor * r;
    return glm::vec2(r * glm::cos(theta), r * glm::sin(theta));
}



__host__ __device__ void blackHoleRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::mat4 invTrans,
    const Material& m,
    thrust::default_random_engine& rng
) {
    glm::vec3 bhCenter = intersect - normal * m.blackHole.oRad;
    glm::vec3 r = intersect - bhCenter;
    glm::vec3 prevR = r;
    glm::vec3 v = glm::normalize(pathSegment.ray.direction);
    float h2 = glm::length(glm::cross(r, v)) * glm::length(glm::cross(r, v));
    float M = 0.5 * m.blackHole.iRad;
    float w = 1.0f;
    int maxSteps = 1024;
    float dt = 0.001f;

    glm::vec3 diskNorm = glm::vec3((invTrans * glm::vec4(0.0, 1.0, 0.0, 0.0)));

    Ray newRay = Ray();
    for (int i = 0; i < maxSteps; i++) {
        if (prevR.y * r.y < 0.0) {
            float t = prevR.y / (prevR.y - r.y);
            glm::vec3 crossingPoint = prevR + t * (r - prevR);

            float distFromCenter = glm::length(crossingPoint);
            float normalizedDist = (distFromCenter - m.blackHole.iRad) / (m.blackHole.oRad - m.blackHole.iRad);
            normalizedDist = glm::clamp(normalizedDist, 0.0f, 1.0f);

            glm::vec2 swirlPos = swirl(glm::vec2(crossingPoint.x, crossingPoint.z), 0.4);

            float shapedFalloff = glm::pow(1.0f - normalizedDist, 2.0f) * glm::smoothstep(0.0f, 0.1f, normalizedDist);
            shapedFalloff = shapedFalloff * 0.75f + shapedFalloff * noise(swirlPos);

            thrust::uniform_real_distribution<float> u01(0, 1);
            if (u01(rng) < shapedFalloff) {
                pathSegment.color *= m.color * shapedFalloff * m.emittance;
                pathSegment.remainingBounces = -1;
                return;
            }
        }
        if (glm::length(r) < m.blackHole.iRad) {
            pathSegment.color *= glm::vec3(0.0);
            pathSegment.remainingBounces = -1;
            return;
        }
        if (glm::length(r) > m.blackHole.oRad && glm::dot(r, v) > 0.0f) {
            newRay.direction = glm::normalize(v);
            newRay.origin = bhCenter + r;
            newRay.origin += newRay.direction * 0.001f;
            pathSegment.ray = newRay;
            pathSegment.remainingBounces--;
            return;
        }
        
        
        w = windowWeight(glm::length(r), m.blackHole.oRad, m.blackHole.iRad);
        prevR = r;
        // RH4 STEP
        dt = chooseDt(r, v, h2, M, w, m.blackHole.iRad, m.blackHole.oRad);
        rk4Step(r, v, h2, dt, M, w);
    }

    newRay.direction = glm::normalize(v);
    newRay.origin = bhCenter + r;
    newRay.origin += newRay.direction * 0.001f;
    pathSegment.ray = newRay;
    pathSegment.remainingBounces--;
    return;
}
