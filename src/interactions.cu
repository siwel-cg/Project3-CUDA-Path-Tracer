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

    //DIFFUSE
    glm::vec3 newRayDir = calculateRandomDirectionInHemisphere(normal, rng);
    newRay.direction = newRayDir;
    newRay.origin = intersect + normal * 0.001f;

    //MIRROR
    //glm::vec3 newRayDir = reflect(normal, pathSegment.ray.direction);
    //newRay.direction = newRayDir;
    //newRay.origin = intersect + normal * 0.001f;

    pathSegment.ray = newRay;
    pathSegment.remainingBounces--;
}
