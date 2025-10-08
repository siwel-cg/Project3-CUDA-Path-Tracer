#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include <thrust/sort.h>
#include <thrust/partition.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static glm::vec3* dev_bloomImage = NULL;
static glm::vec3* dev_bloomMask = NULL;
static glm::vec3* dev_bloomMaskBlur = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static float* dev_EnviMap = NULL;
static int* dev_matIds = NULL;

static bvhNode* dev_bvhTree = NULL;
static int* dev_bvhGeoIdx = NULL;

thrust::device_ptr<int> dev_thrust_matId;
thrust::device_ptr<PathSegment> dev_thrust_pathIdx;
thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_bloomImage, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_bloomImage, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_bloomMask, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_bloomMask, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_bloomMaskBlur, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_bloomMaskBlur, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_bvhTree, scene->bvhTree.size() * sizeof(bvhNode));
    cudaMemcpy(dev_bvhTree, scene->bvhTree.data(), scene->bvhTree.size() * sizeof(bvhNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_bvhGeoIdx, scene->bvhGeoIdx.size() * sizeof(int));
    cudaMemcpy(dev_bvhGeoIdx, scene->bvhGeoIdx.data(), scene->bvhGeoIdx.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_matIds, pixelcount * sizeof(int));

    if (hst_scene->enviMap != nullptr && hst_scene->enviMap->image != nullptr) {
        int enviSize = hst_scene->enviMap->width * hst_scene->enviMap->height * hst_scene->enviMap->channels;

        if (enviSize > 0) {
            cudaMalloc(&dev_EnviMap, enviSize * sizeof(float));
            cudaMemcpy(dev_EnviMap, hst_scene->enviMap->image, enviSize * sizeof(float), cudaMemcpyHostToDevice);
        }

    }
    else {
        dev_EnviMap = nullptr;
    }
    

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_bloomImage);
    cudaFree(dev_bloomMask);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_bvhTree);
    cudaFree(dev_bvhGeoIdx);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_matIds);

    if (dev_EnviMap != nullptr) {
        cudaFree(dev_EnviMap);
        dev_EnviMap = nullptr;
    }

    checkCUDAError("pathtraceFree");
}


/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/

__device__ glm::vec2 uniformDiskConcentric(const glm::vec2& sample)
{

    float M_PI = 3.14159265359;

    float x = 2.0f * sample[0] - 1.0f;
    float y = 2.0f * sample[1] - 1.0f;

    float r = 1.f;
    float a = 0.f;

    if (x == 0 && y == 0) {
        return glm::vec2(0.f, 0.f);
    }
    if (abs(x) > abs(y)) {
        r *= x;
        a = (M_PI * 0.25) * (y / x);
    }
    else {
        r *= y;
        a = (M_PI * 0.5) - ((M_PI * 0.25) * (x / y));
    }

    return glm::vec2(cos(a) * r, sin(a) * r);

}

__device__ glm::vec2 sampleSphericalMap(glm::vec3 v) {
    v = glm::normalize(v);

    // Convert to spherical coordinates
    float theta = atan2f(v.x, v.z);
    float phi = acosf(glm::clamp(v.y, -1.0f, 1.0f));

    float u = (theta + PI) / (2.0f * PI);
    float v_coord = phi / PI;

    return glm::vec2(u, v_coord);
}

__device__ glm::vec3 sampleEnvironmentMap(const glm::vec3& direction, const float* envMap, const int width, const int height) {
    
    // Convert direction to UV
    glm::vec2 uv = sampleSphericalMap(direction);
    
    // Convert UV to pixel coordinates
    int x = (int)(uv.x * width) % width;
    int y = (int)(uv.y * height) % height;
    
    // Handle wrapping for x coordinate
    if (x < 0) x += width;
    if (y < 0) y = 0;
    if (y >= height) y = height - 1;
    
    // Calculate index (assuming RGB interleaved format)
    int index = (y * width + x) * 3;
    
    // Sample RGB values
    return glm::vec3(
        envMap[index],     // R
        envMap[index + 1], // G  
        envMap[index + 2]  // B
    );
}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, float focalDist, float appature)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, x * y, traceDepth);

    thrust::uniform_real_distribution<float> u01(-0.5, 0.5);

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
        
        // JITTER RAY FOR AA
        segment.ray.direction += cam.right * cam.pixelLength.x * u01(rng);
        segment.ray.direction += cam.up * cam.pixelLength.y * u01(rng);

        // DEPTH OF FIELD / THIN LENSE CAMERA
        float focalDistance = focalDist;
        float lenseRad = appature;
        thrust::uniform_real_distribution<float> u01(0.0, 1.0);

        glm::vec3 camForward = glm::normalize(cam.view - cam.position);

        float denom = glm::dot(segment.ray.direction, camForward);
        if (abs(denom) < 0.0001f) {
            return;
        }

        float t = focalDistance / denom;
        glm::vec3 pFocus_world = segment.ray.origin + t * segment.ray.direction;

        glm::vec2 lensUV = uniformDiskConcentric(glm::vec2(u01(rng), u01(rng)));
        lensUV *= lenseRad;

        segment.ray.origin = cam.position + cam.right * lensUV.x + cam.up * lensUV.y;
        segment.ray.direction = glm::normalize(pFocus_world - segment.ray.origin);


        // JITTER RAY FOR AA
        segment.ray.direction += cam.right * cam.pixelLength.x * u01(rng);
        segment.ray.direction += cam.up * cam.pixelLength.y * u01(rng);

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__global__ void fillMaterialId(int num_paths, int* dev_matIds, ShadeableIntersection* dev_intersections) {
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index < num_paths)
    {
        dev_matIds[path_index] = dev_intersections[path_index].materialId;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.


__device__ bool IntersectAABB(const Ray& ray, float t, const glm::vec3 bmin, const glm::vec3 bmax) {
    float tx1 = (bmin.x - ray.origin.x) / ray.direction.x, tx2 = (bmax.x - ray.origin.x) / ray.direction.x;
    float tmin = glm::min(tx1, tx2), tmax = glm::max(tx1, tx2);

    float ty1 = (bmin.y - ray.origin.y) / ray.direction.y, ty2 = (bmax.y - ray.origin.y) / ray.direction.y;
    tmin = glm::max(tmin, glm::min(ty1, ty2)), tmax = glm::min(tmax, glm::max(ty1, ty2));
    float tz1 = (bmin.z - ray.origin.z) / ray.direction.z, tz2 = (bmax.z - ray.origin.z) / ray.direction.z;
    tmin = glm::max(tmin, glm::min(tz1, tz2)), tmax = glm::min(tmax, glm::max(tz1, tz2));

    return tmax >= tmin && tmin < t && tmax > 0;
}

__device__ float computeBVHintersectDist(int numGeos, int numNodes, Ray ray, float& t, Geom* geoms, bvhNode* bvhTree, int* bvhGeoIdx, const int nodeIndex,
    glm::vec3& intersect_point, glm::vec3& normal, int& hit_geom_index) {


    int stack[32];
    int stackPtr = 0;
    stack[stackPtr++] = numNodes - 1;

    while (stackPtr > 0) {
        int nodeIndex = stack[--stackPtr];
        bvhNode curNode = bvhTree[nodeIndex];

        if (!IntersectAABB(ray, t, curNode.aabbMin, curNode.aabbMax)) {
            continue;
        }

        if (curNode.isLeaf == 1) {
            glm::vec3 tmp_intersect;
            glm::vec3 tmp_normal;
            bool outside = true;
            float tTest;

            for (int i = curNode.startIdx; i < curNode.endIdx; i++) {
                if (i < 0 || i >= numGeos) continue;
                int geoIdx = bvhGeoIdx[i];
                if (geoIdx < 0 || geoIdx >= numGeos) continue;

                Geom& geom = geoms[geoIdx];

                if (geom.type == SPHERE) {
                    tTest = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
                }
                else if (geom.type == CUBE) {
                    tTest = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
                }
                else if (geom.type == DISK) {
                    tTest = diskIntersectionTest(geom, ray, tmp_intersect, tmp_normal);
                }
                else if (geom.type == TRIANGLE) {
                    tTest = triangleIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
                }

                if (tTest > 0.0f && tTest < t) {
                    t = tTest;
                    hit_geom_index = geoIdx; 
                    intersect_point = tmp_intersect;
                    normal = tmp_normal;
                }
            }
        }
        else {
            if (curNode.rightChild >= 0) {
                stack[stackPtr++] = curNode.rightChild;
            }
            if (curNode.leftChild >= 0) {
                stack[stackPtr++] = curNode.leftChild;
            }
        }
    }

    return t;

}

__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    bvhNode* bvhTree,
    int* bvhGeoIdx,
    int geoms_size,
    int bvhTree_size,
    ShadeableIntersection* intersections)
{   

    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        // FOR GOING BACK TO NAIVE INTERSECTIONS SWITCH t AND t_min
        float t = FLT_MAX;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;


        //
        //for (int i = 0; i < geoms_size; i++)
        //{
        //    Geom& geom = geoms[i];
        //
        //    if (geom.type == CUBE)
        //    {
        //        t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        //    }
        //    else if (geom.type == SPHERE)
        //    {
        //        t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        //    }
        //    else if (geom.type == DISK) {
        //        t = diskIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal);
        //    }
        //    else if (geom.type == TRIANGLE) {
        //        t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        //    }
        //    // TODO: add more intersection tests here... triangle? metaball? CSG?
        //
        //    // Compute the minimum t from the intersection tests to determine what
        //    // scene geometry object was hit first.
        //    if (t > 0.0f && t_min > t)
        //    {
        //        t_min = t;
        //        hit_geom_index = i;
        //        intersect_point = tmp_intersect;
        //        normal = tmp_normal;
        //    }
        //}

        t_min = computeBVHintersectDist(geoms_size, bvhTree_size, pathSegment.ray, t, geoms, bvhTree, bvhGeoIdx, 0, intersect_point, normal, hit_geom_index);

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].geoId = hit_geom_index;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}


// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

__global__ void mirrorShader(int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        {
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 8 - pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                return;
            }
            else {

                pathSegments[idx].color *= material.color;

                if (pathSegments[idx].remainingBounces <= 0) {
                    pathSegments[idx].color = glm::vec3(0.0f);
                    return;
                }
                else {
                    glm::vec3 magic = getPointOnRay(pathSegments[idx].ray, intersection.t);
                    scatterRay(pathSegments[idx], magic, intersection.surfaceNormal, material, rng);
                }
            }
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }

}

__global__ void shadeRay(int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int depth,
    Geom* geoms,
    int geoms_size,
    float* enviMap, int enviWidth, int enviHeight) {

    float pi = 3.14159265359;
    float INV_PI = 1.f / pi;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_paths)
    {   
        if (pathSegments[idx].remainingBounces < 0) {
            return;
        }
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        {

            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
            //thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            if (intersection.materialId == 0) {
                Geom geo = geoms[intersection.geoId];
                glm::vec3 magic = getPointOnRay(pathSegments[idx].ray, intersection.t);
                blackHoleRay(pathSegments[idx], magic, intersection.surfaceNormal, geo.invTranspose, material, rng);

            }
            else if (material.emittance > 0.0f) {
                pathSegments[idx].color *= materialColor * material.emittance;
                pathSegments[idx].remainingBounces = -1;
            }
            else {
                pathSegments[idx].color *= materialColor;
                glm::vec3 magic = getPointOnRay(pathSegments[idx].ray, intersection.t);
                scatterRay(pathSegments[idx], magic, intersection.surfaceNormal, material, rng);

                /*Ray lightDir = Ray();
                lightDir.direction = -glm::normalize(glm::vec3(0.0, 10.0, 0.0) - magic);
                lightDir.origin = magic + intersection.surfaceNormal * 0.01f;
                float lightDist = glm::length(glm::vec3(0.0, 10.0, 0.0) - magic);
                float intersectDist = computeIntersectDist(lightDir, geoms, geoms_size);
                
                if (abs(lightDist - intersectDist) < 0.01) {
                    pathSegments[idx].color *= glm::vec3(1.0) * 5.f;
                }*/
                
            }
        }
        else {
            glm::vec3 enviColor = sampleEnvironmentMap(pathSegments[idx].ray.direction, enviMap, enviWidth, enviHeight);
            pathSegments[idx].color *= enviColor;
            pathSegments[idx].remainingBounces = -1;
        }
    }
}

// STREAM COMPACTION BOOL
struct IsAlive {
    __host__ __device__
        bool operator()(const PathSegment& pathSeg) {
        return pathSeg.remainingBounces >= 0;
    }
};

struct IsAliveZip {
    __host__ __device__
        bool operator()(const thrust::tuple<PathSegment, ShadeableIntersection>& t) {
        IsAlive isAlive;
        return isAlive(thrust::get<0>(t));
    }
};

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

__global__ void bloomHighPass(int nPaths, PathSegment* iterationPaths, glm::vec3* image, glm::vec3* bloomMask, glm::ivec2 resolution, int iter, float thresh) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 color = image[index];
        //scolor /= float(iter);
        
        float brightness = glm::dot(color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
        if (brightness > thresh) {
            bloomMask[index] = color;
        }
        else {
            bloomMask[index] = glm::vec3(0.0f);
        }
    }
}

__global__ void bloomBlurY(int nPaths, glm::vec3* ibloomMask, glm::vec3* oBloomMask, int imageWidth, int imageHeight) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    float weight[11] = {
     0.1247f,  // center
     0.1226f,  // ±1
     0.1167f,  // ±2
     0.1075f,  // ±3
     0.0957f,  // ±4
     0.0818f,  // ±5
     0.0672f,  // ±6
     0.0530f,  // ±7
     0.0399f,  // ±8
     0.0285f,  // ±9
     0.0193f   // ±10
    };

    if (x < imageWidth && y < imageHeight)
    {
        int index = x + (y * imageWidth);
        glm::vec3 result = ibloomMask[index] * weight[0];
        for (int i = 1; i < 11; ++i)
        {
            int posIdx = x + ((y + i) * imageWidth);
            int negIdx = x + ((y - i) * imageWidth);

            if ((y + i) < imageHeight) {
                result += ibloomMask[posIdx] * weight[i];
            }
            if ((y - i) >= 0) {
                result += ibloomMask[negIdx] * weight[i];
            }

        }
        
        oBloomMask[index] = result;
    }
}

__global__ void bloomBlurX(int nPaths, glm::vec3* ibloomMask, glm::vec3* oBloomMask, int imageWidth, int imageHeight) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    float weight[11] = {
     0.1247f,  // centers
     0.1226f,  // ±1
     0.1167f,  // ±2
     0.1075f,  // ±3
     0.0957f,  // ±4
     0.0818f,  // ±5
     0.0672f,  // ±6
     0.0530f,  // ±7
     0.0399f,  // ±8
     0.0285f,  // ±9
     0.0193f   // ±10
    };

    if (x < imageWidth && y < imageHeight)
    {
        int index = x + (y * imageWidth);
        glm::vec3 result = ibloomMask[index] * weight[0];
        for (int i = 1; i < 11; ++i)
        {
            int posIdx = (x + i) + (y * imageWidth);
            int negIdx = (x - i) + (y * imageWidth);

            if ((x + i) < imageWidth) {
                result += ibloomMask[posIdx] * weight[i];
            }
            if ((x - i) >= 0) {
                result += ibloomMask[negIdx] * weight[i];
            }
        }

        oBloomMask[index] = result;
    }
}

__global__ void bloomBlend(int nPaths, glm::vec3* bloomMask, glm::vec3* image, glm::vec3* dev_bloomImage, int imageWidth, int imageHeight) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * imageWidth);

    if (x < imageWidth && y < imageHeight)
    {
        const float gamma = 2.2;
        const float exposure = 0.7;
        glm::vec3 hdrColor = image[index];
        glm::vec3 bloomColor = bloomMask[index];
        hdrColor += bloomColor; 

        glm::vec3 result = glm::vec3(1.0f) - exp(-hdrColor * exposure);
        // also gamma correct while we're at it       
        result = pow(result, glm::vec3(1.0f / gamma));

        dev_bloomImage[index] = glm::vec3(hdrColor);
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, 36.0, 0.0);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    int og_num_paths = num_paths;

    // ENVI MAP VARIABLES:
    int hdrHeight = hst_scene->enviMap->height;
    int hdrWidth = hst_scene->enviMap->width;

    if (dev_bvhTree == NULL) {
        printf("ERROR: dev_bvhTree is NULL!\n");
    }


    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            dev_bvhTree,
            dev_bvhGeoIdx,
            hst_scene->geoms.size(),
            hst_scene->bvhTree.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        //shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
        //    iter,
        //    num_paths,
        //    dev_intersections,
        //    dev_paths,
        //    dev_materials
        //);
        //iterationComplete = true; // TODO: should be based off stream compaction results.
        

        // SORT RAYS BY MATERIAL ID
        //fillMaterialId << <numblocksPathSegmentTracing, blockSize1d >> > (
        //    num_paths, dev_matIds, dev_intersections
        //);

        //dev_thrust_matId = thrust::device_pointer_cast(dev_matIds);
        //dev_thrust_pathIdx = thrust::device_pointer_cast(dev_paths);
        //dev_thrust_intersections = thrust::device_pointer_cast(dev_intersections);

        //thrust::sort_by_key(
        //    dev_thrust_matId,
        //    dev_thrust_matId + num_paths,
        //    thrust::make_zip_iterator(thrust::make_tuple(
        //        dev_thrust_pathIdx,
        //        dev_thrust_intersections
        //    ))
        //);
        
        shadeRay << < numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            depth,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_EnviMap, hdrWidth, hdrHeight
        );

        // STREAM COMPACTION
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections));
        auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(dev_paths + num_paths, dev_intersections + num_paths));
        auto partition_point = thrust::partition(thrust::device,
            zip_begin, zip_end,
            IsAliveZip{});
        num_paths = partition_point - zip_begin;

        if (depth > traceDepth || num_paths <= 0) {
            iterationComplete = true; // TODO: should be based off stream compaction results.
        }

        depth++;
       
        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////
    
    // BLOOM POST PROCESS

    cudaMemset(dev_bloomMask, 0, pixelcount * sizeof(glm::vec3));
    bloomHighPass << <blocksPerGrid2d, blockSize2d >> > (pixelcount, dev_paths, dev_image, dev_bloomMask,  cam.resolution, iter, 1.0f);
    bloomBlurY << <blocksPerGrid2d, blockSize2d >> > (pixelcount, dev_bloomMask, dev_bloomMaskBlur, cam.resolution.x, cam.resolution.y);
    bloomBlurX << <blocksPerGrid2d, blockSize2d >> > (pixelcount, dev_bloomMaskBlur, dev_bloomMask, cam.resolution.x, cam.resolution.y);
    bloomBlend << <blocksPerGrid2d, blockSize2d >> > (pixelcount, dev_bloomMask, dev_image, dev_bloomImage, cam.resolution.x, cam.resolution.y);
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_bloomImage);

    // Send results to OpenGL buffer for rendering
    //sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_bloomImage, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
