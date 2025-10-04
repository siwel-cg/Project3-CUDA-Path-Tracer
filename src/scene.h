#pragma once

#include "sceneStructs.h"
#include <vector>

//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);
    void loadEnvironmentMap(const std::string& hdrName);
    void loadBVH();
    bvhNode buildTree(int idxStart, int idxEndl, int leafSize);

    std::vector<Geom> geoms;
    std::vector<glm::vec3> centroids;
    std::vector<Material> materials;
    RenderState state;
    EnvironmentMap* enviMap;

    // BVH STRUCTURE
    std::vector<bvhNode> bvhTree;
    std::vector<int> bvhGeoIdx;
    bvhNode root;
};
