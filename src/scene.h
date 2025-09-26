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

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    EnvironmentMap* enviMap;
};
