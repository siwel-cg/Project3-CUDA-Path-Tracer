#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadEnvironmentMap(const std::string& hdrName)
{
    int width, height, channels;
    float* data = stbi_loadf(hdrName.c_str(), &width, &height, &channels, 3);

    enviMap = new EnvironmentMap();
    enviMap->width = width;
    enviMap->height = height;
    enviMap->channels = channels;
    enviMap->image = data;
} 

//bvhNode Scene::buildTree(glm::vec3 min, glm::vec3 max, int idxStart, int idxEnd , int leafSize) {
//    bvhNode curNode;
//    curNode.aabbMax = max;
//    curNode.aabbMin = min;
//    curNode.startIdx = idxStart;
//    curNode.endIdx = idxEnd;
//    curNode.isLeaf = false;
//
//    // BUILD BOUNDING BOX
//    for (int i = idxStart; i < idxEnd; i++) {
//        Geom geo = geoms[bvhGeoIdx[i]];
//        curNode.aabbMax.x = glm::max(curNode.aabbMax.x, geo.translation.x);
//        curNode.aabbMax.y = glm::max(curNode.aabbMax.y, geo.translation.y);
//        curNode.aabbMax.z = glm::max(curNode.aabbMax.z, geo.translation.z);
//
//        curNode.aabbMin.x = glm::min(curNode.aabbMin.x, geo.translation.x);
//        curNode.aabbMin.y = glm::min(curNode.aabbMin.y, geo.translation.y);
//        curNode.aabbMin.z = glm::min(curNode.aabbMin.z, geo.translation.z);
//    }
//    
//    if (curNode.endIdx - curNode.startIdx <= leafSize) { // STOP SUBDIV
//        curNode.isLeaf = true;
//        curNode.leftChild = nullptr;
//        curNode.rightChild = nullptr;
//        bvhTree.push_back(curNode);
//        return curNode;
//    }
//    else {
//        float centerPos = 0.0;
//        printf("Max = [%f, %f, %f]\n", curNode.aabbMax.x, curNode.aabbMax.y, curNode.aabbMax.z);
//        printf("Min = [%f, %f, %f]\n", curNode.aabbMin.x, curNode.aabbMin.y, curNode.aabbMin.z);
//        glm::vec3 ext = curNode.aabbMax - curNode.aabbMin;
//        int maxis = 0;
//        if (ext.y > ext.x) maxis = 1;
//        if (ext.z > ext[maxis]) maxis = 2;
//        centerPos = 0.5f * (curNode.aabbMax[maxis] + curNode.aabbMin[maxis]);
//
//        // Partition current region and recurse
//        std::vector<int> firstHalf = {};
//        std::vector<int> lastHalf = {};
//        for (int i = idxStart; i < idxEnd; i++) {
//            if (geoms[bvhGeoIdx.at(i)].translation[maxis] < centerPos) {
//                firstHalf.push_back(bvhGeoIdx.at(i));
//            }
//            else {
//                lastHalf.push_back(bvhGeoIdx.at(i));
//            }
//        }
//
//
//        printf("First Half: ");
//        for (int i = 0; i < firstHalf.size(); i++) {
//            printf("%d, ", firstHalf.at(i));
//        }
//        printf("\n");
//        printf("Last Half: ");
//        for (int i = 0; i < lastHalf.size(); i++) {
//            printf("%d, ", lastHalf.at(i));
//        }
//        printf("\n");
//
//        ///////////
//        int midIdx = idxStart + firstHalf.size();
//        firstHalf.insert(firstHalf.end(), lastHalf.begin(), lastHalf.end());
//        //////////////
//
//        printf("New First Half: ");
//        for (int i = 0; i < firstHalf.size(); i++) {
//            printf("%d, ", firstHalf.at(i));
//        }
//        printf("\n");
//
//        //////////
//        for (int i = idxStart; i <= idxEnd; i++) {
//            bvhGeoIdx.at(i) = firstHalf.at(i - idxStart);
//        }
//        /////////////////
//
//        printf("Size = %d, front = %d, last = %d \n", bvhGeoIdx.size(), firstHalf.size(), lastHalf.size());
//        for (int i = 0; i < bvhGeoIdx.size(); i++) {
//            printf("%d, ", bvhGeoIdx.at(i));
//        }
//        printf("\n");
//        printf("mid id = %d \n", midIdx);
//
//        glm::vec3 centerUpBound = curNode.aabbMax;
//        centerUpBound[maxis] = centerPos;
//        curNode.leftChild = &buildTree(curNode.aabbMin, centerUpBound, idxStart, midIdx-1, leafSize);
//
//        glm::vec3 centerLowBound = curNode.aabbMin;
//        centerLowBound[maxis] = centerPos;
//        curNode.rightChild = &buildTree(centerLowBound, curNode.aabbMax, midIdx, idxEnd, leafSize); 
//    }
//    
//    bvhTree.push_back(curNode);
//    return curNode;
//}

bvhNode Scene::buildTree(glm::vec3 min, glm::vec3 max, int idxStart, int idxEnd, int leafSize) {
    bvhNode curNode;
    curNode.aabbMin = glm::vec3(+FLT_MAX);
    curNode.aabbMax = glm::vec3(-FLT_MAX);
    curNode.startIdx = idxStart;
    curNode.endIdx = idxEnd;
    
    std::vector<glm::vec3> centroids = std::vector<glm::vec3>(geoms.size(), glm::vec3());

    // CALCULATE BOUNDS
    for (int i = idxStart; i < idxEnd; i++) {
        Geom curGeo = geoms[bvhGeoIdx[i]];
        glm::vec3 translate = curGeo.translation;
        glm::vec3 scale = curGeo.scale;

        curNode.aabbMax.x = glm::max(curNode.aabbMax.x, translate.x + scale.x);
        curNode.aabbMax.y = glm::max(curNode.aabbMax.y, translate.y + scale.y);
        curNode.aabbMax.z = glm::max(curNode.aabbMax.z, translate.z + scale.z);
        
        curNode.aabbMin.x = glm::min(curNode.aabbMin.x, translate.x - scale.x);
        curNode.aabbMin.y = glm::min(curNode.aabbMin.y, translate.y - scale.y);
        curNode.aabbMin.z = glm::min(curNode.aabbMin.z, translate.z - scale.z);

        centroids[bvhGeoIdx[i]] = curGeo.translation;
    }

    if (curNode.endIdx - curNode.startIdx <= leafSize) { // STOP SUBDIV
        curNode.isLeaf = true;
        curNode.leftChild = -1;
        curNode.rightChild = -1;
        return curNode;
    }
    else {
        // CALCULATE MID POINT
        glm::vec3 ext = curNode.aabbMax - curNode.aabbMin;
        int maxis = 0;
        if (ext.y > ext.x) maxis = 1;
        if (ext.z > ext[maxis]) maxis = 2;

        float midPos = (curNode.aabbMax[maxis] + curNode.aabbMin[maxis]) * 0.5;

        auto first = bvhGeoIdx.begin() + idxStart;
        auto last = bvhGeoIdx.begin() + idxEnd;

        auto midIt = std::partition(first, last, [&](int primId) {
            float c = centroids[primId][maxis];
            return c < midPos;
            });
        int mid = int(midIt - bvhGeoIdx.begin());

        if (mid == idxStart || mid == idxEnd) {
            mid = (idxStart + idxEnd) / 2;
            auto first = bvhGeoIdx.begin() + idxStart;
            auto nth = bvhGeoIdx.begin() + mid;
            auto last = bvhGeoIdx.begin() + idxEnd;

            std::nth_element(first, nth, last, [&](int ia, int ib) {
                const Geom& A = geoms[ia];
                const Geom& B = geoms[ib];
                float ca = A.translation[maxis];
                float cb = B.translation[maxis];
                return ca < cb;
               });
        }

        for (int i = 0; i < bvhGeoIdx.size(); i++) {
            printf("%d, ", bvhGeoIdx.at(i));
        }
        printf("\n");

        bvhNode leftChild = buildTree(curNode.aabbMin, curNode.aabbMax, idxStart, mid - 1, leafSize);
        bvhNode rightChild = buildTree(curNode.aabbMin, curNode.aabbMax, mid, idxEnd, leafSize);

        curNode.leftChild = bvhTree.size();
        bvhTree.push_back(leftChild);

        curNode.rightChild = bvhTree.size(); 
        bvhTree.push_back(rightChild);
    }
    
    return curNode;
}

void Scene::loadBVH() {
    this->bvhTree = {};
    if (geoms.size() == 0) {
        return;
    }

    bvhNode root = buildTree(glm::vec3(1e30f), glm::vec3(-1e30f), 0, bvhGeoIdx.size()-1, 4);
    bvhTree.push_back(root);
    this->root = root;
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            const auto& spec = p["ROUGHNESS"];
            newMaterial.specular.exponent = spec;
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];

            const auto& spec = p["ROUGHNESS"];
            newMaterial.specular.exponent = spec;
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            const auto& spec = p["ROUGHNESS"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.exponent = spec;
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "BlackHole")
        {
            const auto& innerRad = p["INNERRAD"];
            const auto& outerRad = p["OUTERRAD"];
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.blackHole.iRad = innerRad;
            newMaterial.blackHole.oRad = outerRad;
            newMaterial.emittance = p["EMITTANCE"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }

    // READ GEOMETRY
    const auto& objectsData = data["Objects"];
    bvhGeoIdx = {};
    int idx = 0;
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else {
            newGeom.type = DISK;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        bvhGeoIdx.push_back(idx);
        idx++;
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
