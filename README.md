CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Lewis Ghrist
* [Personal Website](https://siwel-cg.github.io/siwel.cg_websiteV1/index.html#home), [LinkedIn](https://www.linkedin.com/in/lewis-ghrist-4b1b3728b/)
* Tested on: Windows 11, AMD Ryzen 9 5950X 16-Core Processor, 64GB, NVIDIA GeForce RTX 3080 10GB

### Renders (MORE TO COME)
<p align="center">
  <img src="IMAGES/blackhole_bvh_test.2025-10-05_02-39-31z.775samp.png" alt="Purple accretion disk with lensing" width="49%"/>
  <img src="IMAGES/blackhole_bvh_test.2025-10-05_02-25-49z.537samp.png" alt="Blue accretion disk variant" width="49%"/>
</p>
<p align="center">
  <img src="IMAGES/blackhole_mirrors.2025-10-05_13-27-25z.678samp.png" alt="Orange disk with mirrored highlights" width="49%"/>
  <img src="IMAGES/singleBH_V1.2025-10-05_17-56-29z.437samp.png" alt="Hot swirling disk, planets in a row" width="49%"/>
</p>
<p align="center">
  <img src="IMAGES/singleBH_V1.2025-10-05_17-53-17z.661samp.png" alt="Pastel disk, planets in a row" width="49%"/>
  <img src="IMAGES/singleBH_V1.2025-10-05_16-53-14z.97samp.png" alt="Magenta close-up with hand silhouette" width="49%"/>
</p>
<p align="center">
  <img src= "IMAGES/singleBH_V1.2025-10-04_03-03-55z.950samp.png" alt="Clean pink ring and lensing" width="60%"/>
</p>

### Implemented Features
# Core Features:
- Diffuse and Mirror BSDF shading with stochatic blending based on roughness
- Stream compaction for culling finished paths
- Material sorting 
- Stochastic AA sampling
# Custom Features:
- Cutom OBJ mesh loading
- BVH spatial data structure
- Thin lense depth of field
- Environment mapping
- Physically acuarate light bending black hole with noise based acretion disk
- Bloom post processing

### References
- https://henrikdahlberg.github.io/2016/08/23/stream-compaction.html
- https://nvidia.github.io/cccl/thrust/api/group__stream__compaction_1gaf01d45b30fecba794afae065d625f94f.html
- https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
- http://spiro.fisica.unipd.it/~antonell/schwarzschild/
- https://rantonels.github.io/starless/
- https://web.mit.edu/10.001/Web/Course_Notes/Differential_Equations_Notes/node5.html
- https://blog.seanholloway.com/2022/03/13/visualizing-black-holes-with-general-relativistic-ray-tracing/
- https://learnopengl.com/Advanced-Lighting/Bloom
- https://github.com/tinyobjloader/tinyobjloader/tree/release