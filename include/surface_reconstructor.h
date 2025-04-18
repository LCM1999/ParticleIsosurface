#pragma once
#ifndef SURF_CONSTRUCT_H
#define SURF_CONSTRUCT_H

#include <vector>
#include <string.h>
#include <atomic>
#include <memory>

#include "timer.h"
#include "utils.h"
#include "iso_common.h"

// included packages for cornerstone octree 
#include <box.h>
#include <morton.h>
#include <coord_struct.h>
#include <calculator.h>
#include <utils_helper.h>
#include <octree_func.h>
#include <bitset>
#include <algorithm>
#include <cuda_def.h>
#include "evaluator.h"
#include "multi_level_searcher_gpu.cuh"
#include "hash_grid.h"
#include "multi_level_searcher.h"

class Mesh;

struct TNode;

class SurfReconstructor
{
private:
    // Global Parameters
    int _OVERSAMPLE_QEF = 2;
    float _BORDER = 0.0;//(1.0 / 16.0);
    int _DEPTH_MAX = 8; // 7
    int _DEPTH_MIN = 6; // 4

    bool useCPU;
    // std::shared_ptr<HashGrid> _hashgrid;
    std::shared_ptr<MultiLevelSearcher> _searcherCPU;
    std::shared_ptr<MultiLevelSearcherGPU> _searcherGPU;
    std::shared_ptr<Evaluator> _evaluator;

    std::vector<Vec3f> _GlobalParticles;
    std::vector<float> _GlobalRadiuses;
    float _RADIUS = 0;
    int _GlobalParticlesNum = 0;

    int _STATE = 0;

    static const int inProcessSize = 10000000; //

    float _BoundingBox[6] = {0.0f};
    float _RootHalfLength;
    float _RootCenter[3] = {0.0f};

    std::shared_ptr<TNode> _OurRoot;
	Mesh* _OurMesh;

    std::vector<std::shared_ptr<TNode>*> WaitingStack;
    std::vector<std::shared_ptr<TNode>*> ProcessArray;

    int queue_flag;
    
    // variables for cornerstone octree methods
    std::vector<Vec3f> _particles;
    std::vector<uint64_t> _mortonCodes;
    std::vector<uint64_t> _tree;
    std::vector<uint64_t> _counts;
    cstoneOctree::Box _box;
    cstoneOctree::OctreeNs _octreeNs;

protected:
    void loadRootBox();

    void shrinkBox();

    void resizeRootBoxConstR();

    void resizeRootBoxVarR();

    void genIsoOurs();
    void checkEmptyAndCalcCurv(std::shared_ptr<TNode> tnode, unsigned char& empty, float& curv, float& min_radius);
    void beforeSampleEval(std::shared_ptr<TNode> tnode, float& curv, float& min_radius, unsigned char& empty);
    void afterSampleEval(
        std::shared_ptr<TNode> tnode, float& curv, float& min_radius, float* sample_points, float* sample_grads);

public:
    SurfReconstructor() {}
    SurfReconstructor(
        std::vector<cstoneOctree::Vec3f>& particles,
        std::vector<float>& radiuses, 
        Mesh* mesh, 
        float radius);

    ~SurfReconstructor() {}

    void Run(float iso_factor, float smooth_factor);
    void RunCPU(float iso_factor, float smooth_factor);
    void RunCPU2(float iso_factor, float smooth_factor);
    void RunGPU(float iso_factor, float smooth_factor);

    inline int getOverSampleQEF() {return _OVERSAMPLE_QEF;}
    inline float getBorder() {return _BORDER;}
    inline int getDepthMax() {return _DEPTH_MAX;}
    inline int getDepthMin() {return _DEPTH_MIN;}
    // inline std::shared_ptr<HashGrid> getHashGrid() {return _hashgrid;}
    inline bool getUseCPU() {return useCPU;}
    inline std::shared_ptr<MultiLevelSearcher> getSearcherCPU() {return _searcherCPU;}
    inline std::shared_ptr<MultiLevelSearcherGPU> getSearcherGPU() {return _searcherGPU;}
    inline std::shared_ptr<Evaluator> getEvaluator() {return _evaluator;}
    inline std::vector<cstoneOctree::Vec3f>* getGlobalParticles() {return &_GlobalParticles;}
    inline int getGlobalParticlesNum() {return _GlobalParticlesNum;}
    inline float getConstRadius() {return _RADIUS;}
    inline int getSTATE() {return _STATE;}
    inline std::shared_ptr<TNode> getRoot() {return _OurRoot;}

};

__global__ void estimateTotalInfluenceParticlesKernel(uint64_t* d_iso_tree, int d_iso_tree_size,
                                                        Vec3f* d_iso_centers, Vec3f* d_iso_sizes,
                                                    HashGridGPU** d_searchers, int d_searchers_size);

__global__ void calculateSplitsKernel(uint64_t* d_iso_tree, int d_iso_tree_size,
                                        Vec3f* d_iso_centers, Vec3f* d_iso_sizes,
                                        HashGridGPU** d_searchers, int d_searchers_size,
                                        int* d_estimateNeighborsNums, int* d_estimateNeighborsNumsLayout,
                                        int* d_totalInsideParticlesIdx,
                                        int* d_iso_nodeOps);

#endif
