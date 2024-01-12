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

class HashGrid;
class MultiLevelSearcher;
class Evaluator;
class Mesh;

struct TNode;

class SurfReconstructor
{
private:
    // Global Parameters
    int _OVERSAMPLE_QEF = 2;
    float _BORDER = 0.0;//(1.0 / 16.0);
    int _DEPTH_MAX = 6; // 7
    int _DEPTH_MIN = 5; // 4
    float _NEIGHBOR_FACTOR = 4.0;
    float _SMOOTH_FACTOR = 2.0;
    float _ISO_FACTOR = 1.9;
    float _ISO_VALUE = 0.0;

    float _MAX_SCALAR = -1.0;
    float _MIN_SCALAR = 0.0;
    float _BAD_QEF = 0.0001;
    float _FLATNESS = 0.99;
    int _MIN_NEIGHBORS_NUM = 25;
    int _SPARSE_NEIGHBORS_NUM = 16;

    bool _USE_XMEAN = true;
    float _XMEAN_DELTA = 0.0;

    float _TOLERANCE = 1e-8;
    float _RATIO_TOLERANCE = 0.1;
    //float _LOW_MESH_QUALITY = 0.17320508075688772935274463415059;

    std::shared_ptr<HashGrid> _hashgrid;
    std::shared_ptr<MultiLevelSearcher> _searcher;
    std::shared_ptr<Evaluator> _evaluator;

    std::vector<Eigen::Vector3f> _GlobalParticles;
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
protected:
    void loadRootBox();

    void generalModeRun();

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
        std::vector<Eigen::Vector3f>& particles,
        std::vector<float>& radiuses, 
        Mesh* mesh, 
        float radius, 
        float iso_factor,
        float inf_factor);

    ~SurfReconstructor() {}

    void Run();

    inline int getOverSampleQEF() {return _OVERSAMPLE_QEF;}
    inline float getBorder() {return _BORDER;}
    inline int getDepthMax() {return _DEPTH_MAX;}
    inline int getDepthMin() {return _DEPTH_MIN;}
    inline float getNeighborFactor() {return _NEIGHBOR_FACTOR;}
    inline float getSmoothFactor() {return _SMOOTH_FACTOR;}
    inline void setSmoothFactor(float smooth_factor) {_SMOOTH_FACTOR = smooth_factor;}
    inline float getIsoFactor() {return _ISO_FACTOR;}
    inline void setIsoFactor(float iso_factor) {_ISO_FACTOR = iso_factor;}
    inline float getIsoValue() {return _ISO_VALUE;}
    inline float getMaxScalar() {return _MAX_SCALAR;}
    inline float getMinScalar() {return _MIN_SCALAR;}
    inline float getBadQef() {return _BAD_QEF;}
    inline float getFlatness() {return _FLATNESS;}
    inline int getMinNeighborsNum() {return _MIN_NEIGHBORS_NUM;}
    inline int getSparseNeighborsNum() {return _SPARSE_NEIGHBORS_NUM;};
    inline bool getUseXMean() {return _USE_XMEAN;}
    inline float getXMeanDelta() {return _XMEAN_DELTA;}
    inline float getTolerance() {return _TOLERANCE;}
    inline float getRatioTolerance() {return _RATIO_TOLERANCE;}
    inline std::shared_ptr<HashGrid> getHashGrid() {return _hashgrid;}
    inline std::shared_ptr<MultiLevelSearcher> getSearcher() {return _searcher;}
    inline std::shared_ptr<Evaluator> getEvaluator() {return _evaluator;}
    inline std::vector<Eigen::Vector3f>* getGlobalParticles() {return &_GlobalParticles;}
    inline int getGlobalParticlesNum() {return _GlobalParticlesNum;}
    inline float getConstRadius() {return _RADIUS;}
    inline int getSTATE() {return _STATE;}
    inline std::shared_ptr<TNode> getRoot() {return _OurRoot;}
};

#endif
