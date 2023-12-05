#pragma once
#ifndef SURF_CONSTRUCT_H
#define SURF_CONSTRUCT_H

#include <vector>
#include <string.h>
#include <queue>
#include <atomic>
#include "timer.h"
#include "utils.h"

class HashGrid;
class MultiLevelSearcher;
class Evaluator;
class TNode;
class Mesh;

class SurfReconstructor
{
private:
    // Global Parameters
    int _OVERSAMPLE_QEF = 2;
    double _BORDER = (1.0 / 16.0);
    int _DEPTH_MAX = 6; // 7
    int _DEPTH_MIN = 5; // 4
    double _NEIGHBOR_FACTOR = 4.0;
    double _SMOOTH_FACTOR = 2.0;
    double _ISO_FACTOR = 1.9;
    double _ISO_VALUE = 0.0;

    double _MAX_SCALAR = -1.0;
    double _MIN_SCALAR = 0.0;
    double _BAD_QEF = 0.0001;
    double _FLATNESS = 0.99;
    int _MIN_NEIGHBORS_NUM = 25;
    int _SPARSE_NEIGHBORS_NUM = 16;

    bool _USE_XMEAN = true;
    double _XMEAN_DELTA = 0.0;

    double _TOLERANCE = 1e-8;
    double _RATIO_TOLERANCE = 0.1;
    //double _LOW_MESH_QUALITY = 0.17320508075688772935274463415059;

    HashGrid* _hashgrid = NULL;
    MultiLevelSearcher* _searcher = NULL;
    Evaluator* _evaluator = NULL;

    std::vector<Eigen::Vector3d> _GlobalParticles;
    std::vector<double>* _GlobalRadiuses;
    
    double _RADIUS = 0;

    int _GlobalParticlesNum = 0;
    int _STATE = 0;

    static const int inProcessSize = 10000000; //

    double _BoundingBox[6] = {0.0f};
    double _RootHalfLength;
    double _RootCenter[3] = {0.0f};

    TNode* _OurRoot;
	Mesh* _OurMesh;

    std::vector<TNode*> WaitingStack;
    TNode* ProcessArray[inProcessSize];

    int queue_flag;
protected:
    void loadRootBox();

    void generalModeRun();

    void shrinkBox();

    void resizeRootBoxConstR();

    void resizeRootBoxVarR();

    // Method for CSV mode
    void genIsoOurs();
    void checkEmptyAndCalcCurv(TNode* tnode, bool& empty, double& curv, double& min_radius);
    void beforeSampleEval(TNode* tnode, double& curv, double& min_radius, bool& empty);
    void afterSampleEval(
        TNode* tnode, double& curv, double& min_radius, double* sample_points, double* sample_grads);

public:
    SurfReconstructor() {};
    SurfReconstructor(
        std::vector<Eigen::Vector3d>& particles,
        std::vector<double>* radiuses, 
        Mesh* mesh, 
        double radius, 
        double iso_factor,
        double inf_factor);

    ~SurfReconstructor() {};

    void Run();

    inline int getOverSampleQEF() {return _OVERSAMPLE_QEF;}
    inline double getBorder() {return _BORDER;}
    inline int getDepthMax() {return _DEPTH_MAX;}
    inline int getDepthMin() {return _DEPTH_MIN;}
    inline double getNeighborFactor() {return _NEIGHBOR_FACTOR;}
    inline double getSmoothFactor() {return _SMOOTH_FACTOR;}
    inline double getIsoValue() {return _ISO_VALUE;}
    inline double getMaxScalar() {return _MAX_SCALAR;}
    inline double getMinScalar() {return _MIN_SCALAR;}
    inline double getBadQef() {return _BAD_QEF;}
    inline double getFlatness() {return _FLATNESS;}
    inline int getMinNeighborsNum() {return _MIN_NEIGHBORS_NUM;}
    inline int getSparseNeighborsNum() {return _SPARSE_NEIGHBORS_NUM;};
    inline bool getUseXMean() {return _USE_XMEAN;}
    inline double getXMeanDelta() {return _XMEAN_DELTA;}
    inline double getTolerance() {return _TOLERANCE;}
    inline double getRatioTolerance() {return _RATIO_TOLERANCE;}
    inline HashGrid* getHashGrid() {return _hashgrid;}
    inline MultiLevelSearcher* getSearcher() {return _searcher;}
    inline Evaluator* getEvaluator() {return _evaluator;}
    inline std::vector<Eigen::Vector3d>* getGlobalParticles() {return &_GlobalParticles;}
    inline int getGlobalParticlesNum() {return _GlobalParticlesNum;}
    inline std::vector<double>* getRadiuses() {return _GlobalRadiuses;}
    inline double getConstRadius() {return _RADIUS;}
    inline int getSTATE() {return _STATE;}
    inline TNode* getRoot() {return _OurRoot;}
};

#endif
