#pragma once

#include "iso_common.h"
#include "hash_grid.h"
#include "multi_level_researcher.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/SVD>

class SurfReconstructor;

class Evaluator
{
public:
	Evaluator();
	~Evaluator();

    SurfReconstructor* constructor;
	std::vector<Eigen::Vector3d>* GlobalPoses;
    std::vector<double>* GlobalRadius;
    double _MAX_RADIUS, _MIN_RADIUS;

    double Radius = 0;
    std::vector<std::uint8_t> GlobalSplash;
    std::vector<Eigen::Vector3d> PariclesNormals;
	Eigen::Vector3d* GlobalxMeans;
    Eigen::Matrix3d* GlobalGs;

	Evaluator(  SurfReconstructor* surf_constructor, std::vector<Eigen::Vector3d>* global_particles, 
                std::vector<double>* radiuses,
                double radius);

	void SingleEval(const Eigen::Vector3d& pos, double& scalar);
    void SingleEvalWithGrad(const Eigen::Vector3d& pos, double& scalar, Eigen::Vector3d& gradient, bool use_normalize = true, bool use_signed = true, bool grad_normalize = true);
    void GridEval(
        double* sample_points, double* field_gradients, double cellsize, 
        bool& signchange, int oversample, bool grad_normalize = true);

    bool CheckSplash(const int& pIdx);
    double CalculateMaxScalarConstR();
    double CalculateMaxScalarVarR();
    double RecommendIsoValueConstR(const double iso_factor);
    double RecommendIsoValueVarR(const double iso_factor);
    void CalcParticlesNormal();
private:
    const double sqrt3 = 1.7320508075688772935274463415059;
    const double sqrt11 = 3.3166247903553998491149327366707;
    const double inv_pi = 0.31830988618379067153776752674503;

    double neighbor_factor;
    double smooth_factor;

    double general_kernel(double d2, double h2, double h);
    double spiky_kernel(double d, double h);
    double viscosity_kernel(double d, double h);

	double IsotropicInterpolate(const int pIdx, const double d);
	double AnisotropicInterpolate(const int pIdx, const Eigen::Vector3d& diff);
	void compute_Gs_xMeans();
    void compute_single_xMean(Eigen::Vector3d p, Eigen::Vector3d neighbor, Eigen::Vector3d &xMean, double r);
    void compute_xMeans(int pIdx, std::vector<int> temp_neighbors, std::vector<int> &neighbors, int &closer_neighbor, Eigen::Vector3d &xMean);
    void compute_single_G(Eigen::Vector3d p, Eigen::Vector3d pm, Eigen::Vector3d n, Eigen::Matrix3d &G, double r);
    void compute_G(Eigen::Vector3d p, Eigen::Vector3d xMean, std::vector<int> neighbors, Eigen::Matrix3d &G);
	double wij(double d, double r);

    void IsotropicEval(const Eigen::Vector3d& pos, double& info, double* temp_scalars, double& sample_radius);
    void AnisotropicEval(const Eigen::Vector3d& pos, double& info, double* temp_scalars, double& sample_radius);
};
