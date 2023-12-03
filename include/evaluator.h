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
    std::vector<double>* GlobalRadius2;
    std::vector<double>* GlobalRadius3;
    std::vector<double>* GlobalInflunce2;
    std::vector<double>* GlobalSigma;

    double Radius = 0;
    double Radius2 = 0;
    double Radius3 = 0;
    double Influnce2 = 0;
    double Sigma = 0;
    std::vector<std::uint8_t> GlobalSplash;
    std::vector<Eigen::Vector3d> PariclesNormals;
	Eigen::Vector3d* GlobalxMeans;
    Eigen::Matrix3d* GlobalGs;
    std::vector<double> GlobalDeterminant;

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

    double NeighborFactor;
    double SmoothFactor;

    double general_kernel(double d2, double h2, double sigma);

	double IsotropicInterpolate(const int pIdx, const double d);
	double AnisotropicInterpolate(const int pIdx, const Eigen::Vector3d& diff);
	void compute_Gs_xMeans();
    void compute_xMeans(int pIdx, std::vector<int> temp_neighbors, std::vector<int> &neighbors, int &closer_neighbor, Eigen::Vector3d &xMean);
    void compute_G(Eigen::Vector3d p, Eigen::Vector3d xMean, std::vector<int> neighbors, Eigen::Matrix3d &G);
	double wij(double d, double r);

    void IsotropicEval(const Eigen::Vector3d& pos, double& info, double* temp_scalars, double& sample_radius);
    void AnisotropicEval(const Eigen::Vector3d& pos, double& info, double* temp_scalars, double& sample_radius);
};
