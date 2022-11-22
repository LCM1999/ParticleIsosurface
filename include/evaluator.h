#pragma once

#include "iso_common.h"
#include "hash_grid.h"
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
	std::vector<Eigen::Vector3f>* GlobalPoses;
	std::vector<float>* GlobalDensity;
	std::vector<float>* GlobalMass;
    std::vector<int> GlobalSplash;
    std::map<int, Eigen::Vector3f> SurfaceNormals;
    std::vector<int> SurfaceParticles;
	Eigen::Vector3f* GlobalxMeans;
    Eigen::Matrix3f* GlobalGs;

	Evaluator(SurfReconstructor* surf_constructor,
		std::vector<Eigen::Vector3f>* global_particles, std::vector<float>* global_density, std::vector<float>* global_mass);

	void SingleEval(const Eigen::Vector3f& pos, float& scalar, Eigen::Vector3f& gradient, bool use_normalize = true, bool use_signed = true, bool grad_normalize = true);

    void GridEval(
        std::vector<Eigen::Vector3f>& sample_points, std::vector<float>& field_scalars, std::vector<Eigen::Vector3f>& field_gradients,
        bool& signchange, int oversample);

    bool CheckSplash(const int& pIdx);
    float RecommendIsoValue();
    float RecommendSurfaceThreshold();
    void GetSurfaceParticles();
    float CurvEval(std::vector<int>& p_list);
private:
    float sample_step;
    double influnce2;

    float general_kernel(double d2, double h2);
    float spiky_kernel(double d, double h);
    float viscosity_kernel(double d, double h);

	float IsotrpicInterpolate(const int pIdx, const float d);
	float AnisotropicInterpolate(const int pIdx, const Eigen::Vector3f& diff);
	void compute_Gs_xMeans();
	double wij(double d, double r);

    void IsotropicEval(const Eigen::Vector3f& pos, float& info, float* temp_scalars);
    void AnisotropicEval(const Eigen::Vector3f& pos, float& info, float* temp_scalars);
};
