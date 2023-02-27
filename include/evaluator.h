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
	std::vector<Eigen::Vector3f>* GlobalPoses;
	std::vector<float>* GlobalDensity;
	std::vector<float>* GlobalMass;
    std::vector<float>* GlobalRadius;
    float _MAX_DENSITY, _MIN_DENSITY;
    float _MAX_MASS, _MIN_MASS;
    float _MAX_RADIUS, _MIN_RADIUS;

    float Density = 0;
    float Mass = 0;
    float Radius = 0;
    std::vector<char> GlobalSplash;
    std::vector<Eigen::Vector3f> PariclesNormals;
    std::vector<int> SurfaceParticles;
	Eigen::Vector3f* GlobalxMeans;
    Eigen::Matrix3f* GlobalGs;

	Evaluator(SurfReconstructor* surf_constructor, std::vector<Eigen::Vector3f>* global_particles, 
    std::vector<float>* densities, std::vector<float>* masses, std::vector<float>* radiuses,
    float density, float mass, float radius);

	void SingleEval(const Eigen::Vector3f& pos, float& scalar, Eigen::Vector3f& gradient, bool use_normalize = true, bool use_signed = true, bool grad_normalize = true);

    void GridEval(
        std::vector<Eigen::Vector3f>& sample_points, std::vector<float>& field_scalars, std::vector<Eigen::Vector3f>& field_gradients,
        bool& signchange, int oversample);

    bool CheckSplash(const int& pIdx);
    float CalculateMaxScalarConstR();
    float CalculateMaxScalarVarR();
    float RecommendIsoValueConstR();
    float RecommendIsoValueVarR();
    void CalcParticlesNormal();
    float CurvEval(std::vector<int>& p_list);
private:
    float inf_factor;

    float general_kernel(double d2, double h2, double h);
    float spiky_kernel(double d, double h);
    float viscosity_kernel(double d, double h);

	float IsotropicInterpolate(const int pIdx, const double d);
	float AnisotropicInterpolate(const int pIdx, const Eigen::Vector3f& diff);
	void compute_Gs_xMeans();
	double wij(double d, double r);

    void IsotropicEval(const Eigen::Vector3f& pos, float& info, float* temp_scalars, float& sample_radius);
    void AnisotropicEval(const Eigen::Vector3f& pos, float& info, float* temp_scalars, float& sample_radius);
};
