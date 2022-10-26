#pragma once

#include <iostream>
#include <vector>
#include "iso_common.h"
#include "vect.h"
#include "hash_grid.h"
#include "array2d.h"


class Evaluator
{
public:
	Evaluator();
	~Evaluator();

	std::vector<vect3f>* GlobalPoses;
	std::vector<float>* GlobalDensity;
	std::vector<float>* GlobalMass;
    std::vector<int> GlobalSplash;
	vect3f* GlobalxMeans;
	Array2D<3, float>* GlobalGs;

	Evaluator(
		std::vector<vect3f>& global_particles, std::vector<float>& global_density, std::vector<float>& global_mass);

	void SingleEval(const vect3f& pos, float& scalar, vect3f& gradient, bool use_normalize, bool use_signed);

    void GridEval(
        std::vector<vect3f>& sample_points, std::vector<float>& field_scalars, std::vector<vect3f>& field_gradients,
        bool& signchange, int oversample);

    float RecommendIsoValue();

private:
    const float sample_step = P_RADIUS;
    const double influnce2 = INFLUENCE * INFLUENCE;

    float general_kernel(double d2, double h2);
    float spiky_kernel(double d, double h);
    float viscosity_kernel(double d, double h);

	float IsotrpicInterpolate(const int pIdx, const float d);
	float AnisotropicInterpolate(const int pIdx, const vect3f& diff);
	void compute_Gs_xMeans();
	double wij(double d, double r);
	void svd(const Array2D<3, double>& a, Array2D<3, double>& u, vect3d& w, Array2D<3, double>& v);
    template<typename T>
    T sign(T a, T b);
    template <typename T>
    T pythag(T a, T b);
    template<class T>
    float determinant(const Array2D<3, T>& mat);
    template<int N, class T>
    Array2D<N, T> makeMat(const vect<N, T>& v);

    void IsotropicEval(const vect3f& pos, float& info, float* temp_scalars);
    void AnisotropicEval(const vect3f& pos, float& info, float* temp_scalars);
};

inline Evaluator::Evaluator(
	std::vector<vect3f>& global_particles, std::vector<float>& global_density, std::vector<float>& global_mass)
{
	GlobalPoses = &global_particles;
	GlobalDensity = &global_density;
	GlobalMass = &global_mass;

    GlobalxMeans = new vect3f[GlobalParticlesNum];
    GlobalGs = new Array2D<3, float>[GlobalParticlesNum];
    //compute_Gs_xMeans();

	if (USE_ANI)
	{
        compute_Gs_xMeans();
	}
}

inline void Evaluator::SingleEval(const vect3f& pos, float& scalar, vect3f& gradient, bool use_normalize = true, bool use_signed = true)
{
	if (MAX_VALUE >= 0)
	{
		scalar = 0;
        if (gradient.v != 0)
        {
		    gradient.zero();
        }
	}

	float info = 0.0f;
	float temp_scalars[6] = {0.0f};

    if (USE_ANI)
    {
        AnisotropicEval(pos, info, temp_scalars);
    }
    else
    {
        IsotropicEval(pos, info, temp_scalars);
    }
	
	if (MAX_VALUE < 0)
	{
		scalar = info;
		return;
	}
	else
	{
        if (use_normalize)
        {
            scalar = ((info - MIN_VALUE) / MAX_VALUE * 255);
        }
        if (use_signed)
        {
            scalar = ISO_VALUE - scalar;
        }
		gradient[0] = ((temp_scalars[1] - temp_scalars[0]) / MAX_VALUE * 255) / P_RADIUS;
		gradient[1] = ((temp_scalars[3] - temp_scalars[2]) / MAX_VALUE * 255) / P_RADIUS;
		gradient[2] = ((temp_scalars[5] - temp_scalars[4]) / MAX_VALUE * 255) / P_RADIUS;
		gradient.normalize();
		gradient[0] = isnan(gradient[0]) ? 0.0f : gradient[0];
		gradient[1] = isnan(gradient[1]) ? 0.0f : gradient[1];
		gradient[2] = isnan(gradient[2]) ? 0.0f : gradient[2];
	}
}

inline void Evaluator::GridEval(
    std::vector<vect3f>& sample_points, std::vector<float>& field_scalars, std::vector<vect3f>& field_gradients,
    bool& signchange, int oversample)
{
    if (sample_points.empty())
    {
        exit(1);
    }
    bool origin_sign;
    for (vect3f p : sample_points)
    {
        float scalar = 0.0f;
        if (USE_ANI)
        {
            std::vector<int> pIdxList;
            hashgrid->GetPIdxList(p, pIdxList);
            if (!pIdxList.empty())
            {
                for (int pIdx : pIdxList)
                {
                    scalar += AnisotropicInterpolate(pIdx, p - GlobalxMeans[pIdx]);
                }
            }
        }
        else
        {
            std::vector<int> pIdxList;
            hashgrid->GetPIdxList(p, pIdxList);
            if (!pIdxList.empty())
            {
                float d;
                for (int pIdx : pIdxList)
                {
                    d = (p - GlobalPoses->at(pIdx)).length();
                    scalar += IsotrpicInterpolate(pIdx, d);
                }
            }
        }
        scalar = ISO_VALUE - ((scalar - MIN_VALUE) / MAX_VALUE * 255);
        field_scalars.push_back(scalar);
        origin_sign = field_scalars[0] >= 0;
        if (!signchange)
        {
            signchange = origin_sign ^ (scalar >= 0);
        }
    }
    int index, next_idx, last_idx;
    for (int z = 0; z <= oversample; z++)
    {
        for (int y = 0; y <= oversample; y++)
        {
            for (int x = 0; x <= oversample; x++)
            {
                index = (z * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
                vect3f gradient(0.0f, 0.0f, 0.0f);

                next_idx = (z * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + (x + 1));
                last_idx = (z * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + (x - 1));
                if (x == 0)
                {
                    gradient[0] = field_scalars[index] - field_scalars[next_idx];
                }
                else if (x == oversample)
                {
                    gradient[0] = field_scalars[last_idx] - field_scalars[index];
                }
                else
                {
                    gradient[0] = field_scalars[last_idx] - field_scalars[next_idx];
                }

                next_idx = (z * (oversample + 1) * (oversample + 1) + (y + 1) * (oversample + 1) + x);
                last_idx = (z * (oversample + 1) * (oversample + 1) + (y - 1) * (oversample + 1) + x);
                if (y == 0)
                {
                    gradient[1] = field_scalars[index] - field_scalars[next_idx];
                }
                else if (y == oversample)
                {
                    gradient[1] = field_scalars[last_idx] - field_scalars[index];
                }
                else
                {
                    gradient[1] = field_scalars[last_idx] - field_scalars[next_idx];
                }

                next_idx = ((z + 1) * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
                last_idx = ((z - 1) * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
                if (z == 0)
                {
                    gradient[2] = field_scalars[index] - field_scalars[next_idx];
                }
                else if (z == oversample)
                {
                    gradient[2] = field_scalars[last_idx] - field_scalars[index];
                }
                else
                {
                    gradient[2] = field_scalars[last_idx] - field_scalars[next_idx];
                }

                gradient.normalize();
                gradient[0] = isnan(gradient[0]) ? 0.0f : gradient[0];
                gradient[1] = isnan(gradient[1]) ? 0.0f : gradient[1];
                gradient[2] = isnan(gradient[2]) ? 0.0f : gradient[2];
                field_gradients.push_back(gradient);
            }
        }
    }
}

inline float Evaluator::RecommendIsoValue()
{
    double k_value;
    double recommend_dist = P_RADIUS * 1.5;
    switch (KERNEL_TYPE)
    {
    case 0:
        k_value = general_kernel(recommend_dist * recommend_dist, influnce2);
        break;
    case 1:
        k_value = spiky_kernel(recommend_dist, INFLUENCE);
        break;
    case 2:
        k_value = viscosity_kernel(recommend_dist, INFLUENCE);
        break;
    default:
        k_value = general_kernel(recommend_dist * recommend_dist, influnce2);
        break;
    }
    float scalar = (GlobalMass->at(0) / (*std::min_element(GlobalDensity->begin(), GlobalDensity->end())) * k_value);
    return ((scalar - MIN_VALUE) / MAX_VALUE * 255);
}

inline float Evaluator::general_kernel(double d2, double h2)
{
    double p_dist = (d2 >= h2 ? 0.0 : pow(h2 - d2, 3));
    double sigma = (315 / (64 * pow(INFLUENCE, 9))) * 0.318309886183790671538;
    return p_dist * sigma;
}

inline float Evaluator::spiky_kernel(double d, double h)
{
    double p_dist = (d >= h ? 0.0 : pow(h - d, 3));
    double sigma = (15 / pow(INFLUENCE, 6)) * 0.318309886183790671538;
    return p_dist * sigma;
}

inline float Evaluator::viscosity_kernel(double d, double h)
{
    double p_dist = (d >= h ? 0.0 : (-pow(d, 3) / (2 * pow(h, 3)) + pow(d, 2) / pow(h, 2) + d / (2 * h) - 1));
    double sigma = (15 / (2 * pow(h, 3))) * 0.318309886183790671538;
    return p_dist * sigma;
}

inline float Evaluator::IsotrpicInterpolate(const int pIdx, const float d)
{
	if (d > INFLUENCE)
		return 0.0f;
	float kernel_value = 315 / (64 * pow(INFLUENCE, 9)) * 0.318309886183790671538 * pow(((INFLUENCE * INFLUENCE) - (d * d)), 3);
	return (*GlobalMass)[pIdx] / (*GlobalDensity)[pIdx] * kernel_value;
}

inline float Evaluator::AnisotropicInterpolate(const int pIdx, const vect3f& diff)
{
    double k_value;
    switch (KERNEL_TYPE)
    {
    case 0:
        k_value = general_kernel((GlobalGs[pIdx] * diff).length2(), influnce2);
        break;
    case 1:
        k_value = spiky_kernel((GlobalGs[pIdx] * diff).length(), INFLUENCE);
        break;
    case 2:
        k_value = viscosity_kernel((GlobalGs[pIdx] * diff).length(), INFLUENCE);
        break;
    default:
        k_value = general_kernel((GlobalGs[pIdx] * diff).length2(), influnce2);
        break;
    }
    //static const double sigma = (315 / (64 * pow(INFLUENCE, 9))) * 0.318309886183790671538;
    //double p_dis, dist2 = (GlobalGs[pIdx] * diff).length2();
    //if (dist2 >= influnce2)
    //{
    //    p_dis = 0.0;
    //}
    //else
    //{
    //    p_dis = pow(influnce2 - dist2, 3);
    //}
    double gDet = determinant(GlobalGs[pIdx]);
    return (GlobalMass->at(pIdx) / GlobalDensity->at(pIdx)) * (gDet * k_value);
}

inline void Evaluator::compute_Gs_xMeans()
{
	const double h = P_RADIUS;
	const double h2 = h * h;
	const double R = INFLUENCE;
    const double R2 = R * R;
    const double invH = 1.0;

//#pragma omp parallel
    {
#pragma omp parallel for schedule(static, 16) //shared(hashgrid)// shared(globalxmeans, globalgs) 
        for (int pIdx = 0; pIdx < GlobalParticlesNum; pIdx++)
        {
            //printf("%d, ", pIdx);
            std::vector<int> tempNeighborList;
            std::vector<int> neighborList;
            vect3f xMean(0.0f, 0.0f, 0.0f);
            Array2D<3, float> G;
            hashgrid->GetPIdxList((GlobalPoses->at(pIdx)), tempNeighborList);
            if (tempNeighborList.empty())
                continue;
            double wSum = 0, d2, d, wj;
            for (int nIdx : tempNeighborList)
            {
                if (nIdx == pIdx)
                    continue;
                d2 = (((GlobalPoses->at(nIdx))) - ((GlobalPoses->at(pIdx)))).length2();
                if (d2 > R2)
                {
                    continue;
                }
                d = sqrt(d2);
                wj = wij(d, R);
                wSum += wj;
                xMean += ((GlobalPoses->at(nIdx))) * wj;
                neighborList.push_back(nIdx);
            }
            if (USE_XMEAN)
            {
                if (wSum > 0)
                {
                    xMean /= wSum;
                    xMean = (GlobalPoses->at(pIdx)) * (1 - XMEAN_DELTA) + xMean * XMEAN_DELTA;
                }
                else
                {
                    xMean = (GlobalPoses->at(pIdx));
                }
            }
            else
            {
                xMean = (GlobalPoses->at(pIdx));
            }

            //xMean = (GlobalPoses->at(pIdx));

            if (neighborList.size() < MIN_NEIGHBORS_NUM)
            {
                G.makeScaleMat(invH, invH, invH);
            }
            else
            {
                vect3d wd;
                Array2D<3, double> cov;
                cov.makeScaleMat(h2);
                wSum = 0.0f;
                for (int nIdx : neighborList)
                {
                    d = (xMean - ((GlobalPoses->at(nIdx)))).length();
                    wj = wij(d, R);
                    wSum += wj;
                    wd = ((GlobalPoses->at(nIdx))) - xMean;
                    cov += (makeMat(wd) * wj);
                }
                cov /= wSum;

                Array2D<3, double> u;
                vect3d v;
                Array2D<3, double> w;
                svd(cov, u, v, w);

                v.absolute();

                const double maxSingularVal = std::max({ v[0], v[1], v[2] });

                const double kr = 4.0;
                v[0] = std::max(v[0], maxSingularVal / kr);
                v[1] = std::max(v[1], maxSingularVal / kr);
                v[2] = std::max(v[2], maxSingularVal / kr);

                Array2D<3, double> invSigma;
                invSigma.makeScaleMat(v.inverse());


                // Compute G
                const double scale =
                    std::pow(v[0] * v[1] * v[2], 1.0 / 3.0);  // volume preservation
                G = ((w * invSigma * u.transposed()) * invH * scale);
            }
            GlobalxMeans[pIdx] = vect3f(xMean);
            GlobalGs[pIdx] = Array2D<3, float>(G);
        }
    }
}

inline double Evaluator::wij(double d, double r)
{
	if (d < r)
	{
		return (1.0 - pow(d / r, 3));
	}
	else
	{
		return 0.0;
	}
}



inline void Evaluator::svd(const Array2D<3, double>& a, Array2D<3, double>& u, vect3d& w, Array2D<3, double>& v)
{
    const int m = 3;
    const int n = 3;

    int flag, i = 0, its = 0, j = 0, jj = 0, k = 0, l = 0, nm = 0;
    double c = 0, f = 0, h = 0, s = 0, x = 0, y = 0, z = 0;
    double anorm = 0, g = 0, scale = 0;

    // Prepare workspace
    vect3d rv1;
    u = a;
    w = vect3d();
    v = Array2D<3, double>();

    // Householder reduction to bidiagonal form
    for (i = 0; i < n; i++) {
        // left-hand reduction
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0;
        if (i < m) {
            for (k = i; k < m; k++) {
                scale += std::fabs(u(k, i));
            }
            if (scale) {
                for (k = i; k < m; k++) {
                    u(k, i) /= scale;
                    s += u(k, i) * u(k, i);
                }
                f = u(i, i);
                g = -sign(std::sqrt(s), f);
                h = f * g - s;
                u(i, i) = f - g;
                if (i != n - 1) {
                    for (j = l; j < n; j++) {
                        for (s = 0, k = i; k < m; k++) {
                            s += u(k, i) * u(k, j);
                        }
                        f = s / h;
                        for (k = i; k < m; k++) {
                            u(k, j) += f * u(k, i);
                        }
                    }
                }
                for (k = i; k < m; k++) {
                    u(k, i) *= scale;
                }
            }
        }
        w[i] = scale * g;

        // right-hand reduction
        g = s = scale = 0;
        if (i < m && i != n - 1) {
            for (k = l; k < n; k++) {
                scale += std::fabs(u(i, k));
            }
            if (scale) {
                for (k = l; k < n; k++) {
                    u(i, k) /= scale;
                    s += u(i, k) * u(i, k);
                }
                f = u(i, l);
                g = -sign(std::sqrt(s), f);
                h = f * g - s;
                u(i, l) = f - g;
                for (k = l; k < n; k++) {
                    rv1[k] = (double)u(i, k) / h;
                }
                if (i != m - 1) {
                    for (j = l; j < m; j++) {
                        for (s = 0, k = l; k < n; k++) {
                            s += u(j, k) * u(i, k);
                        }
                        for (k = l; k < n; k++) {
                            u(j, k) += s * rv1[k];
                        }
                    }
                }
                for (k = l; k < n; k++) {
                    u(i, k) *= scale;
                }
            }
        }
        anorm = std::max(anorm, (std::fabs((double)w[i]) + std::fabs(rv1[i])));
    }

    // accumulate the right-hand transformation
    for (i = n - 1; i >= 0; i--) {
        if (i < n - 1) {
            if (g) {
                for (j = l; j < n; j++) {
                    v(j, i) = ((u(i, j) / u(i, l)) / g);
                }
                // T division to avoid underflow
                for (j = l; j < n; j++) {
                    for (s = 0, k = l; k < n; k++) {
                        s += u(i, k) * v(k, j);
                    }
                    for (k = l; k < n; k++) {
                        v(k, j) += s * v(k, i);
                    }
                }
            }
            for (j = l; j < n; j++) {
                v(i, j) = v(j, i) = 0;
            }
        }
        v(i, i) = 1;
        g = rv1[i];
        l = i;
    }

    // accumulate the left-hand transformation
    for (i = n - 1; i >= 0; i--) {
        l = i + 1;
        g = w[i];
        if (i < n - 1) {
            for (j = l; j < n; j++) {
                u(i, j) = 0;
            }
        }
        if (g) {
            g = 1 / g;
            if (i != n - 1) {
                for (j = l; j < n; j++) {
                    for (s = 0, k = l; k < m; k++) {
                        s += u(k, i) * u(k, j);
                    }
                    f = (s / u(i, i)) * g;
                    for (k = i; k < m; k++) {
                        u(k, j) += f * u(k, i);
                    }
                }
            }
            for (j = i; j < m; j++) {
                u(j, i) = u(j, i) * g;
            }
        }
        else {
            for (j = i; j < m; j++) {
                u(j, i) = 0;
            }
        }
        ++u(i, i);
    }

    // diagonalize the bidiagonal form
    for (k = n - 1; k >= 0; k--) {
        // loop over singular values
        for (its = 0; its < 30; its++) {
            // loop over allowed iterations
            flag = 1;
            for (l = k; l >= 0; l--) {
                // test for splitting
                nm = l - 1;
                if (std::fabs(rv1[l]) + anorm == anorm) {
                    flag = 0;
                    break;
                }
                if (std::fabs((double)w[nm]) + anorm == anorm) {
                    break;
                }
            }
            if (flag) {
                c = 0;
                s = 1;
                for (i = l; i <= k; i++) {
                    f = s * rv1[i];
                    if (std::fabs(f) + anorm != anorm) {
                        g = w[i];
                        h = pythag(f, g);
                        w[i] = (double)h;
                        h = 1 / h;
                        c = g * h;
                        s = -f * h;
                        for (j = 0; j < m; j++) {
                            y = u(j, nm);
                            z = u(j, i);
                            u(j, nm) = y * c + z * s;
                            u(j, i) = z * c - y * s;
                        }
                    }
                }
            }
            z = w[k];
            if (l == k) {
                // convergence
                if (z < 0) {
                    // make singular value nonnegative
                    w[k] = -z;
                    for (j = 0; j < n; j++) {
                        v(j, k) = -v(j, k);
                    }
                }
                break;
            }
            if (its >= 30) {
                throw("No convergence after 30 iterations");
            }

            // shift from bottom 2 x 2 minor
            x = w[l];
            nm = k - 1;
            y = w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y);
            g = pythag(f, (double)1);
            f = ((x - z) * (x + z) + h * ((y / (f + sign(g, f))) - h)) / x;

            // next QR transformation
            c = s = 1;
            for (j = l; j <= nm; j++) {
                i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s * g;
                g = c * g;
                z = pythag(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < n; jj++) {
                    x = v(jj, j);
                    z = v(jj, i);
                    v(jj, j) = x * c + z * s;
                    v(jj, i) = z * c - x * s;
                }
                z = pythag(f, h);
                w[j] = z;
                if (z) {
                    z = 1 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < m; jj++) {
                    y = u(jj, j);
                    z = u(jj, i);
                    u(jj, j) = y * c + z * s;
                    u(jj, i) = z * c - y * s;
                }
            }
            rv1[l] = 0;
            rv1[k] = f;
            w[k] = x;
        }
    }
}

inline void Evaluator::IsotropicEval(const vect3f& pos, float& info, float* temp_scalars)
{
    vect3f x_up_pos(pos.v[0] + sample_step, pos.v[1], pos.v[2]),
        x_down_pos(pos.v[0] - sample_step, pos.v[1], pos.v[2]),
        y_up_pos(pos.v[0], pos.v[1] + sample_step, pos.v[2]),
        y_down_pos(pos.v[0], pos.v[1] - sample_step, pos.v[2]),
        z_up_pos(pos.v[0], pos.v[1], pos.v[2] + sample_step),
        z_down_pos(pos.v[0], pos.v[1], pos.v[2] - sample_step);
    
    std::vector<int> pIdxList;
    hashgrid->GetPIdxList(pos, pIdxList);
    if (pIdxList.empty())
    {
        return;
    }
    float d;
    for (int pIdx : pIdxList)
    {
        d = (pos - GlobalPoses->at(pIdx)).length();
        info += IsotrpicInterpolate(pIdx, d);

        d = (x_up_pos - GlobalPoses->at(pIdx)).length();
        temp_scalars[0] += IsotrpicInterpolate(pIdx, d);

        d = (x_down_pos - GlobalPoses->at(pIdx)).length();
        temp_scalars[1] += IsotrpicInterpolate(pIdx, d);

        d = (y_up_pos - GlobalPoses->at(pIdx)).length();
        temp_scalars[2] += IsotrpicInterpolate(pIdx, d);

        d = (y_down_pos - GlobalPoses->at(pIdx)).length();
        temp_scalars[3] += IsotrpicInterpolate(pIdx, d);

        d = (z_up_pos - GlobalPoses->at(pIdx)).length();
        temp_scalars[4] += IsotrpicInterpolate(pIdx, d);

        d = (z_down_pos - GlobalPoses->at(pIdx)).length();
        temp_scalars[5] += IsotrpicInterpolate(pIdx, d);
    }
}

inline void Evaluator::AnisotropicEval(const vect3f& pos, float& info, float* temp_scalars)
{
    vect3f x_up_pos(pos.v[0] + sample_step, pos.v[1], pos.v[2]),
        x_down_pos(pos.v[0] - sample_step, pos.v[1], pos.v[2]),
        y_up_pos(pos.v[0], pos.v[1] + sample_step, pos.v[2]),
        y_down_pos(pos.v[0], pos.v[1] - sample_step, pos.v[2]),
        z_up_pos(pos.v[0], pos.v[1], pos.v[2] + sample_step),
        z_down_pos(pos.v[0], pos.v[1], pos.v[2] - sample_step);

    std::vector<int> pIdxList;
    hashgrid->GetPIdxList(pos, pIdxList);
    if (pIdxList.empty())
    {
        return;
    }
    vect3f diff;
    for (int pIdx : pIdxList)
    {
        //if (std::find(GlobalSplash.begin(), GlobalSplash.end(), pIdx) != GlobalSplash.end())
        //{
        //    continue;
        //}
        diff = pos - GlobalxMeans[pIdx];
        info += AnisotropicInterpolate(pIdx, diff);

        diff = x_up_pos - GlobalxMeans[pIdx];
        temp_scalars[0] += AnisotropicInterpolate(pIdx, diff);

        diff = x_down_pos - GlobalxMeans[pIdx];
        temp_scalars[1] += AnisotropicInterpolate(pIdx, diff);

        diff = y_up_pos - GlobalxMeans[pIdx];
        temp_scalars[2] += AnisotropicInterpolate(pIdx, diff);

        diff = y_down_pos - GlobalxMeans[pIdx];
        temp_scalars[3] += AnisotropicInterpolate(pIdx, diff);

        diff = z_up_pos - GlobalxMeans[pIdx];
        temp_scalars[4] += AnisotropicInterpolate(pIdx, diff);

        diff = z_down_pos - GlobalxMeans[pIdx];
        temp_scalars[5] += AnisotropicInterpolate(pIdx, diff);
    }
}

template<typename T>
inline T Evaluator::sign(T a, T b)
{
    return b >= 0.0 ? std::fabs(a) : -std::fabs(a);
}

template<typename T>
inline T Evaluator::pythag(T a, T b)
{
    T at = std::fabs(a);
    T bt = std::fabs(b);
    T ct;
    T result;

    if (at > bt) {
        ct = bt / at;
        result = at * std::sqrt(1 + ct * ct);
    }
    else if (bt > 0) {
        ct = at / bt;
        result = bt * std::sqrt(1 + ct * ct);
    }
    else {
        result = 0;
    }

    return result;
}

template<class T>
inline float Evaluator::determinant(const Array2D<3, T>& mat)
{
    return  mat.data[0][0] * mat.data[1][1] * mat.data[2][2] -
            mat.data[0][0] * mat.data[1][2] * mat.data[2][1] +
            mat.data[0][1] * mat.data[1][2] * mat.data[2][0] -
            mat.data[0][1] * mat.data[1][0] * mat.data[2][2] +
            mat.data[0][2] * mat.data[1][0] * mat.data[2][1] -
            mat.data[0][2] * mat.data[1][1] * mat.data[2][0];
}

template<int N, class T>
inline Array2D<N, T> Evaluator::makeMat(const vect<N, T>& v)
{
    Array2D<N, T> r;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            r.data[i][j] = v.v[i] * v.v[j];
        }
    }
    return r;
}
