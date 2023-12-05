#include <float.h>
#include "evaluator.h"
#include "surface_reconstructor.h"


Evaluator::Evaluator(SurfReconstructor* surf_constructor,
		std::vector<Eigen::Vector3d>* global_particles, 
        std::vector<double>* radiuses, double radius)
{
    constructor = surf_constructor;
	GlobalPoses = global_particles;
    NeighborFactor = constructor->getNeighborFactor();
    SmoothFactor = constructor->getSmoothFactor();
    GlobalRadius = radiuses;
    if (!IS_CONST_RADIUS)
    {
        GlobalRadius2 = new std::vector<double>(GlobalRadius->size());
        GlobalRadius3 = new std::vector<double>(GlobalRadius->size());
        GlobalInflunce2 = new std::vector<double>(GlobalRadius->size());
        GlobalSigma = new std::vector<double>(GlobalRadius->size());
        double influnce;
#pragma omp parallel for
        for (int i = 0; i < GlobalRadius->size(); i++)
        {
            influnce = GlobalRadius->at(i) * SmoothFactor;
            GlobalRadius2->at(i) = pow(GlobalRadius->at(i), 2);
            GlobalRadius3->at(i) = GlobalRadius->at(i) * GlobalRadius2->at(i);
            GlobalInflunce2->at(i) = pow(influnce, 2);
            GlobalSigma->at(i) = (315 / (64 * pow(influnce, 9))) * inv_pi;
        }
    } else {
        Radius = radius;
        Radius2 = Radius * Radius;
        Radius3 = Radius * Radius2;
        Influnce2 = pow(Radius * SmoothFactor, 2);
        Sigma = (315 / (64 * pow(radius * SmoothFactor, 9))) * inv_pi;
    }

    if (CALC_P_NORMAL)
    {
        PariclesNormals.clear();
        PariclesNormals.resize(constructor->getGlobalParticlesNum(), Eigen::Vector3d(0, 0, 0)); 
    }
    
    GlobalSplash.clear();
    GlobalSplash.resize(constructor->getGlobalParticlesNum(), 0);
    GlobalxMeans = new Eigen::Vector3d[constructor->getGlobalParticlesNum()];
    if (USE_ANI)
    {
        GlobalGs = new Eigen::Matrix3d[constructor->getGlobalParticlesNum()];
        GlobalDeterminant.resize(constructor->getGlobalParticlesNum());
    }
    
    compute_Gs_xMeans();
}

void Evaluator::SingleEval(const Eigen::Vector3d& pos, double& scalar)
{
	scalar = 0;
    std::vector<int> neighbors;
    if (IS_CONST_RADIUS)
    {
        constructor->getHashGrid()->GetPIdxList(pos, neighbors);
    } else {
        constructor->getSearcher()->GetNeighbors(pos, neighbors);
    }
    Eigen::Vector3d diff;
    for (int pIdx : neighbors)
    {
        if (this->CheckSplash(pIdx))
        {
            continue;
        }

        diff = pos - GlobalxMeans[pIdx];
        if (USE_ANI)
        {
            scalar += AnisotropicInterpolate(pIdx, diff);
        } else {
            scalar += IsotropicInterpolate(pIdx, diff.squaredNorm());
        }
    }
    scalar = constructor->getIsoValue() - ((scalar - constructor->getMinScalar()) / constructor->getMaxScalar() * 255);
}

void Evaluator::SingleEvalWithGrad(const Eigen::Vector3d& pos, double& scalar, Eigen::Vector3d& gradient, bool use_normalize, bool use_signed, bool grad_normalize)
{
	if (constructor->getMaxScalar() >= 0)
	{
		scalar = 0;
        if (!gradient.isZero())
        {
            gradient.setZero();
        }
	}

	double info = 0.0f;
	double temp_scalars[6] = {0.0f};
    double sample_radius = 0.0f;

    if (USE_ANI)
    {
        AnisotropicEval(pos, info, temp_scalars, sample_radius);
    }
    else
    {
        IsotropicEval(pos, info, temp_scalars, sample_radius);
    }
	
    if (use_normalize)
    {
        scalar = ((info - constructor->getMinScalar()) / constructor->getMaxScalar() * 255);
    }
    if (use_signed)
    {
        scalar = constructor->getIsoValue() - scalar;
    }
	gradient[0] = ((temp_scalars[1] - temp_scalars[0]) / constructor->getMaxScalar() * 255) / (sample_radius * 2);
	gradient[1] = ((temp_scalars[3] - temp_scalars[2]) / constructor->getMaxScalar() * 255) / (sample_radius * 2);
	gradient[2] = ((temp_scalars[5] - temp_scalars[4]) / constructor->getMaxScalar() * 255) / (sample_radius * 2);
    if (grad_normalize)
    {
        gradient.normalize();
        gradient[0] = std::isnan(gradient[0]) ? 0.0f : gradient[0];
        gradient[1] = std::isnan(gradient[1]) ? 0.0f : gradient[1];
        gradient[2] = std::isnan(gradient[2]) ? 0.0f : gradient[2];
    }
}

void Evaluator::GridEval(
    double* sample_points, double* field_gradients, double cellsize, 
    bool& signchange, int oversample, bool grad_normalize)
{
    //assert(!sample_points.empty());
    bool origin_sign;
    double step = cellsize / oversample;
    for (int i = 0; i < pow(oversample+1, 3); i++)
    {
        Eigen::Vector4d p(sample_points[i*4 + 0], sample_points[i*4 + 1], sample_points[i*4 + 2], sample_points[i*4 + 3]);
        double scalar = 0.0f;
        std::vector<int> neighbors;
        if (IS_CONST_RADIUS)
        {
            constructor->getHashGrid()->GetPIdxList(p.head(3), neighbors);
        } else {
            constructor->getSearcher()->GetNeighbors(p.head(3), neighbors);
        }
        if (!neighbors.empty())
        {
            for (const int pIdx : neighbors)
            {
                if (USE_ANI)
                {
                    scalar += AnisotropicInterpolate(pIdx, p.head(3) - GlobalxMeans[pIdx]);
                } else {
                    scalar += IsotropicInterpolate(pIdx, (p.head(3) - GlobalxMeans[pIdx]).squaredNorm());
                }
            }
        }

        scalar = constructor->getIsoValue() - ((scalar - constructor->getMinScalar()) / constructor->getMaxScalar() * 255);
        sample_points[i*4 + 3] = scalar;
        origin_sign = (sample_points[3] >= 0);
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
                Eigen::Vector3d gradient(0.0f, 0.0f, 0.0f);

                next_idx = (z * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + (x + 1));
                last_idx = (z * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + (x - 1));
                if (x == 0)
                {
                    gradient[0] = (sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3]) / step;
                }
                else if (x == oversample)
                {
                    gradient[0] = (sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3]) / step;
                }
                else
                {
                    gradient[0] = (sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3]) / (step * 2);
                }

                next_idx = (z * (oversample + 1) * (oversample + 1) + (y + 1) * (oversample + 1) + x);
                last_idx = (z * (oversample + 1) * (oversample + 1) + (y - 1) * (oversample + 1) + x);
                if (y == 0)
                {
                    gradient[1] = (sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3]) / step;
                }
                else if (y == oversample)
                {
                    gradient[1] = (sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3]) / step;
                }
                else
                {
                    gradient[1] = (sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3]) / (step * 2);
                }

                next_idx = ((z + 1) * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
                last_idx = ((z - 1) * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
                if (z == 0)
                {
                    gradient[2] = (sample_points[index * 4 + 3] - sample_points[next_idx * 4 + 3]) / step;
                }
                else if (z == oversample)
                {
                    gradient[2] = (sample_points[last_idx * 4 + 3] - sample_points[index * 4 + 3]) / step;
                }
                else
                {
                    gradient[2] = (sample_points[last_idx * 4 + 3] - sample_points[next_idx * 4 + 3]) / (step * 2);
                }
                if (grad_normalize)
                {
                    gradient.normalize();
                    gradient[0] = std::isnan(gradient[0]) ? 0.0f : gradient[0];
                    gradient[1] = std::isnan(gradient[1]) ? 0.0f : gradient[1];
                    gradient[2] = std::isnan(gradient[2]) ? 0.0f : gradient[2];
                }
                field_gradients[index * 3 + 0] = gradient[0];
                field_gradients[index * 3 + 1] = gradient[1];
                field_gradients[index * 3 + 2] = gradient[2];
            }
        }
    }
}

bool Evaluator::CheckSplash(const int& pIdx)
{
    if (GlobalSplash[pIdx])
    {
        return true;
    }
    return false;
}

double Evaluator::CalculateMaxScalarConstR()
{
    double k_value = 0;

    k_value += general_kernel(0.0, Influnce2, Sigma);
    k_value += general_kernel(4 * Radius2, Influnce2, Sigma) * 6;
    k_value += general_kernel(8 * Radius2, Influnce2, Sigma) * 12;
    k_value += general_kernel(12 * Radius2, Influnce2, Sigma) * 8;

    return Radius3 * k_value;
}

double Evaluator::CalculateMaxScalarVarR()
{
    double max_scalar = 0;
    auto searchers = constructor->getSearcher();
    unsigned int rId;
    // double r, radius2, influnce, influnce2, temp_scalar;
    double temp_scalar;
    for (const auto searcher : *(searchers->getSearchers()))
    {
        temp_scalar = 0.0;
        rId = searcher->RadiusId;
        // r = searcher->Radius;
        // radius2 = r * r;
        // influnce = r * SmoothFactor;
        // influnce2 = influnce * influnce;
        temp_scalar += general_kernel(0.0, GlobalInflunce2->at(rId), GlobalSigma->at(rId));
        temp_scalar += general_kernel(4 * GlobalRadius2->at(rId), GlobalInflunce2->at(rId), GlobalSigma->at(rId)) * 6;
        temp_scalar += general_kernel(8 * GlobalRadius2->at(rId), GlobalInflunce2->at(rId), GlobalSigma->at(rId)) * 12;
        temp_scalar += general_kernel(12 * GlobalRadius2->at(rId), GlobalInflunce2->at(rId), GlobalSigma->at(rId)) * 8;
        temp_scalar = GlobalRadius3->at(rId) * temp_scalar;
        if (temp_scalar > max_scalar)
        {
            max_scalar = temp_scalar;
        }
    }
    return max_scalar;
}

double Evaluator::RecommendIsoValueConstR(const double iso_factor)
{
    double k_value = 0.0;

    k_value += general_kernel(iso_factor * Radius2, Influnce2, Sigma);

    return (((Radius3 * k_value) 
    - constructor->getMinScalar()) / constructor->getMaxScalar() * 255);
}

double Evaluator::RecommendIsoValueVarR(const double iso_factor)
{
    double recommend = 0.0;
    auto searchers = constructor->getSearcher();
    unsigned int rId;
    // double r, radius2, influnce, influnce2, temp_scalar;
    double temp_scalar;

    for (const auto searcher : *(searchers->getSearchers()))
    {
        temp_scalar = 0.0;
        rId = searcher->RadiusId;
        // r = searcher->Radius;
        // radius2 = r * r;
        // influnce = r * SmoothFactor;
        // influnce2 = influnce * influnce;
        temp_scalar += general_kernel(iso_factor * GlobalRadius2->at(rId), GlobalInflunce2->at(rId), GlobalSigma->at(rId));
        temp_scalar = GlobalRadius3->at(rId) * temp_scalar;
        if (temp_scalar > recommend)
        {
            recommend = temp_scalar;
        }
    }
    return (recommend - constructor->getMinScalar()) / constructor->getMaxScalar() * 255;
}

void Evaluator::CalcParticlesNormal()
{
#pragma omp parallel for
    for (int pIdx = 0; pIdx < constructor->getGlobalParticlesNum(); pIdx++)
    {
        double tempScalar = 0;
        Eigen::Vector3d tempGrad = Eigen::Vector3d::Zero();
        if (!CheckSplash(pIdx))
        {
            SingleEvalWithGrad(GlobalPoses->at(pIdx), tempScalar, tempGrad);
            PariclesNormals[pIdx][0] = tempGrad[0];
            PariclesNormals[pIdx][1] = tempGrad[1];
            PariclesNormals[pIdx][2] = tempGrad[2];
        }
    }
}

inline double Evaluator::general_kernel(double d2, double h2, double sigma)
{
    double p_dist = (d2 >= h2 ? 0.0 : pow(h2 - d2, 3));
    return p_dist * sigma;
}

inline double Evaluator::IsotropicInterpolate(const int pIdx, const double d2)
{
    double k_value = general_kernel(
        d2, 
        (IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2->at(pIdx)), 
        (IS_CONST_RADIUS ? Sigma : GlobalSigma->at(pIdx)));
	return (IS_CONST_RADIUS ? Radius3 : GlobalRadius3->at(pIdx)) * k_value;
}

inline double Evaluator::AnisotropicInterpolate(const int pIdx, const Eigen::Vector3d& diff)
{
    double k_value = general_kernel(
        (GlobalGs[pIdx] * diff).squaredNorm(), 
        (IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2->at(pIdx)), 
        (IS_CONST_RADIUS ? Sigma : GlobalSigma->at(pIdx)));
    return (IS_CONST_RADIUS ? Radius3 : GlobalRadius3->at(pIdx)) * (GlobalDeterminant[pIdx] * k_value);
}

inline void Evaluator::compute_xMeans(int pIdx, std::vector<int> temp_neighbors, std::vector<int> &neighbors, int &closer_neighbor, Eigen::Vector3d &xMean)
{
    double pR, pD2;
    pR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(pIdx);
    // pD = SmoothFactor * pR;
    pD2 = IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2->at(pIdx);
    double nR, nD, nD2, nI, nI2, d, d2, wj, wSum = 0;
    for (int nIdx : temp_neighbors)
    {
        if (nIdx == pIdx)
            continue;
        nR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(nIdx);
        nD = SmoothFactor * nR * 1.3;
        nD2 = nD * nD;
        nI = nR * NeighborFactor;
        nI2 = nI * nI;
        d2 = (((GlobalPoses->at(nIdx))) - ((GlobalPoses->at(pIdx)))).squaredNorm();
        if (d2 >= nI2)
        {
            continue;
        }
        d = sqrt(d2);
        wj = wij(d, nI);
        wSum += wj;
        xMean += ((GlobalPoses->at(nIdx))) * wj;
        neighbors.push_back(nIdx);
        if (d2 <= std::max(pD2, nD2))
        {
            closer_neighbor++;
        }
    }
    if (constructor->getUseXMean())
    {
        if (wSum > 0)
        {
            xMean /= wSum;
            xMean = (GlobalPoses->at(pIdx)) * (1 - constructor->getXMeanDelta()) + xMean * constructor->getXMeanDelta();
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
}

inline void Evaluator::compute_G(Eigen::Vector3d p, Eigen::Vector3d xMean, std::vector<int> neighbors, Eigen::Matrix3d &G)
{
    const double invH = 1/SmoothFactor;
    double wSum = 0, d, wj;
    double nR, nI;

    Eigen::Vector3d wd = Eigen::Vector3d::Zero();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    cov += Eigen::DiagonalMatrix<double, 3>(invH, invH, invH);
    wSum = 0.0f;
    for (int nIdx : neighbors)
    {
        nR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(nIdx);
        nI = nR * NeighborFactor;
        d = (p - GlobalPoses->at(nIdx)).norm();
        wj = wij(d, nI);
        wSum += wj;
        wd = ((GlobalPoses->at(nIdx))) - xMean;
        cov += ((wd * wd.transpose()).cast<double>() * wj);
    }
    cov /= wSum;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    Eigen::Matrix3d u = svd.matrixU();
    Eigen::Vector3d w = svd.singularValues();
    Eigen::Matrix3d v = svd.matrixV();
    
    w = Eigen::Vector3d(w.array().abs());
    const double maxSingularVal = w.maxCoeff();

    const double kr = 4.0;
    w[0] = std::max(w[0], maxSingularVal / kr);
    w[1] = std::max(w[1], maxSingularVal / kr);
    w[2] = std::max(w[2], maxSingularVal / kr);

    Eigen::Matrix3d invSigma = w.asDiagonal().inverse();
    // Compute G
    const double scale =
        std::pow(w[0] * w[1] * w[2], 1.0 / 3.0);  // volume preservation
    G = ((v * invSigma * u.transpose()) * invH * scale).cast<double>();
}

inline void Evaluator::compute_Gs_xMeans()
{
    const double invH = 1.0;
#pragma omp parallel for
    for (int pIdx = 0; pIdx < constructor->getGlobalParticlesNum(); pIdx++)
    {
        std::vector<int> tempNeighbors;
        std::vector<int> neighbors;
        int closerNeigbors = 0;
        Eigen::Vector3d xMean = Eigen::Vector3d::Zero();
        Eigen::Matrix3d G = Eigen::Matrix3d::Zero();
        if (IS_CONST_RADIUS)
        {
            constructor->getHashGrid()->GetPIdxList((GlobalPoses->at(pIdx)), tempNeighbors);
        } else {
            constructor->getSearcher()->GetNeighbors((GlobalPoses->at(pIdx)), tempNeighbors);
        }
        if (tempNeighbors.size() <= 1)
        {
            G = Eigen::DiagonalMatrix<double, 3>(invH, invH, invH);
            GlobalSplash[pIdx] = 1;
        }

        compute_xMeans(pIdx, tempNeighbors, neighbors, closerNeigbors, xMean);

        if (closerNeigbors < 1)
        {
            G = Eigen::DiagonalMatrix<double, 3>(invH, invH, invH);
            GlobalSplash[pIdx] = 1;
        } 
        else
        {
            if (USE_ANI)
            {
                compute_G(GlobalPoses->at(pIdx), xMean, neighbors, G);
            }
        }
        
        GlobalxMeans[pIdx] = Eigen::Vector3d(xMean);
        if (USE_ANI)
        {
            GlobalGs[pIdx] = Eigen::Matrix3d(G);
            GlobalDeterminant[pIdx] = G.determinant();
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

inline void Evaluator::IsotropicEval(const Eigen::Vector3d& pos, double& info, double* temp_scalars, double& sample_radius)
{    
    std::vector<int> neighbors;
    if (IS_CONST_RADIUS)
    {
        constructor->getHashGrid()->GetPIdxList(pos, neighbors);
    } else {
        constructor->getSearcher()->GetNeighbors(pos, neighbors);
    }
    if (neighbors.empty())
    {
        return;
    }

    sample_radius = IS_CONST_RADIUS ? Radius : 
    (*std::min_element(neighbors.begin(), neighbors.end(), [&](const int& a, const int& b) {
        return GlobalRadius[a] < GlobalRadius[b];
    }));

    Eigen::Vector3d x_up_pos(pos[0] + sample_radius, pos[1], pos[2]),
                    x_down_pos(pos[0] - sample_radius, pos[1], pos[2]),
                    y_up_pos(pos[0], pos[1] + sample_radius, pos[2]),
                    y_down_pos(pos[0], pos[1] - sample_radius, pos[2]),
                    z_up_pos(pos[0], pos[1], pos[2] + sample_radius),
                    z_down_pos(pos[0], pos[1], pos[2] - sample_radius);

    double d;
    for (int pIdx : neighbors)
    {
        if (this->CheckSplash(pIdx))
        {
            continue;
        }
        
        d = (pos - GlobalPoses->at(pIdx)).norm();
        info += IsotropicInterpolate(pIdx, d);

        d = (x_up_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[0] += IsotropicInterpolate(pIdx, d);

        d = (x_down_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[1] += IsotropicInterpolate(pIdx, d);

        d = (y_up_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[2] += IsotropicInterpolate(pIdx, d);

        d = (y_down_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[3] += IsotropicInterpolate(pIdx, d);

        d = (z_up_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[4] += IsotropicInterpolate(pIdx, d);

        d = (z_down_pos - GlobalPoses->at(pIdx)).norm();
        temp_scalars[5] += IsotropicInterpolate(pIdx, d);
    }
}

inline void Evaluator::AnisotropicEval(const Eigen::Vector3d& pos, double& info, double* temp_scalars, double& sample_radius)
{
    std::vector<int> neighbors;
    if (IS_CONST_RADIUS)
    {
        constructor->getHashGrid()->GetPIdxList(pos, neighbors);
    } else {
        constructor->getSearcher()->GetNeighbors(pos, neighbors);
    }
    if (neighbors.empty())
    {
        return;
    }

    sample_radius = IS_CONST_RADIUS ? Radius : 
    GlobalRadius->at(*std::min_element(neighbors.begin(), neighbors.end(), [&](const int& a, const int& b) {
        return GlobalRadius->at(a) < GlobalRadius->at(b);
    }));

    Eigen::Vector3d x_up_pos(pos[0] + sample_radius, pos[1], pos[2]),
                    x_down_pos(pos[0] - sample_radius, pos[1], pos[2]),
                    y_up_pos(pos[0], pos[1] + sample_radius, pos[2]),
                    y_down_pos(pos[0], pos[1] - sample_radius, pos[2]),
                    z_up_pos(pos[0], pos[1], pos[2] + sample_radius),
                    z_down_pos(pos[0], pos[1], pos[2] - sample_radius);

    Eigen::Vector3d diff;
    for (int pIdx : neighbors)
    {
        if (this->CheckSplash(pIdx))
        {
            continue;
        }
        
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
