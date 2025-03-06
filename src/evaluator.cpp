#include <float.h>
#include "evaluator.h"


Evaluator::Evaluator(
    std::shared_ptr<HashGrid>& hashgrid,
    std::shared_ptr<MultiLevelSearcher>& searcher,
    std::vector<Eigen::Vector3f>* global_particles, 
    std::vector<float>* radiuses, float radius)
{
    _hashgrid = hashgrid;
    _searcher = searcher;
	GlobalPoses = global_particles;
    GlobalRadius = radiuses;
    _GlobalParticlesNum = global_particles->size();
    if (!IS_CONST_RADIUS)
    {
        GlobalRadius2.resize(_GlobalParticlesNum);
        GlobalRadius3.resize(_GlobalParticlesNum);
        GlobalInflunce2.resize(_GlobalParticlesNum);
        GlobalSigma.resize(_GlobalParticlesNum);
#pragma omp parallel for
        for (int i = 0; i < _GlobalParticlesNum; i++)
        {
            float smooth = GlobalRadius->at(i) * _SMOOTH_FACTOR;
            GlobalRadius2[i] = pow(GlobalRadius->at(i), 2);
            GlobalRadius3[i] = GlobalRadius->at(i) * GlobalRadius2[i];
            GlobalInflunce2[i] = pow(smooth, 2);
            GlobalSigma[i] = 1/pow(smooth, 6);   //(315 / (64 * pow(influnce, 9))) * inv_pi;
        }
    } else {
        Radius = radius;
        Radius2 = Radius * Radius;
        Radius3 = Radius * Radius2;
        Influnce2 = pow(Radius * _SMOOTH_FACTOR, 2);
        Sigma = 1 / pow(radius * _SMOOTH_FACTOR, 6);    //(315 / (64 * pow(radius * _SMOOTH_FACTOR, 9))) * inv_pi;
    }

    if (!USE_POLY6)
    {
        _XMEAN_DELTA = 0.9;
    }

    if (CALC_P_NORMAL)
    {
        PariclesNormals.clear();
        PariclesNormals.resize(_GlobalParticlesNum, Eigen::Vector3f(0, 0, 0)); 
    }
    
    GlobalSplash.resize(_GlobalParticlesNum, 0);
    GlobalSurface.resize(_GlobalParticlesNum, 0);
    GlobalxMeans.resize(_GlobalParticlesNum);
    if (USE_ANI)
    {
        GlobalGs.resize(_GlobalParticlesNum);
        GlobalDeterminant.resize(_GlobalParticlesNum);
    }
}

void Evaluator::SingleEval(const Eigen::Vector3f& pos, float& scalar)
{
	scalar = 0;
    std::vector<int> neighbors;
    if (IS_CONST_RADIUS)
    {
        _hashgrid->GetPIdxList(pos, neighbors);
    } else {
        // _searcher->GetNeighbors(pos, neighbors);
    }
    Eigen::Vector3f diff;
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
    scalar = _ISO_VALUE - scalar;   //((scalar - _MIN_SCALAR) / _MAX_SCALAR * 255);
}

void Evaluator::SingleEvalWithGrad(const Eigen::Vector3f& pos, float& scalar, Eigen::Vector3f& gradient)
{
	if (_MAX_SCALAR >= 0)
	{
		scalar = 0;
        if (!gradient.isZero())
        {
            gradient.setZero();
        }
	}

    std::vector<int> neighbors;
    if (IS_CONST_RADIUS)
    {
        _hashgrid->GetPIdxList(pos, neighbors);
    } else {
        // _searcher->GetNeighbors(pos, neighbors);
    }
    Eigen::Vector3f diff;
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
            gradient += AnisotropicInterpolateGrad(pIdx, diff);
        } else {
            scalar += IsotropicInterpolate(pIdx, diff.squaredNorm());
            gradient += IsotropicInterpolateGrad(pIdx, diff.squaredNorm(), diff);
        }
    }
    scalar = _ISO_VALUE - scalar;   //((scalar - _MIN_SCALAR) / _MAX_SCALAR * 255);
}

void Evaluator::GridEval(
    float* sample_points, float* field_gradients, float cellsize, 
    bool& signchange, int oversample, bool grad_normalize)
{
    bool origin_sign;
    float step = cellsize / oversample;
    for (int i = 0; i < pow(oversample+1, 3); i++)
    {
        // Eigen::Vector4f p(sample_points[i*4 + 0], sample_points[i*4 + 1], sample_points[i*4 + 2], sample_points[i*4 + 3]);
        // float scalar = 0.0f;
        // Eigen::Vector3f diff;   //, gradient
        // std::vector<int> neighbors;
        // if (IS_CONST_RADIUS)
        // {
        //     _hashgrid->GetPIdxList(p.head(3), neighbors);
        // } else {
        //     _searcher->GetNeighbors(p.head(3), neighbors);
        // }
        // if (!neighbors.empty())
        // {
        //     for (const int pIdx : neighbors)
        //     {
        //         if (!CheckSplash(pIdx))
        //         {
        //             diff = p.head(3) - GlobalxMeans[pIdx];
        //             if (USE_ANI)
        //             {
        //                 scalar += AnisotropicInterpolate(pIdx, diff);
        //                 // gradient += AnisotropicInterpolateGrad(pIdx, diff);
        //             } else {
        //                 scalar += IsotropicInterpolate(pIdx, diff.squaredNorm());
        //                 // gradient += IsotropicInterpolateGrad(pIdx, diff.squaredNorm(), diff);
        //             }
        //         }
        //     }
        // }
        SingleEval(
            Eigen::Vector3f(sample_points[i*4 + 0], sample_points[i*4 + 1], sample_points[i*4 + 2]),
            sample_points[i*4 + 3]
        );
        // scalar = _ISO_VALUE - scalar;   //((scalar - _MIN_SCALAR) / _MAX_SCALAR * 255);
        // sample_points[i*4 + 3] = scalar;
        origin_sign = (sample_points[3] >= 0);
        if (!signchange)
        {
            signchange = origin_sign ^ (sample_points[i*4 + 3] >= 0);
        }
        // field_gradients[i * 3 + 0] = gradient[0];
        // field_gradients[i * 3 + 1] = gradient[1];
        // field_gradients[i * 3 + 2] = gradient[2];
        // if (grad_normalize)
        // {
        //     gradient.normalize();
        //     gradient[0] = std::isnan(gradient[0]) ? 0.0f : gradient[0];
        //     gradient[1] = std::isnan(gradient[1]) ? 0.0f : gradient[1];
        //     gradient[2] = std::isnan(gradient[2]) ? 0.0f : gradient[2];
        // }
    }
    int index, next_idx, last_idx;
    for (int z = 0; z <= oversample; z++)
    {
        for (int y = 0; y <= oversample; y++)
        {
            for (int x = 0; x <= oversample; x++)
            {
                index = (z * (oversample + 1) * (oversample + 1) + y * (oversample + 1) + x);
                Eigen::Vector3f gradient(0.0f, 0.0f, 0.0f);

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

bool Evaluator::CheckSurface(const int& pIdx)
{
    if (GlobalSurface[pIdx])
    {
        return true;
    }
    return false;
}

void Evaluator::CalculateMaxScalarConstR()
{
    float k_value = 0;

    if (USE_POLY6)
    {
        k_value += poly6_kernel(0.0, Influnce2, Sigma);
        k_value += poly6_kernel(4 * Radius2, Influnce2, Sigma);
        k_value += poly6_kernel(8 * Radius2, Influnce2, Sigma);
        k_value += poly6_kernel(12 * Radius2, Influnce2, Sigma);
    } else {
        k_value += Bspline_kernel(0.0, Sigma);
        k_value += Bspline_kernel(1.0, Sigma) * 6;
        k_value += Bspline_kernel(sqrt2, Sigma) * 12;
        k_value += Bspline_kernel(sqrt3, Sigma) * 8;
    }
    // k_value = general_kernel(0.0, Influnce2, Sigma);

    _MAX_SCALAR = k_value;    //Radius3 * 
}

void Evaluator::CalculateMaxScalarVarR()
{
    float max_scalar = 0;
    unsigned int rId;
    float temp_scalar;
    for (const auto searcher : *(_searcher->getSearchers()))
    {
        temp_scalar = 0.0;
        rId = searcher->RadiusId;
        if (USE_POLY6)
        {
            temp_scalar += poly6_kernel(0.0, GlobalInflunce2[rId], GlobalSigma[rId]);
            temp_scalar += poly6_kernel(4 * GlobalRadius2[rId], GlobalInflunce2[rId], GlobalSigma[rId]) * 6;
            temp_scalar += poly6_kernel(8 * GlobalRadius2[rId], GlobalInflunce2[rId], GlobalSigma[rId]) * 12;
            temp_scalar += poly6_kernel(12 * GlobalRadius2[rId], GlobalInflunce2[rId], GlobalSigma[rId]) * 8;
        } else {
            temp_scalar += Bspline_kernel(0.0, GlobalSigma[rId]);
            temp_scalar += Bspline_kernel(1.0, GlobalSigma[rId]) * 6;
            temp_scalar += Bspline_kernel(sqrt2, GlobalSigma[rId]) * 12;
            temp_scalar += Bspline_kernel(sqrt3, GlobalSigma[rId]) * 8;
        }
        // temp_scalar = GlobalRadius3[rId] * temp_scalar;
        if (temp_scalar > max_scalar)
        {
            max_scalar = temp_scalar;
        }
    }
    _MAX_SCALAR = max_scalar;
}

void Evaluator::RecommendIsoValueConstR()
{
    float k_value = 0.0;

    if (USE_POLY6)
    {
        k_value = poly6_kernel(_ISO_FACTOR * Radius2, Influnce2, Sigma);
    } else {
        k_value = Bspline_kernel(0.01, Sigma);
    }

    _ISO_VALUE = k_value;   //(((k_value) - _MIN_SCALAR) / _MAX_SCALAR * 255);
}

void Evaluator::RecommendIsoValueVarR()
{
    float recommend = 0.0;
    auto searchers = _searcher;
    unsigned int rId;
    float temp_scalar;

    for (const auto searcher : *(searchers->getSearchers()))
    {
        temp_scalar = 0.0;
        rId = searcher->RadiusId;
        if (USE_POLY6)
        {
            temp_scalar = poly6_kernel(_ISO_FACTOR * GlobalRadius2[rId], GlobalInflunce2[rId], GlobalSigma[rId]);
        } else {
            temp_scalar = Bspline_kernel(0.25, GlobalSigma[rId]);
        }
        // temp_scalar = GlobalRadius3[rId] * temp_scalar;
        if (temp_scalar > recommend)
        {
            recommend = temp_scalar;
        }
    }
    _ISO_VALUE = recommend; //(recommend - _MIN_SCALAR) / _MAX_SCALAR * 255;
}

void Evaluator::CalcParticlesNormal()
{
#pragma omp parallel for
    for (int pIdx = 0; pIdx < _GlobalParticlesNum; pIdx++)
    {
        float tempScalar = 0;
        Eigen::Vector3f tempGrad = Eigen::Vector3f::Zero();
        if (!CheckSplash(pIdx))
        {
            SingleEvalWithGrad(GlobalPoses->at(pIdx), tempScalar, PariclesNormals[pIdx]);
        }
    }
}

inline float Evaluator::poly6_kernel(float d2, float h2, float sigma)
{
    float p_dist = (d2 > h2 ? 0.0 : pow(h2 - d2, 3));
    return p_dist * sigma;
}

inline float Evaluator::Bspline_kernel(float ratio, float sigma)
{
    if (ratio < 2)
    {
        if (ratio <= 1)
        {
            return sigma * (1 - 1.5*pow(ratio, 2) + 0.75*pow(ratio,3));
        } else {
            return sigma * (0.25 * pow(2 - ratio, 3));
        }
    } else {
        return 0.0f;
    }
}

inline float Gaussian_kernel(float ratio2, float sigma)
{
    if (ratio2 <= 9)
    {
        return sigma * pow(2.718281828459045, -1 * ratio2);
    } else {
        0.0f;
    }
}

inline Eigen::Vector3f Evaluator::poly6_gradient_kernel(float d2, float h2, float sigma, const Eigen::Vector3f diff)
{
    Eigen::Vector3f grad = Eigen::Vector3f::Zero();
    grad[0] = sigma * (-6 * diff[0]) * (d2 > h2 ? 0.0f : ((h2 - d2) * (h2 - d2)));
    grad[1] = sigma * (-6 * diff[1]) * (d2 > h2 ? 0.0f : ((h2 - d2) * (h2 - d2)));
    grad[2] = sigma * (-6 * diff[2]) * (d2 > h2 ? 0.0f : ((h2 - d2) * (h2 - d2)));
    return grad;
}

inline float Evaluator::IsotropicInterpolate(const int pIdx, const float d2)
{
    float k_value = poly6_kernel(
        d2, (IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx]), 
        (IS_CONST_RADIUS ? Sigma : GlobalSigma[pIdx]));
    // float k_value = Bspline_kernel(
    //     sqrt(d2/(IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx])), 
    //     (IS_CONST_RADIUS ? Sigma : GlobalSigma[pIdx]));
	return k_value; //(IS_CONST_RADIUS ? Radius3 : GlobalRadius3[pIdx]) * 
}

inline Eigen::Vector3f Evaluator::IsotropicInterpolateGrad(const int pIdx, const float d2, const Eigen::Vector3f diff)
{
    Eigen::Vector3f grad = poly6_gradient_kernel(
        d2, 
        (IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx]), 
        (IS_CONST_RADIUS ? Sigma : GlobalSigma[pIdx]), diff);
    return grad;    //(IS_CONST_RADIUS ? Radius3 : GlobalRadius3[pIdx]) * 
}

inline float Evaluator::AnisotropicInterpolate(const int pIdx, const Eigen::Vector3f diff)
{
    float k_value;
    if (USE_POLY6)
    {
        k_value = poly6_kernel(
            (GlobalGs[pIdx] * diff).squaredNorm(), 
            (IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx]), 
            (IS_CONST_RADIUS ? Sigma : GlobalSigma[pIdx]));        
    } else {
        k_value = Bspline_kernel(
            sqrt((GlobalGs[pIdx] * diff).squaredNorm() /  (IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx])), 
            (IS_CONST_RADIUS ? Sigma : GlobalSigma[pIdx]));
    }

    return (GlobalDeterminant[pIdx] * k_value); //(IS_CONST_RADIUS ? Radius3 : GlobalRadius3[pIdx]) * 
}

inline Eigen::Vector3f Evaluator::AnisotropicInterpolateGrad(const int pIdx, const Eigen::Vector3f diff)
{
    Eigen::Vector3f grad = poly6_gradient_kernel(
        (GlobalGs[pIdx] * diff).squaredNorm(), 
        // diff.squaredNorm(), 
        (IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx]), 
        (IS_CONST_RADIUS ? Sigma : GlobalSigma[pIdx]), (GlobalGs[pIdx] * diff));
    return (GlobalDeterminant[pIdx] * grad);    //(IS_CONST_RADIUS ? Radius3 : GlobalRadius3[pIdx]) * 
}

inline void Evaluator::compute_xMeans(int pIdx, std::vector<int> temp_neighbors, std::vector<int> &neighbors, int &closer_neighbor, Eigen::Vector3f &xMean)
{
    float pR, pD2;
    pR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(pIdx);
    // pD = SmoothFactor * pR;
    pD2 = IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx];
    float nR, nD, nD2, nI, nI2, d, d2, wj, wSum = 0;
    for (int nIdx : temp_neighbors)
    {
        if (nIdx == pIdx)
            continue;
        nR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(nIdx);
        nD = _SMOOTH_FACTOR * nR * 1.6;
        nD2 = nD * nD;
        nI = nR * _NEIGHBOR_FACTOR;
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
    if (_USE_XMEAN && !USE_POLY6)
    {
        if (wSum > 0)
        {
            xMean /= wSum;
            xMean = (GlobalPoses->at(pIdx)) * (1 - _XMEAN_DELTA) + xMean * _XMEAN_DELTA;
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

inline void Evaluator::compute_G_ours(int pIdx, Eigen::Vector3f xMean, std::vector<int> neighbors, Eigen::Matrix3f &G)
{
    // const float invH = 1/(_SMOOTH_FACTOR * (IS_CONST_RADIUS ? Radius : GlobalRadius->at(pIdx)));
    // const float invH = 1/_SMOOTH_FACTOR;
    const float invH = 0.5;


    float wSum = 0, d, wj;
    float nR, nI, nI2, d2;

    Eigen::Vector3f wd = Eigen::Vector3f::Zero();
    Eigen::Matrix3f cov = Eigen::Matrix3f::Identity();
    cov += Eigen::DiagonalMatrix<float, 3>(invH, invH, invH);
    // cov += Eigen::DiagonalMatrix<float, 3>(
    //     IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx],
    //     IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx], 
    //     IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx]);
    wSum = 0.0f;
    float pR, pD2, pI, pI2;
    pR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(pIdx);
    // pD = SmoothFactor * pR;
    // pD2 = IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx];
    pI = pR * _NEIGHBOR_FACTOR;
    pI2 = pI * pI;
    for (int nIdx : neighbors)
    {
        nR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(nIdx);
        nI = nR * _NEIGHBOR_FACTOR;
        nI2 = nI * nI;
        d2 = (((GlobalPoses->at(nIdx))) - ((GlobalPoses->at(pIdx)))).squaredNorm();
        if (d2 >= nI2)  //pI2
        {
            continue;
        }
        d = (GlobalPoses->at(pIdx) - GlobalxMeans.at(nIdx)).norm();
        wj = wij(d, nI);    //pI
        wSum += wj;
        wd = ((GlobalxMeans.at(nIdx))) - xMean;
        cov += (wd * wd.transpose()) * wj;
    }
    cov /= wSum;

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    Eigen::Matrix3f u = svd.matrixU();
    Eigen::Vector3f w = svd.singularValues();
    Eigen::Matrix3f v = svd.matrixV();
    
    w = Eigen::Vector3f(w.array().abs());
    const float maxSingularVal = w.maxCoeff();
    // const float minSingularVal = w.minCoeff();

    // if ((maxSingularVal / minSingularVal) > M && (!GlobalSplash[pIdx]))
    // {
    //     GlobalSurface[pIdx] = true;
    // }

    const float kr = 4.0;
    w[0] = std::max(w[0], maxSingularVal / kr);  // * invH
    w[1] = std::max(w[1], maxSingularVal / kr);  // * invH
    w[2] = std::max(w[2], maxSingularVal / kr);  // * invH

    Eigen::Matrix3f invSigma = w.asDiagonal().inverse();
    // Compute G
    const float scale =
        std::pow(w[0] * w[1] * w[2], 1.0 / 3.0);  // volume preservation
    G = (v * invSigma * u.transpose()) * scale * invH;//
    // float detC = cov.determinant();
    // float detG = G.determinant();
}

inline void Evaluator::compute_G_Yus(int pIdx, Eigen::Vector3f xMean, std::vector<int> neighbors, Eigen::Matrix3f &G)
{
    const float invH = 1/(_SMOOTH_FACTOR * (IS_CONST_RADIUS ? Radius : GlobalRadius->at(pIdx)));
    // const float invH = 1/_SMOOTH_FACTOR;
    // const float invH = 0.5;


    float wSum = 0, d, wj;
    float nR, nI, nI2, d2;

    Eigen::Vector3f wd = Eigen::Vector3f::Zero();
    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    // cov += Eigen::DiagonalMatrix<float, 3>(invH, invH, invH);
    cov += Eigen::DiagonalMatrix<float, 3>(
        IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx],
        IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx], 
        IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx]);
    wSum = 0.0f;
    float pR, pD2, pI, pI2;
    pR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(pIdx);
    // pD = SmoothFactor * pR;
    // pD2 = IS_CONST_RADIUS ? Influnce2 : GlobalInflunce2[pIdx];
    pI = pR * _NEIGHBOR_FACTOR;
    pI2 = pI * pI;
    for (int nIdx : neighbors)
    {
        nR = IS_CONST_RADIUS ? Radius : GlobalRadius->at(nIdx);
        nI = nR * _NEIGHBOR_FACTOR;
        nI2 = nI * nI;
        d2 = (((GlobalxMeans.at(nIdx))) - ((GlobalxMeans.at(pIdx)))).squaredNorm();
        if (d2 >= nI2)  //pI2
        {
            continue;
        }
        d = (GlobalxMeans.at(pIdx) - GlobalxMeans.at(nIdx)).norm();
        wj = wij(d, nI);    //pI
        wSum += wj;
        wd = ((GlobalxMeans.at(nIdx))) - xMean;
        cov += (wd * wd.transpose()) * wj;
    }
    cov /= wSum;

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    Eigen::Matrix3f u = svd.matrixU();
    Eigen::Vector3f w = svd.singularValues();
    Eigen::Matrix3f v = svd.matrixV();
    
    w = Eigen::Vector3f(w.array().abs());
    const float maxSingularVal = w.maxCoeff();
    // const float minSingularVal = w.minCoeff();

    // if ((maxSingularVal / minSingularVal) > M && (!GlobalSplash[pIdx]))
    // {
    //     GlobalSurface[pIdx] = true;
    // }

    const float kr = 4.0, ks = 1e7;//;1
    w[0] = std::max(w[0], maxSingularVal / kr);  // * invH* ks
    w[1] = std::max(w[1], maxSingularVal / kr);  // * invH* ks
    w[2] = std::max(w[2], maxSingularVal / kr);  // * invH* ks

    Eigen::Matrix3f invSigma = w.asDiagonal().inverse();
    // Compute G
    const float scale =
        std::pow(w[0] * w[1] * w[2], 1.0 / 3.0);  // volume preservation
    G = (v * invSigma * u.transpose()) * scale;//* invH
    // float detC = cov.determinant();
    // float detG = G.determinant();
}

void Evaluator::compute_Gs_xMeans()
{
#pragma omp parallel for
    for (int pIdx = 0; pIdx < _GlobalParticlesNum; pIdx++)
    {
        std::vector<int> tempNeighbors;
        std::vector<int> neighbors;
        int closerNeigbors = 0;
        Eigen::Vector3f xMean = Eigen::Vector3f::Zero();
        Eigen::Matrix3f G = Eigen::Matrix3f::Zero();
        if (IS_CONST_RADIUS)
        {
            _hashgrid->GetPIdxList((GlobalPoses->at(pIdx)), tempNeighbors);
        } else {
            // _searcher->GetNeighbors((GlobalPoses->at(pIdx)), tempNeighbors);
        }
        if (tempNeighbors.size() <= 2)
        {
            G = Eigen::DiagonalMatrix<float, 3>(1.0, 1.0, 1.0);
            // GlobalSplash[pIdx] = true;
            if (USE_ANI)
            {
                GlobalGs[pIdx] = Eigen::Matrix3f(G);
                GlobalDeterminant[pIdx] = G.determinant();
            }
            GlobalxMeans[pIdx] = GlobalPoses->at(pIdx);
            continue;
        }

        compute_xMeans(pIdx, tempNeighbors, neighbors, closerNeigbors, xMean);

        if (neighbors.size() <= 2)
        {
            G = Eigen::DiagonalMatrix<float, 3>(1.0, 1.0, 1.0);
            // GlobalSplash[pIdx] = true;
            if (USE_ANI)
            {
                GlobalGs[pIdx] = Eigen::Matrix3f(G);
                GlobalDeterminant[pIdx] = G.determinant();
            }
            GlobalxMeans[pIdx] = Eigen::Vector3f(xMean);
            continue;
        }
        // if (closerNeigbors < 1)
        // {
        //     G = Eigen::DiagonalMatrix<float, 3>(1.0, 1.0, 1.0);
        //     GlobalSplash[pIdx] = true;
        // } 
        // else
        // {
        //     if (USE_ANI)
        //     {
        //         compute_G_ours(pIdx, xMean, neighbors, G);
        //     }
        // }
        
        GlobalxMeans[pIdx] = Eigen::Vector3f(xMean);
        // if (USE_ANI)
        // {
        //     GlobalGs[pIdx] = Eigen::Matrix3f(G);
        //     GlobalDeterminant[pIdx] = G.determinant();
        // }
    }

#pragma omp parallel for
    for (int pIdx = 0; pIdx < _GlobalParticlesNum; pIdx++)
    {
        if (CheckSplash(pIdx))
        {
            continue;
        }
        std::vector<int> tempNeighbors;
        Eigen::Matrix3f G = Eigen::Matrix3f::Identity();
        if (IS_CONST_RADIUS)
        {
            _hashgrid->GetPIdxList((GlobalPoses->at(pIdx)), tempNeighbors);
        } else {
            // _searcher->GetNeighbors((GlobalPoses->at(pIdx)), tempNeighbors);
        }


        if (USE_ANI)
        {
            if (USE_POLY6)
            {
                compute_G_ours(pIdx, GlobalxMeans[pIdx], tempNeighbors, G);
            } else {
                compute_G_Yus(pIdx, GlobalxMeans[pIdx], tempNeighbors, G);
            }
            GlobalGs[pIdx] = Eigen::Matrix3f(G);
            GlobalDeterminant[pIdx] = G.determinant();
        }
    }
}

inline float Evaluator::wij(float d, float h)
{
	if (d < h)
	{
        if (d < h / 2) {    //(8 / pow(h, 3)) * inv_pi * 
            return (1 - (6 * pow(d, 2) / pow(h, 3) * (h - d)));
        } else {
            return (2 / pow(h, 3) * pow(h - d, 3));
        }
		// return (1.0 - pow(d / h, 3));
	}
	else
	{
		return 0.0;
	}
}

