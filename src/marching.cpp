#include "marching.h"
#include "hash_grid.h"
#include "multi_level_researcher.h"
#include "evaluator.h"

#include "vtkXMLImageDataWriter.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include "vtkImageData.h"
#include "vtkNew.h"

UniformGrid::UniformGrid(const std::vector<Eigen::Vector3f> &particles, const std::vector<float> &radiuses)
{
    _GlobalParticles = particles;
	_GlobalParticlesNum = _GlobalParticles.size();
	_GlobalRadiuses = radiuses;
}

UniformGrid::UniformGrid(const std::vector<Eigen::Vector3f> &particles, float radius)
{
    _GlobalParticles = particles;
	_GlobalParticlesNum = _GlobalParticles.size();
	_RADIUS = radius;
}

void UniformGrid::loadRootBox()
{
	_BoundingBox[0] = _BoundingBox[2] = _BoundingBox[4] = FLT_MAX;
	_BoundingBox[1] = _BoundingBox[3] = _BoundingBox[5] = -FLT_MAX;
	for (const Eigen::Vector3f& p: _GlobalParticles)
	{
		if (p.x() < _BoundingBox[0]) _BoundingBox[0] = p.x();
		if (p.x() > _BoundingBox[1]) _BoundingBox[1] = p.x();
		if (p.y() < _BoundingBox[2]) _BoundingBox[2] = p.y();
		if (p.y() > _BoundingBox[3]) _BoundingBox[3] = p.y();
		if (p.z() < _BoundingBox[4]) _BoundingBox[4] = p.z();
		if (p.z() > _BoundingBox[5]) _BoundingBox[5] = p.z();
	}
}

void UniformGrid::resizeRootBoxConstR()
{
    float length, resizeLen, center;
    for (size_t i = 0; i < 3; i++)
    {
        center = (_BoundingBox[i * 2] + _BoundingBox[i * 2 + 1]) / 2;
        length = _BoundingBox[i*2+1] - _BoundingBox[i*2];
        resizeLen = ceil(length / _RADIUS) * _RADIUS + _RADIUS * 4;
        _BoundingBox[i*2] = center - resizeLen / 2;
        _BoundingBox[i*2+1] = center + resizeLen / 2;
        steps[i] = int(resizeLen / (_RADIUS));    // * 4
        dims[i] = steps[i] + 1;
    }
}

void UniformGrid::resizeRootBoxVarR()
{
    float length, resizeLen, center;
    for (size_t i = 0; i < 3; i++)
    {
        center = (_BoundingBox[i * 2] + _BoundingBox[i * 2 + 1]) / 2;
        length = _BoundingBox[i*2+1] - _BoundingBox[i*2];
        resizeLen = ceil(length / _searcher->getMaxRadius()) * _searcher->getMaxRadius() + _searcher->getMaxRadius() * 4;
        resizeLen = ceil(resizeLen / _searcher->getMinRadius()) * _searcher->getMinRadius() + _searcher->getMinRadius() * 4;
        _BoundingBox[i*2] = center - resizeLen / 2;
        _BoundingBox[i*2+1] = center + resizeLen / 2;
        steps[i] = int(resizeLen / _searcher->getMinRadius()) + 1;
        dims[i] = steps[i] + 1;
    }
}

void UniformGrid::gridSampling()
{
    Eigen::Vector3f minV(_BoundingBox[0], _BoundingBox[2], _BoundingBox[4]);
    Eigen::Vector3f maxV(_BoundingBox[1], _BoundingBox[3], _BoundingBox[5]);

#pragma omp parallel for
    for (size_t i = 0; i < _Scalars.size(); i++)    //
    {
        int x,y,z;
        z = (i / (dims[0] * dims[1]));
        y = (i % (dims[0] * dims[1])) / dims[0];
        x = (i % (dims[0] * dims[1])) % dims[0];
        // std::cout << i << ", " << x << ", " << y << ", " << z <<std::endl;
        _evaluator->SingleEval(
            Eigen::Vector3f(
                float(steps[0] - x) / float(steps[0]) * minV[0] + float(x) / float(steps[0]) * maxV[0],
                float(steps[1] - y) / float(steps[1]) * minV[1] + float(y) / float(steps[1]) * maxV[1],
                float(steps[2] - z) / float(steps[2]) * minV[2] + float(z) / float(steps[2]) * maxV[2]),
            _Scalars[i]
        );
    }
}

void UniformGrid::Run(float iso_value, std::string filename, std::string filepath) 
{
    timer t_total, t;
	float temp_time, last_temp_time;
    vtkNew<vtkImageData> grid;

	printf("-= Box =-\n");
    loadRootBox();

	printf("-= Build Neighbor Searcher =-\n");
    if (IS_CONST_RADIUS)
    {
        _hashgrid = std::make_shared<HashGrid>(&_GlobalParticles, _BoundingBox, _RADIUS, 4.0f);
    } else {
        _searcher = std::make_shared<MultiLevelSearcher>(&_GlobalParticles, _BoundingBox, &_GlobalRadiuses, 4.0f);
    }
	printf("   Build Neighbor Searcher Time = %f \n", t.elapsed());
    t.reset();

    printf("-= Initialize Evaluator =-\n");
	_evaluator = std::make_shared<Evaluator>(_hashgrid, _searcher, &_GlobalParticles, &_GlobalRadiuses, _RADIUS);
	_evaluator->setSmoothFactor(2.0f);
	IS_CONST_RADIUS ? _evaluator->CalculateMaxScalarConstR() : _evaluator->CalculateMaxScalarVarR();
    printf("   Max Scalar Value = %f\n", _evaluator->getMaxScalar());
	
	IS_CONST_RADIUS ? _evaluator->RecommendIsoValueConstR() : _evaluator->RecommendIsoValueVarR();
    printf("   Recommend Iso Value = %f\n", _evaluator->getIsoValue());
	_evaluator->compute_Gs_xMeans();
	printf("   Initialize Evaluator Time = %f \n", t.elapsed());

	if (USE_POLY6 && USE_ANI)
	{
		if (IS_CONST_RADIUS)
		{
			_hashgrid.reset();
			_hashgrid = std::make_shared<HashGrid>(&_GlobalParticles, _BoundingBox, _RADIUS, 2.0f);
			_evaluator->_hashgrid = _hashgrid;
		} else {
			_searcher.reset();
			_searcher = std::make_shared<MultiLevelSearcher>(&_GlobalParticles, _BoundingBox, &_GlobalRadiuses, 2.0f);
			_evaluator->_searcher = _searcher;
		}
	}

	printf("-= Resize Box =-\n");
	if (IS_CONST_RADIUS)
	{
		resizeRootBoxConstR();
	} else {
		resizeRootBoxVarR();
	}
    _Scalars.resize(dims[0] * dims[1] * dims[2]);
	printf("   Dimensions = %d, %d, %d\n", dims[0], dims[1], dims[2]);

	printMem();

	printf("-= Grid Sampling =-\n");
    t.reset();
    gridSampling();
	printf("   Grid Sampling Time = %f \n", t.elapsed());
	printf("   All End Time 1 = %f \n", t_total.elapsed());

	printMem();

    grid->SetDimensions(dims[0], dims[1], dims[2]);
    if (IS_CONST_RADIUS)
    {
        grid->SetSpacing(_RADIUS, _RADIUS, _RADIUS);
    } else {
        grid->SetSpacing(_searcher->getMinRadius(), _searcher->getMinRadius(), _searcher->getMinRadius());
    }
    grid->SetOrigin(_BoundingBox[0], _BoundingBox[2], _BoundingBox[4]);

    std::cout << grid->GetNumberOfPoints() << ", " << _Scalars.size() << std::endl;
    
    vtkNew<vtkFloatArray> scalars;
    scalars->SetName("scalars");
    scalars->SetNumberOfComponents(1);
    scalars->SetNumberOfTuples(_Scalars.size());
    for (size_t i = 0; i < _Scalars.size(); i++)
    {
        scalars->SetValue(i, _Scalars[i]);    // / _MAX_SCALAR
    }

    grid->GetPointData()->AddArray(scalars);

    vtkNew<vtkXMLImageDataWriter> writer;
    writer->SetFileName((filepath + "/" + filename + ".vti").c_str());
    writer->SetInputData(grid);
    writer->Write();
	printf("   All End Time 2 = %f \n", t_total.elapsed());
}