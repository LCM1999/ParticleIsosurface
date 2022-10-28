#include <iostream>
#include <fstream>
#include <stack>
#include <iterator>
#include <omp.h>
#include "global.h"
#include "iso_common.h"
#include "timer.h"

#include "json.hpp"
#include "hdf5.h"

#ifdef _WIN32
#include <windows.h>
#else
#include "unistd.h"
#endif


int OVERSAMPLE_QEF = 4;
float BORDER = (1.0 / 4096.0);
int DEPTH_MAX = 6; // 7
int DEPTH_MIN = 5; // 4
int FIND_ROOT_DEPTH = 0;
float P_RADIUS = 0.0f;	// 0.00025f;
float INFLUENCE = 2 * P_RADIUS;
float ISO_VALUE = 37.5f;

int KERNEL_TYPE = 0;

float MAX_VALUE = -1.0f;
float MIN_VALUE = 0.0f;
float BAD_QEF = 0.0001f;
float FLATNESS = 0.99f;
int MIN_NEIGHBORS_NUM = 5;

bool USE_ANI = true;
bool USE_XMEAN = true;
float XMEAN_DELTA = 0.3;

float TOLERANCE = 1e-8;
float RATIO_TOLERANCE = 0.15f;
float LOW_MESH_QUALITY = 0.17320508075688772935274463415059;
float MESH_TOLERANCE = 1e4;
float VARIANCE_TOLERANCE = 100.0;

HashGrid* hashgrid = NULL;
Evaluator* evaluator = NULL;

std::string CASE_PATH = "F:\\BaiduNetdiskDownload\\inlet_csv\\";
std::vector<std::string> CSV_PATHES;
std::string CSV_PATH = "";
int CSV_TYPE = 0;
std::string OUTPUT_PREFIX = "";
std::string RECORD_PREFIX = "";

int OMP_USE_DYNAMIC_THREADS = 0;
int OMP_THREADS_NUM = 16;

bool LOAD_RECORD = false;
bool NEED_RECORD = false;
int RECORD_STEP = 50000;

static std::vector<Eigen::Vector3f> GlobalParticles;
static std::vector<float> GlobalDensity;
static std::vector<float> GlobalMass;

static int GlobalParticlesNum = 0;

double BoundingBox[6] = {0.0f};
double RootHalfLength;
float RootCenter[3] = {0.0f};

int INDEX = 0;
int STATE = 0;

void loadConfig()
{
	// read parameters from json file
	nlohmann::json readInJSON;
	std::ifstream inJSONFile(CASE_PATH + "//controlData.json", std::fstream::in);

	if (inJSONFile.good())
	{
		inJSONFile >> readInJSON;
		inJSONFile.close();
		const std::string filePath = readInJSON.at("CSV_PATH");
		std::string csv_pathes = filePath;
		parseString(&CSV_PATHES, csv_pathes, ",");
		CSV_TYPE = readInJSON.at("CSV_TYPE");
		P_RADIUS = readInJSON.at("P_RADIUS");
		NEED_RECORD = readInJSON.at("NEED_RECORD");
		RECORD_STEP = readInJSON.at("RECORD_STEP");
		OUTPUT_PREFIX = readInJSON.at("OUTPUT_PREFIX");
		RECORD_PREFIX = readInJSON.at("RECORD_PREFIX");

		INFLUENCE = 4 * P_RADIUS;
	}
	else
	{
		std::cout << "Cannot open case!" << std::endl;
	}
}

void loadParticles()
{
	std::ifstream ifn;
	ifn.open(CSV_PATH.c_str());

	GlobalParticles.clear();
	GlobalDensity.clear();
	GlobalMass.clear();
	GlobalParticlesNum = 0;
	MAX_VALUE = -1.0f;
	MIN_VALUE = 0.0f;

	std::string line;
	std::vector<float> elements;
	std::getline(ifn, line);
	std::getline(ifn, line);

	float xmax = -1e10f, ymax = -1e10f, zmax = -1e10f;
	float xmin = 1e10f, ymin = 1e10f, zmin = 1e10f;

	while (!line.empty())
	{
		elements.clear();
		std::string lines = line + ",";
		size_t pos = lines.find(",");

		while (pos != lines.npos)
		{
			elements.push_back(atof(lines.substr(0, pos).c_str()));
			lines = lines.substr(pos + 1, lines.size());
			pos = lines.find(",");
		}
		switch (CSV_TYPE)
		{
		case 0:
			if (elements[0] > xmax) { xmax = elements[0]; }
			if (elements[1] > ymax) { ymax = elements[1]; }
			if (elements[2] > zmax) { zmax = elements[2]; }
			if (elements[0] < xmin) { xmin = elements[0]; }
			if (elements[1] < ymin) { ymin = elements[1]; }
			if (elements[2] < zmin) { zmin = elements[2]; }
			GlobalParticles.push_back(Eigen::Vector3f(elements[0], elements[1], elements[2]));
			GlobalMass.push_back(elements[6]);
			GlobalDensity.push_back(elements[7] + 1000.0f);
			break;
		case 1:
			if (elements[1] > xmax) { xmax = elements[1]; }
			if (elements[2] > ymax) { ymax = elements[2]; }
			if (elements[3] > zmax) { zmax = elements[3]; }
			if (elements[1] < xmin) { xmin = elements[1]; }
			if (elements[2] < ymin) { ymin = elements[2]; }
			if (elements[3] < zmin) { zmin = elements[3]; }
			GlobalParticles.push_back(Eigen::Vector3f(elements[1], elements[2], elements[3]));
			GlobalMass.push_back(1.0f);
			GlobalDensity.push_back(elements[0] + 1000.0f);
			break;
		case 2:
			if (elements[0] > xmax) { xmax = elements[0]; }
			if (elements[1] > ymax) { ymax = elements[1]; }
			if (elements[2] > zmax) { zmax = elements[2]; }
			if (elements[0] < xmin) { xmin = elements[0]; }
			if (elements[1] < ymin) { ymin = elements[1]; }
			if (elements[2] < zmin) { zmin = elements[2]; }
			GlobalParticles.push_back(Eigen::Vector3f(elements[0], elements[1], elements[2]));
			GlobalMass.push_back(1.0f);
			GlobalDensity.push_back(elements[3] + 1000.0f);
			break;
		default:
			printf("Unknown type of csv format.");
			exit(1);
			break;
		}
		
		GlobalParticlesNum++;
		getline(ifn, line);
	}
	BoundingBox[0] = xmin;
	BoundingBox[1] = xmax;
	BoundingBox[2] = ymin;
	BoundingBox[3] = ymax;
	BoundingBox[4] = zmin;
	BoundingBox[5] = zmax;
}

void writeFile(Mesh &m, std::string fn)
{
	FILE *f = fopen(fn.c_str(), "w");
	for (Eigen::Vector3f& p : m.vertices)
	{
		fprintf(f, "v %f %f %f\n", p[0], p[1], p[2]);
	}
	for (Triangle<int>& t : m.tris)
	{
		fprintf(f, "f %d %d %d\n", t.v[0], t.v[1], t.v[2]);
	}
	fclose(f);
}

void init()
{
	g.ourRoot = nullptr;
	g.ourMesh.reset();
	STATE = 0;
	// generate surfaces
	double time_all_start = get_time();
	double temp_time, last_temp_time;
	printf("-= Load Particles =-\n");

	loadParticles();
	last_temp_time = time_all_start;
	temp_time = get_time();

	printf("Load Particles Time = %f \n", temp_time - last_temp_time);
	// Cube Box
	double maxLen, resizeLen;
	maxLen = (std::max)({ 
		(BoundingBox[1] - BoundingBox[0]) , 
		(BoundingBox[3] - BoundingBox[2]) , 
		(BoundingBox[5] - BoundingBox[4]) });
	DEPTH_MAX = int(ceil(log2(ceil(maxLen / P_RADIUS))));
	resizeLen = pow(2, DEPTH_MAX) * P_RADIUS;
	while (resizeLen - maxLen < (2 * INFLUENCE))
	{
		DEPTH_MAX++;
		resizeLen = pow(2, DEPTH_MAX) * P_RADIUS;
	}
	resizeLen *= 0.99;
	RootHalfLength = resizeLen / 2;
	for (size_t i = 0; i < 3; i++)
	{
		double center = (BoundingBox[i * 2] + BoundingBox[i * 2 + 1]) / 2;
		BoundingBox[i * 2] = center - resizeLen / 2;
		BoundingBox[i * 2 + 1] = center + resizeLen / 2;
		RootCenter[i] = center;
	}
	
	//DEPTH_MIN = DEPTH_MAX - 2;
	DEPTH_MIN = (DEPTH_MAX - int(DEPTH_MAX / 3));

	printf("-= Build Hash Grid =-\n");

	hashgrid = new HashGrid(GlobalParticles, BoundingBox, INFLUENCE);
	last_temp_time = temp_time;
	temp_time = get_time();

	printf("Build Hash Grid Time = %f \n", temp_time - last_temp_time);

	printf("-= Initialize Evaluator =-\n");

	evaluator = new Evaluator(&GlobalParticles, &GlobalDensity, &GlobalMass);
	last_temp_time = temp_time;
	temp_time = get_time();

	printf("Initialize Evaluator Time = %f \n", temp_time - last_temp_time);

	int max_density_index = std::distance(GlobalDensity.begin(), std::max_element(GlobalDensity.begin(), GlobalDensity.end()));
	evaluator->SingleEval(GlobalParticles[max_density_index], MAX_VALUE, *(Eigen::Vector3f*)NULL);

	ISO_VALUE = evaluator->RecommendIsoValue();

	gen_iso_ours(); // generate tree
	gen_iso_ours();	// check tree
	gen_iso_ours();	// generate mesh

	writeFile(g.ourMesh, CASE_PATH + "//" + OUTPUT_PREFIX + std::to_string(INDEX) + ".obj");
}

int main(int argc, char **argv)
{

#ifdef _WIN32
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	std::cout << "SYSINFO Threads: " << sysinfo.dwNumberOfProcessors << std::endl;
	OMP_THREADS_NUM = sysinfo.dwNumberOfProcessors;
#else
	OMP_THREADS_NUM = sysconf(_SC_NPROCESSORS_CONF);
#endif
	std::cout << "Threads: " << OMP_THREADS_NUM << std::endl;

	if (argc > 1)
	{
		CASE_PATH = std::string(argv[1]);
	}

	loadConfig();

	omp_set_dynamic(OMP_USE_DYNAMIC_THREADS);
	omp_set_num_threads(OMP_THREADS_NUM);

	double frameStart = 0;

	for (std::string p: CSV_PATHES)
	{
		std::cout << "-=   Frame " << INDEX << "   =-" << std::endl;
		CSV_PATH = CASE_PATH + "//" + p;
		frameStart = get_time();
		::init();
		printf("-=  Frame %d total time= %f  =-\n", INDEX, get_time() - frameStart);
		INDEX++;
	}

	return 0;
}
