#include <iostream>
#include <fstream>
#include <omp.h>
#include "iso_common.h"
#include "surface_reconstructor.h"
#include "global.h"

#include "json.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include "unistd.h"
#endif

int OMP_USE_DYNAMIC_THREADS = 0;
int OMP_THREADS_NUM = 16;

// variants for test
std::string OUTPUT_PREFIX;
//bool NEED_RECORD;
//std::string RECORD_PREFIX;
//int RECORD_STEP;
std::vector<std::string> CSV_PATHES;
int CSV_TYPE;
float P_RADIUS;

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

void loadConfigJson(std::string& controlJsonPath)
{
    nlohmann::json readInJSON;
	std::ifstream inJSONFile(controlJsonPath.c_str(), std::fstream::in);

    if (inJSONFile.good())
    {
        inJSONFile >> readInJSON;
		inJSONFile.close();
		const std::string filePath = readInJSON.at("CSV_PATH");
		std::string csv_pathes = filePath;
		parseString(&CSV_PATHES, csv_pathes, ",");
		CSV_TYPE = readInJSON.at("CSV_TYPE");
		P_RADIUS = readInJSON.at("P_RADIUS");
		OUTPUT_PREFIX = readInJSON.at("OUTPUT_PREFIX");
		//NEED_RECORD = readInJSON.at("NEED_RECORD");
		//RECORD_STEP = readInJSON.at("RECORD_STEP");
		//RECORD_PREFIX = readInJSON.at("RECORD_PREFIX");
    }
    else
    {
		std::cout << "Cannot open case!" << std::endl;
    }
}

void loadParticlesFromCSV(std::string& csvPath, std::vector<Eigen::Vector3f>& particles, std::vector<float>& density, std::vector<float>& mass)
{
	std::ifstream ifn;
    ifn.open(csvPath.c_str());

    particles.clear();
	density.clear();
	mass.clear();

    std::string line;
	std::vector<float> elements;
	std::getline(ifn, line);
	std::getline(ifn, line);

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
			particles.push_back(Eigen::Vector3f(elements[0], elements[1], elements[2]));
			mass.push_back(elements[6]);
			density.push_back(elements[7] + 1000.0f);
			break;
		case 1:
			particles.push_back(Eigen::Vector3f(elements[1], elements[2], elements[3]));
			mass.push_back(1.0f);
			density.push_back(elements[0] + 1000.0f);
			break;
		case 2:
			particles.push_back(Eigen::Vector3f(elements[0], elements[1], elements[2]));
			mass.push_back(1.0f);
			density.push_back(elements[3] + 1000.0f);
			break;
		default:
			printf("Unknown type of csv format.");
			exit(1);
			break;
		}
		getline(ifn, line);
	}
}

void testWithCSV(std::string& csvDirPath)
{
	loadConfigJson(csvDirPath + "\\controlData.json");
	double frameStart = 0;
	int index = 0;
	std::vector<Eigen::Vector3f> particles;
    std::vector<float> density;
    std::vector<float> mass;
    for (std::string frame: CSV_PATHES)
	{
		Mesh mesh;
		std::cout << "-=   Frame " << index << " " << frame << "   =-" << std::endl;
		std::string csvPath = csvDirPath + "\\" + frame;
		frameStart = get_time();

		loadParticlesFromCSV(csvPath, particles, density, mass);

		SurfReconstructor constructor(particles, density, mass, mesh, P_RADIUS);
		constructor.Run();
		writeFile(mesh, csvDirPath + "\\" + OUTPUT_PREFIX + std::to_string(index) + ".obj");
		index++;
	}
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
	std::cout << "OMP Threads Num: " << OMP_THREADS_NUM << std::endl;

	omp_set_dynamic(OMP_USE_DYNAMIC_THREADS);
	omp_set_num_threads(OMP_THREADS_NUM);

	if (argc == 2)
	{
		std::string csvDirPath = std::string(argv[1]);
		testWithCSV(csvDirPath);
	} 
	else 
	{
		std::string csvDirPath = "F:\\BaiduNetdiskDownload\\inlet_csv";
		testWithCSV(csvDirPath);
	}

	return 0;
}
