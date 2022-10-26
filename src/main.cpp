#include <iostream>
#include <fstream>
#include "global.h"
#include "iso_common.h"
#include "iso_method_ours.h"
#include "mctable.h"
#include "hash_grid.h"
#include "evaluator.h"
#include "timer.h"
#include "parse.h"

#include "json.hpp"

#include <stack>
#include <iterator>
#include <omp.h>

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

using namespace std;

string CASE_PATH = "F:\\BaiduNetdiskDownload\\inlet_csv\\";
vector<string> CSV_PATHES;
string CSV_PATH = "";
int CSV_TYPE = 0;
string OUTPUT_PREFIX = "";
string RECORD_PREFIX = "";

int USE_DYNAMIC_THREADS = 0;
int THREAD_NUMS = 16;

bool LOAD_RECORD = false;
bool NEED_RECORD = false;
int RECORD_STEP = 50000;

static vector<vect3f> GlobalParticles;
static vector<float> GlobalDensity;
static vector<float> GlobalMass;

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
	ifstream inJSONFile(CASE_PATH + "//controlData.json", fstream::in);

	if (inJSONFile.good())
	{
		inJSONFile >> readInJSON;
		inJSONFile.close();
		const string filePath = readInJSON.at("CSV_PATH");
		string csv_pathes = filePath;
		parseString(&CSV_PATHES, csv_pathes, ",");
		NEED_RECORD = readInJSON.at("NEED_RECORD");
		CSV_TYPE = readInJSON.at("CSV_TYPE");
		P_RADIUS = readInJSON.at("P_RADIUS");
		RECORD_STEP = readInJSON.at("RECORD_STEP");
		OUTPUT_PREFIX = readInJSON.at("OUTPUT_PREFIX");
		RECORD_PREFIX = readInJSON.at("RECORD_PREFIX");

		INFLUENCE = 4 * P_RADIUS;
	}
	else
	{
		cout << "Cannot open case!" << endl;
	}
}

void loadParticles()
{
	ifstream ifn;
	ifn.open(CSV_PATH.c_str());

	GlobalParticles.clear();
	GlobalDensity.clear();
	GlobalMass.clear();
	GlobalParticlesNum = 0;
	MAX_VALUE = -1.0f;
	MIN_VALUE = 0.0f;

	string line;
	vector<float> elements;
	getline(ifn, line);
	getline(ifn, line);

	float xmax = -1e10f, ymax = -1e10f, zmax = -1e10f;
	float xmin = 1e10f, ymin = 1e10f, zmin = 1e10f;

	while (!line.empty())
	{
		elements.clear();
		string lines = line + ",";
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
			GlobalParticles.push_back(vect3f(elements[0], elements[1], elements[2]));
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
			GlobalParticles.push_back(vect3f(elements[1], elements[2], elements[3]));
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
			GlobalParticles.push_back(vect3f(elements[0], elements[1], elements[2]));
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

void writeFile(Mesh &m, string fn)
{
	FILE *f = fopen(fn.c_str(), "w");

#ifdef JOIN_VERTS
	map<TopoEdge, int> vt;

	int vert_num = 0;
	for (int i = 0; i < m.tris.size(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			map<TopoEdge, int>::iterator it = vt.find(m.topoTris[i][j]);
			if (it == vt.end())
			{
				vert_num++;

				vt[m.topoTris[i][j]] = vert_num;
				
				vect3f pos = m.tris[i][j];
				fprintf(f, "v %f %f %f\n", pos[0], pos[1], pos[2]);
			}
		}
	}

	set<vect<2, TopoEdge> > edges; // only for genus
	for (int i = 0; i < m.tris.size(); i++)
	{
		fprintf(f, "f %d %d %d\n", vt[m.topoTris[i][0]], vt[m.topoTris[i][1]], vt[m.topoTris[i][2]]);
		
		for (int j = 0; j < 3; j++)
		{
			vect<2, TopoEdge> e;
			e[0] = m.topoTris[i][j];
			e[1] = m.topoTris[i][(j+1) % 3];
			if (e[1] < e[0])
				swap(e[0], e[1]);
			edges.insert(e);
		}
	}

	// check genus
	int edge_num = edges.size();
	int face_num = m.tris.size();
	printf("verts = %d, edges = %d, faces = %d, genus = %d\n", vert_num, edge_num, face_num, vert_num - edge_num + face_num);
#else
	for (vect3f& p : m.vertices)
	{
		fprintf(f, "v %f %f %f\n", p[0], p[1], p[2]);
	}
	for (Triangle<int>& t : m.tris)
	{
		fprintf(f, "f %d %d %d\n", t.v[0], t.v[1], t.v[2]);
	}
#endif
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
	//double diameter = INFLUENCE;
	// Rect Box
	//for (size_t i = 0; i < 3; i++)
	//{
	//	int temp_depth_max = int(ceil(log2(ceil((BoundingBox[i * 2 + 1] - BoundingBox[i * 2]) / P_RADIUS))));
	//	
	//	double length = pow(2, temp_depth_max) * P_RADIUS;
	//	while (length - (BoundingBox[i * 2 + 1] - BoundingBox[i * 2]) < 2 * INFLUENCE)
	//	{
	//		temp_depth_max++;
	//		length = pow(2, temp_depth_max) * P_RADIUS;
	//	}
	//	if (DEPTH_MAX < temp_depth_max)
	//	{
	//		DEPTH_MAX = temp_depth_max;
	//	}
	//	length *= 0.99;
	//	double center = (BoundingBox[i * 2] + BoundingBox[i * 2 + 1]) / 2;
	//	BoundingBox[i * 2] = center - length / 2;
	//	BoundingBox[i * 2 + 1] = center + length / 2;
	//}
	//DEPTH_MIN = DEPTH_MAX - 1;
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
	//resizeLen *= 0.99;
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

	evaluator = new Evaluator(GlobalParticles, GlobalDensity, GlobalMass);
	last_temp_time = temp_time;
	temp_time = get_time();

	printf("Initialize Evaluator Time = %f \n", temp_time - last_temp_time);

	int max_density_index = std::distance(GlobalDensity.begin(), std::max_element(GlobalDensity.begin(), GlobalDensity.end()));
	evaluator->SingleEval(GlobalParticles[max_density_index], MAX_VALUE, *(vect3f*)NULL);

	ISO_VALUE = evaluator->RecommendIsoValue();
#ifdef USE_DMT
	initMCTable();
#endif // USE_DMT

	gen_iso_ours(); // pregenerate tree
	gen_iso_ours();
	gen_iso_ours();

	writeFile(g.ourMesh, CASE_PATH + "//" + OUTPUT_PREFIX + std::to_string(INDEX) + ".obj");
}

int main(int argc, char **argv)
{
	if (argc > 1)
	{
		CASE_PATH = string(argv[1]);
	}

	loadConfig();

	omp_set_dynamic(USE_DYNAMIC_THREADS);
	omp_set_num_threads(THREAD_NUMS);

	cout << "Threads: " << omp_get_num_threads() << endl;

	double frameStart = 0;

	for (string p: CSV_PATHES)
	{
		cout << "-=   Frame " << INDEX << "   =-" << endl;
		CSV_PATH = CASE_PATH + "//" + p;
		frameStart = get_time();
		::init();
		printf("-=  Frame %d total time= %f  =-\n", INDEX, get_time() - frameStart);
		INDEX++;
	}

	return 0;
}
