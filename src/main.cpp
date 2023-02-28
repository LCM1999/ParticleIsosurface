#include <fstream>
#include <iostream>
#include <omp.h>
#include <regex>

#include "global.h"
#include "hdf5Utils.hpp"
#include "iso_common.h"
#include "iso_method_ours.h"
#include "json.hpp"
#include "recorder.h"
#include "surface_reconstructor.h"

#ifdef _WIN32
#include <windows.h>
#else
#include "unistd.h"
#endif

int OMP_USE_DYNAMIC_THREADS = 0;
int OMP_THREADS_NUM = 16;

bool IS_CONST_DENSITY = false;
bool IS_CONST_MASS = false;
bool IS_CONST_RADIUS = false;

// variants for test
short DATA_TYPE = 0;    // CSV:0, H5: 1
bool NEED_RECORD;
std::vector<std::string> DATA_PATHES;
float DENSITY = 0;
float MASS = 0;
float RADIUS = 0;

void writeFile(Mesh &m, std::string fn)
{
    FILE *f = fopen(fn.c_str(), "w");
    for (Eigen::Vector3f &p : m.vertices)
    {
        fprintf(f, "v %f %f %f\n", p[0], p[1], p[2]);
    }
    for (Triangle<int> &t : m.tris)
    {
        fprintf(f, "f %d %d %d\n", t.v[0], t.v[1], t.v[2]);
    }
    fclose(f);
}

void loadConfigJson(const std::string controlJsonPath)
{
    nlohmann::json readInJSON;
    std::ifstream inJSONFile(controlJsonPath.c_str(), std::fstream::in);

    if (inJSONFile.good())
    {
        inJSONFile >> readInJSON;
        inJSONFile.close();
        DATA_TYPE = readInJSON.at("DATA_TYPE");
        const std::string filePath = readInJSON.at("DATA_FILE");
        std::string data_pathes = filePath;
        parseString(&DATA_PATHES, data_pathes, ",");
        if (readInJSON.contains("DENSITY"))
        {
            DENSITY = readInJSON.at("DENSITY");
            IS_CONST_DENSITY = true;
        }
        if (readInJSON.contains("MASS"))
        {
            MASS = readInJSON.at("MASS");
            IS_CONST_MASS = true;
        }
        if (readInJSON.contains("RADIUS"))
        {
            RADIUS = readInJSON.at("RADIUS");
            IS_CONST_RADIUS = true;
        }
        NEED_RECORD = readInJSON.at("NEED_RECORD");
    }
    else
    {
        std::cout << "Cannot open case!" << std::endl;
    }
}

void loadParticlesFromCSV(std::string &csvPath,
                          std::vector<Eigen::Vector3f> &particles,
                          std::vector<float>* densities, std::vector<float>* masses, std::vector<float>* radiuses)
{
    std::ifstream ifn;
    ifn.open(csvPath.c_str());

    int densityIdx = -1, massIdx = -1, radiusIdx = -1, xIdx = -1, yIdx = -1, zIdx = -1;

    particles.clear();

    std::string line;
    std::vector<std::string> titles;
    std::vector<float> elements;
    std::getline(ifn, line);
    parseString(&titles, line, ",");
    if (!IS_CONST_DENSITY)
    {
        densityIdx = std::distance(titles.begin(), 
        std::find_if(titles.begin(), titles.end(), [&](const std::string &title) {
            std::regex reg(".*density.*", std::regex::icase);
            return std::regex_match(title, reg);
        }));
    }
    if (!IS_CONST_MASS)
    {
        massIdx = std::distance(titles.begin(), 
        std::find_if(titles.begin(), titles.end(), [&](const std::string &title) {
            std::regex reg(".*mass.*", std::regex::icase);
            return std::regex_match(title, reg);
        }));
    }
    if (!IS_CONST_RADIUS)
    {
        radiusIdx = std::distance(titles.begin(),
        std::find_if(titles.begin(), titles.end(), 
        [&](const std::string &title) {
            std::regex reg(".*radius.*", std::regex::icase);
            return std::regex_match(title, reg);
        }));
    }
    xIdx = std::distance(titles.begin(), 
    std::find_if(titles.begin(), titles.end(), [&](const std::string &title) {
        std::regex reg(".*(Position|Point).*(x|0).*", std::regex::icase);
        return std::regex_match(title, reg);
    }));
    yIdx = std::distance(titles.begin(), 
    std::find_if(titles.begin(), titles.end(), [&](const std::string &title) {
        std::regex reg(".*(Position|Point).*(y|1).*", std::regex::icase);
        return std::regex_match(title, reg);
    }));
    zIdx = std::distance(titles.begin(), 
    std::find_if(titles.begin(), titles.end(), [&](const std::string &title) {
        std::regex reg(".*(Position|Point).*(z|2).*", std::regex::icase);
        return std::regex_match(title, reg);
    }));

    std::getline(ifn, line);

    while (!line.empty())
    {
        elements.clear();
        parseStringToElements(&elements, line, ",");
        particles.push_back(Eigen::Vector3f(elements[xIdx], elements[yIdx], elements[zIdx]));
        if (!IS_CONST_DENSITY)
        {
            densities->push_back(elements[densityIdx] + 1000.0f);
        }
        if (!IS_CONST_MASS)
        {
            masses->push_back(elements[massIdx]);
        }
        if (!IS_CONST_RADIUS)
        {
            radiuses->push_back(elements[radiusIdx]);
        }

        // switch (CSV_TYPE)
        // {
        //     case 0:
        //         particles.push_back(
        //             Eigen::Vector3f(elements[0], elements[1], elements[2]));
        //         masses.push_back(elements[6]);
        //         densities.push_back(elements[7] + 1000.0f);
        //         break;
        //     case 1:
        //         particles.push_back(
        //             Eigen::Vector3f(elements[1], elements[2], elements[3]));
        //         masses.push_back(1.0f);
        //         densities.push_back(elements[0] + 1000.0f);
        //         break;
        //     case 2:
        //         particles.push_back(
        //             Eigen::Vector3f(elements[0], elements[1], elements[2]));
        //         masses.push_back(1.0f);
        //         densities.push_back(elements[3] + 1000.0f);
        //         break;
        //     default:
        //         printf("Unknown type of csv format.");
        //         exit(1);
        //         break;
        // }
        getline(ifn, line);
    }
}

void run(std::string &dataDirPath)
{
    loadConfigJson(dataDirPath + "/controlData.json");
    double frameStart = 0;
    int index = 0;
    std::vector<Eigen::Vector3f> particles;
    std::vector<float>* densities = (IS_CONST_DENSITY ? nullptr : new std::vector<float>());
    std::vector<float>* masses = (IS_CONST_MASS ? nullptr : new std::vector<float>());
    std::vector<float>* radiuses = (IS_CONST_RADIUS ? nullptr : new std::vector<float>());
    for (const std::string frame : DATA_PATHES)
    {
        Mesh mesh;
        std::cout << "-=   Frame " << index << " " << frame << "   =-"
                  << std::endl;
        std::string dataPath = dataDirPath + "/" + frame;
        frameStart = get_time();

        switch (DATA_TYPE)
        {
        case 0:
            loadParticlesFromCSV(dataPath, particles, densities, masses, radiuses);
            break;
        case 1:
            readShonDyParticleData(dataPath, particles, densities, masses, radiuses);
            break;
        default:
            printf("ERROR: Unknown DATA TYPE;");
            exit(1);
        }

        printf("Particles Number = %zd\n", particles.size());

        SurfReconstructor constructor(particles, densities, masses, radiuses, mesh, DENSITY, MASS, RADIUS, 3.1f);
        Recorder recorder(dataDirPath, frame.substr(0, frame.size() - 4),
                          &constructor);
        constructor.Run();

        if (NEED_RECORD)
        {
            recorder.RecordProgress();
            recorder.RecordParticles();
        }

        writeFile(mesh,
                  dataDirPath + "/" + frame.substr(0, frame.size() - 4) + ".obj");
        index++;
    }
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    std::cout << "SYSINFO Threads: " << sysinfo.dwNumberOfProcessors
              << std::endl;
    OMP_THREADS_NUM = sysinfo.dwNumberOfProcessors;
#else
    OMP_THREADS_NUM = sysconf(_SC_NPROCESSORS_CONF);
#endif
    std::cout << "OMP Threads Num: " << OMP_THREADS_NUM << std::endl;

    omp_set_dynamic(OMP_USE_DYNAMIC_THREADS);
    omp_set_num_threads(OMP_THREADS_NUM);

    std::cout << std::to_string(argc) << std::endl;

    if (argc == 2)
    {
        std::string h5DirPath = std::string(argv[1]);
        std::cout << h5DirPath << std::endl;
        run(h5DirPath);
    }
    else
    {
        std::string dataDirPath =
            //"E:/data/multiR/mr_csv";
            "E:/data/vtk/csv";
        run(dataDirPath);
    }

    return 0;
}
