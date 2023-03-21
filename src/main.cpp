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

bool IS_CONST_RADIUS = false;

// variants for test
short DATA_TYPE = 0;    // CSV:0, H5: 1
bool NEED_RECORD;
std::vector<std::string> DATA_PATHES;
float RADIUS = 0;
float SCALE = 1;
float FLATNESS = 0.99;
float INF_FACTOR = 4.0;

void writeFile(Mesh &m, std::string fn)
{
    FILE *f = fopen(fn.c_str(), "w");
    for (auto &p : m.vertices)
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
        if (readInJSON.contains("RADIUS"))
        {
            RADIUS = readInJSON.at("RADIUS");
            IS_CONST_RADIUS = true;
        } else {
            IS_CONST_RADIUS = false;
        }
        if (readInJSON.contains("SCALE"))
        {
            SCALE = readInJSON.at("SCALE");
        }
        if (readInJSON.contains("FLATNESS"))
        {
            FLATNESS = readInJSON.at("FLATNESS");
        }
        if (readInJSON.contains("INF_FACTOR"))
        {
            INF_FACTOR = readInJSON.at("INF_FACTOR");
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
                          std::vector<float>* radiuses)
{
    std::ifstream ifn;
    ifn.open(csvPath.c_str());

    int radiusIdx = -1, xIdx = -1, yIdx = -1, zIdx = -1;

    particles.clear();

    std::string line;
    std::vector<std::string> titles;
    std::vector<float> elements;
    std::getline(ifn, line);
    replaceAll(line, "\"", "");
    parseString(&titles, line, ",");
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
        std::regex reg("^(.*(:|_|-)( *)(x|0)|x)$", std::regex::icase);
        return std::regex_match(title, reg);
    }));
    yIdx = std::distance(titles.begin(), 
    std::find_if(titles.begin(), titles.end(), [&](const std::string &title) {
        std::regex reg("^(.*(:|_|-)( *)(y|1)|y)$", std::regex::icase);
        return std::regex_match(title, reg);
    }));
    zIdx = std::distance(titles.begin(), 
    std::find_if(titles.begin(), titles.end(), [&](const std::string &title) {
        std::regex reg("^(.*(:|_|-)( *)(z|2)|z)$", std::regex_constants::ECMAScript | std::regex::icase);
        return std::regex_match(title, reg);
    }));

    printf("%d %d %d \n", xIdx, yIdx, zIdx);

    if (xIdx == titles.size() || yIdx == titles.size() || zIdx == titles.size())
    {
        printf("Error: Failed to locate axis column;");
        exit(1);
    }

    std::getline(ifn, line);

    while (!line.empty())
    {
        elements.clear();
        parseStringToElements(&elements, line, ",");
        particles.push_back(Eigen::Vector3f(elements[xIdx], elements[yIdx], elements[zIdx]));
        if (!IS_CONST_RADIUS)
        {
            radiuses->push_back(elements[radiusIdx] * SCALE);
        }
        getline(ifn, line);
    }
}

void run(std::string &dataDirPath)
{
    loadConfigJson(dataDirPath + "/controlData.json");
    double frameStart = 0;
    int index = 0;
    std::vector<Eigen::Vector3f> particles;
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
            loadParticlesFromCSV(dataPath, particles, radiuses);
            break;
        case 1:
            readShonDyParticleData(dataPath, particles, radiuses, SCALE);
            break;
        default:
            printf("ERROR: Unknown DATA TYPE;");
            exit(1);
        }

        printf("Particles Number = %zd\n", particles.size());

        SurfReconstructor constructor(particles, radiuses, mesh, RADIUS, FLATNESS, INF_FACTOR);
        Recorder recorder(dataDirPath, frame.substr(0, frame.size() - 4),
                          &constructor);
        constructor.Run();

        if (NEED_RECORD)
        {
            // recorder.RecordProgress();
            // recorder.RecordParticles();
            recorder.RecordFeatures();
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
            // "C:/Users/11379/Desktop/protein";
            "E:/data/multiR/mr_csv";
            // "E:/data/vtk/csv";
        run(dataDirPath);
    }

    return 0;
}
