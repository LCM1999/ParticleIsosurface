#include <fstream>
#include <iostream>
#include <omp.h>

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

// variants for test
bool NEED_RECORD;
std::vector<std::string> H5_PATHES;
float P_RADIUS;

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
        const std::string filePath = readInJSON.at("H5_PATH");
        std::string h5_pathes = filePath;
        parseString(&H5_PATHES, h5_pathes, ",");
        P_RADIUS = readInJSON.at("P_RADIUS");
        NEED_RECORD = readInJSON.at("NEED_RECORD");
    }
    else
    {
        std::cout << "Cannot open case!" << std::endl;
    }
}

// void loadParticlesFromCSV(std::string &csvPath,
//                           std::vector<Eigen::Vector3f> &particles,
//                           std::vector<float> &density, std::vector<float>
//                           &mass)
// {
//     std::ifstream ifn;
//     ifn.open(csvPath.c_str());

//     particles.clear();
//     density.clear();
//     mass.clear();

//     std::string line;
//     std::vector<float> elements;
//     std::getline(ifn, line);
//     std::getline(ifn, line);

//     while (!line.empty())
//     {
//         elements.clear();
//         std::string lines = line + ",";
//         size_t pos = lines.find(",");

//         while (pos != lines.npos)
//         {
//             elements.push_back(atof(lines.substr(0, pos).c_str()));
//             lines = lines.substr(pos + 1, lines.size());
//             pos = lines.find(",");
//         }
//         switch (CSV_TYPE)
//         {
//             case 0:
//                 particles.push_back(
//                     Eigen::Vector3f(elements[0], elements[1], elements[2]));
//                 mass.push_back(elements[6]);
//                 density.push_back(elements[7] + 1000.0f);
//                 break;
//             case 1:
//                 particles.push_back(
//                     Eigen::Vector3f(elements[1], elements[2], elements[3]));
//                 mass.push_back(1.0f);
//                 density.push_back(elements[0] + 1000.0f);
//                 break;
//             case 2:
//                 particles.push_back(
//                     Eigen::Vector3f(elements[0], elements[1], elements[2]));
//                 mass.push_back(1.0f);
//                 density.push_back(elements[3] + 1000.0f);
//                 break;
//             default:
//                 printf("Unknown type of csv format.");
//                 exit(1);
//                 break;
//         }
//         getline(ifn, line);
//     }
// }

void testWithH5(std::string &h5DirPath)
{
    loadConfigJson(h5DirPath + "/controlData.json");
    double frameStart = 0;
    int index = 0;
    std::vector<Eigen::Vector3f> particles;
    std::vector<float> densities;
    std::vector<float> masses;
    for (const std::string frame : H5_PATHES)
    {
        Mesh mesh(P_RADIUS);
        std::cout << "-=   Frame " << index << " " << frame << "   =-"
                  << std::endl;
        std::string h5Path = h5DirPath + "/" + frame;
        frameStart = get_time();

        readShonDyParticleData(h5Path, particles, densities, masses);

        printf("%d\n", particles.size());

        SurfReconstructor constructor(particles, densities, masses, mesh,
                                      P_RADIUS);
        Recorder recorder(h5DirPath, frame.substr(0, frame.size() - 4),
                          &constructor);
        constructor.Run();

        if (NEED_RECORD)
        {
            recorder.RecordProgress();
            recorder.RecordParticles();
        }

        writeFile(mesh,
                  h5DirPath + "/" + frame.substr(0, frame.size() - 4) + ".obj");
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
        testWithH5(h5DirPath);
    }
    else
    {
        std::string h5DirPath =
            "E:/data/water_manage/water_manage/water_manage";
        testWithH5(h5DirPath);
    }

    return 0;
}
