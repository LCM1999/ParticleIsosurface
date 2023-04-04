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
#include "rply.h"

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
bool NEED_RECORD = false;
bool WITH_NORMAL;
std::vector<std::string> DATA_PATHES;
std::string OUTPUT_TYPE = "ply";
float RADIUS = 0;
float SCALE = 1;
float FLATNESS = 0.99;
float INF_FACTOR = 4.0;

void writeObjFile(Mesh &m, std::string fn)
{
    FILE *f = fopen(fn.c_str(), "w");
    for (auto &p : m.vertices)
    {
        fprintf(f, "v %f %f %f\n", p[0], p[1], p[2]);
    }
    for (Triangle &t : m.tris)
    {
        fprintf(f, "f %d %d %d\n", t.v[0], t.v[1], t.v[2]);
    }
    fclose(f);
}

int writePlyFile(Mesh& m, std::string fn)
{
    int num_vertices = int(m.verticesNum);
    int num_faces = int(m.trianglesNum);
    float version;
    p_ply ply = ply_create(fn.c_str(), PLY_DEFAULT, NULL, 0, NULL);
    if (!ply) 
        return 0;   

    ply_add_element(ply, "vertex", num_vertices);
    ply_add_scalar_property(ply, "x", PLY_FLOAT32);
    ply_add_scalar_property(ply, "y", PLY_FLOAT32);
    ply_add_scalar_property(ply, "z", PLY_FLOAT32);
    
    ply_add_element(ply, "face", num_faces);
    ply_add_list_property(ply, "vertex_indices", PLY_UCHAR, PLY_INT);

    if (WITH_NORMAL)
    {
        //TODO
    }

    if (!ply_write_header(ply))
        return 0;

    for (size_t i = 0; i < num_vertices; i++)
    {
        ply_write(ply, m.vertices[i].v[0]);
        ply_write(ply, m.vertices[i].v[1]);
        ply_write(ply, m.vertices[i].v[2]);
    }
    
    for (size_t i = 0; i < num_faces; i++)
    {
        ply_write(ply, 3);
        for (size_t j = 0; j < 3; j++)
        {
            ply_write(ply, m.tris[i].v[j] - 1);
        }
    }
    if (!ply_close(ply))
    {
        return 0;
    }
    
    return 1;
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
        if (readInJSON.contains("NEED_RECORD"))
        {
            NEED_RECORD = readInJSON.at("NEED_RECORD");
        }
        if (readInJSON.contains("OUTPUT_TYPE"))
        {
            OUTPUT_TYPE = readInJSON.at("OUTPUT_TYPE");
        }
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

        if ("ply" == OUTPUT_TYPE || "PLY" == OUTPUT_TYPE)
        {
            writePlyFile(mesh,
                  dataDirPath + "/" + frame.substr(0, frame.size() - 4) + ".ply");
        } else if ("obj" == OUTPUT_TYPE || "OBJ" == OUTPUT_TYPE) 
        {
            writeObjFile(mesh,
                  dataDirPath + "/" + frame.substr(0, frame.size() - 4) + ".obj");    
        } else {
            writePlyFile(mesh,
                  dataDirPath + "/" + frame.substr(0, frame.size() - 4) + ".ply");
        }        
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
            // "E:/data/multiR/mr_csv";
            "E:/data/vtk/csv";
        run(dataDirPath);
    }

    return 0;
}
