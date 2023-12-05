#include <fstream>
#include <iostream>
#include <filesystem>
#include <omp.h>
#include <regex>

#include "global.h"
// #include "hdf5Utils.hpp"
#include "iso_common.h"
#include "iso_method_ours.h"
#include "json.hpp"
#include "recorder.h"
#include "surface_reconstructor.h"
#include "rply.h"

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include "unistd.h"
#endif

int OMP_USE_DYNAMIC_THREADS = 0;
int OMP_THREADS_NUM = 16;

bool IS_CONST_RADIUS = false;
bool USE_ANI = true;

// variants for test
short DATA_TYPE = 0;    // CSV:0, H5: 1
bool NEED_RECORD = false;
int TARGET_FRAME = 0;
std::string PREFIX = "";
bool WITH_NORMAL;
std::vector<std::string> DATA_PATHES;
std::string OUTPUT_TYPE = "ply";
double RADIUS = 0;
float SMOOTH_FACTOR = 2.0;
float ISO_FACTOR = 1.9;
// bool USE_CUDA = false;
bool CALC_P_NORMAL = true;
bool GEN_SPLASH = true;

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
    std::cout << "Output Done" << std::endl;
}

int writePlyFile(Mesh& m, std::string fn)
{
    int num_vertices = int(m.verticesNum);
    int num_faces = int(m.trianglesNum);
    double version;
    p_ply ply = ply_create(fn.c_str(), PLY_DEFAULT, NULL, 0, NULL);
    if (!ply) 
        return 0;   

    ply_add_element(ply, "vertex", num_vertices);
    ply_add_scalar_property(ply, "x", PLY_double32);
    ply_add_scalar_property(ply, "y", PLY_double32);
    ply_add_scalar_property(ply, "z", PLY_double32);
    
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
    std::cout << "Output Done" << std::endl;
    return 1;
}

void loadConfigJson(std::string dataPath)
{
    nlohmann::json readInJSON;

    if (! std::filesystem::exists(dataPath + "/controlData.json"))
    {
        std::cout << "Error: cannot find ConfigJson file: " << dataPath + "/controlData.json" << std::endl;
        exit(1);
    }
    
    std::ifstream inJSONFile(dataPath + "/controlData.json", std::fstream::in);

    if (inJSONFile.good())
    {
        inJSONFile >> readInJSON;
        inJSONFile.close();
        DATA_TYPE = readInJSON.at("DATA_TYPE");
        if (readInJSON.contains("USE_ANI"))
        {
            USE_ANI = readInJSON.at("USE_ANI");
        }
        if (readInJSON.contains("CALC_P_NORMAL"))
        {
            CALC_P_NORMAL = readInJSON.at("CALC_P_NORMAL");
        }
        if (readInJSON.contains("GEN_SPLASH"))
        {
            GEN_SPLASH = readInJSON.at("GEN_SPLASH");
        }
        if (readInJSON.contains("NEED_RECORD"))
        {
            NEED_RECORD = readInJSON.at("NEED_RECORD");
        }
        if (readInJSON.contains("OUTPUT_TYPE"))
        {
            OUTPUT_TYPE = readInJSON.at("OUTPUT_TYPE");
        }
        if (DATA_TYPE == 1) {
            if (readInJSON.contains("TARGET_FRAME"))
            {
                TARGET_FRAME = readInJSON.at("TARGET_FRAME");
            }
            const auto filePath = readInJSON.at("DATA_FILE");
            for (auto it = filePath.begin(); it != filePath.end(); it++)
            {
                if (TARGET_FRAME != 0) {
                    if (TARGET_FRAME != int(std::distance(filePath.begin(), it))+1)
                    {
                        continue;
                    }
                }
                DATA_PATHES.push_back(*it);
            }
        } else {
            if (readInJSON.contains("PREFIX"))
            {
                PREFIX = readInJSON.at("PREFIX");
            }
            if (PREFIX.empty() || PREFIX == "")
            {
                const std::string filePath = readInJSON.at("DATA_FILE");
                std::string data_pathes = filePath;
                parseString(&DATA_PATHES, data_pathes, ",");
            } else {
#ifdef _WIN32
                intptr_t hFile;
                struct _finddata_t fileInfo;
                std::string p;
                if ((hFile = _findfirst(p.assign(dataPath).append("/").append(PREFIX).append("*.csv").c_str(), &fileInfo)) != -1)
                {
                    do
                    {
                        if (!(fileInfo.attrib & _A_SUBDIR))
                        {
                            DATA_PATHES.push_back(std::string(fileInfo.name));
                        }
                    } while (_findnext(hFile, &fileInfo) == 0);
                }   
#else
                // TODO: In Linux
#endif
            }
            if (readInJSON.contains("RADIUS"))
            {
                RADIUS = readInJSON.at("RADIUS");
                IS_CONST_RADIUS = true;
            } else {
                IS_CONST_RADIUS = false;
            }
            if (readInJSON.contains("SMOOTH_FACTOR"))
            {
                SMOOTH_FACTOR = readInJSON.at("SMOOTH_FACTOR");
            }
            if (readInJSON.contains("ISO_FACTOR"))
            {
                ISO_FACTOR = readInJSON.at("ISO_FACTOR");
            }
        }
    }
    else
    {
        std::cout << "Cannot open case!" << std::endl;
    }
}

void loadParticlesFromCSV(std::string &csvPath,
                          std::vector<Eigen::Vector3d> &particles,
                          std::vector<double>* radiuses)
{
    std::ifstream ifn;
    ifn.open(csvPath.c_str());

    int radiusIdx = -1, xIdx = -1, yIdx = -1, zIdx = -1;

    particles.clear();

    std::string line;
    std::vector<std::string> titles;
    std::vector<double> elements;
    std::getline(ifn, line);
    replaceAll(line, "\"", "");
    parseString(&titles, line, ",");
    if (!IS_CONST_RADIUS)
    {
        radiuses->clear();
        radiusIdx = std::distance(titles.begin(),
        std::find_if(titles.begin(), titles.end(), 
        [&](const std::string &title) {
            std::regex reg("^((.*)radius|r)$", std::regex::icase);
            return std::regex_match(title, reg);
        }));
    }
    xIdx = std::distance(titles.begin(), 
    std::find_if(titles.begin(), titles.end(), [&](const std::string &title) {
        std::regex reg("^(.*(:|_|-)( *)(x|0)(\"*)|x)$", std::regex::icase);
        return std::regex_match(title, reg);
    }));
    yIdx = std::distance(titles.begin(), 
    std::find_if(titles.begin(), titles.end(), [&](const std::string &title) {
        std::regex reg("^(.*(:|_|-)( *)(y|1)(\"*)|y)$", std::regex::icase);
        return std::regex_match(title, reg);
    }));
    zIdx = std::distance(titles.begin(), 
    std::find_if(titles.begin(), titles.end(), [&](const std::string &title) {
        std::regex reg("^(.*(:|_|-)( *)(z|2)(\"*)|z)$", std::regex_constants::ECMAScript | std::regex::icase);
        return std::regex_match(title, reg);
    }));

    printf("%d %d %d %d \n", xIdx, yIdx, zIdx, radiusIdx);

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
        particles.push_back(Eigen::Vector3d(elements[xIdx], elements[yIdx], elements[zIdx]));
        if (!IS_CONST_RADIUS)
        {
            radiuses->push_back(elements[radiusIdx]);
        }
        getline(ifn, line);
    }
}

void run(std::string dataDirPath, std::string outPath)
{
    loadConfigJson(dataDirPath);
    double frameStart = 0;
    int index = 1;
    std::vector<Eigen::Vector3d> particles;
    std::vector<double>* radiuses = (IS_CONST_RADIUS ? nullptr : new std::vector<double>());
    for (const std::string frame : DATA_PATHES)
    {
        Mesh* mesh = new Mesh(int(pow(10, 4)));
        std::cout << "-=   Frame " << (TARGET_FRAME == 0 ? index : TARGET_FRAME) << " " << frame << "   =-"
                  << std::endl;
        std::string dataPath = dataDirPath + "/" + frame;
        frameStart = get_time();

        switch (DATA_TYPE)
        {
        case 0:
            loadParticlesFromCSV(dataPath, particles, radiuses);
            break;
        case 1:
            // readShonDyParticleData(dataPath, particles, radiuses);

            break;
        default:
            printf("ERROR: Unknown DATA TYPE;");
            exit(1);
        }
        if (!IS_CONST_RADIUS)
        {
            if (abs(*std::max_element(radiuses->begin(), radiuses->end()) - *std::min_element(radiuses->begin(), radiuses->end())) < 1e-7)
            {
                    IS_CONST_RADIUS = true;
                    RADIUS = radiuses->at(0);
            }            
        }
        printf("Particles Number = %zd\n", particles.size());
        SurfReconstructor* constructor = new SurfReconstructor(particles, radiuses, mesh, RADIUS, ISO_FACTOR, SMOOTH_FACTOR);
        Recorder* recorder = new Recorder(dataDirPath, frame.substr(0, frame.size() - 4), constructor);
        constructor->Run();
        if (NEED_RECORD)
        {
            // recorder.RecordProgress();
            recorder->RecordParticles();
            // recorder->RecordFeatures();
        }
        std::string output_name = frame.substr(0, frame.find_last_of('.'));
        if (! std::filesystem::exists(outPath))
        {
            std::filesystem::create_directories(outPath);
        }
        std::cout << "Output path: " << outPath + "/" + output_name + "." + OUTPUT_TYPE<< std::endl;
        try
        {
            if ("ply" == OUTPUT_TYPE || "PLY" == OUTPUT_TYPE)
            {
                writePlyFile(*mesh,
                    outPath + "/" + output_name + ".ply");
            } else if ("obj" == OUTPUT_TYPE || "OBJ" == OUTPUT_TYPE) 
            {
                writeObjFile(*mesh,
                    outPath + "/" + output_name + ".obj");    
            } else {
                writePlyFile(*mesh,
                    outPath + "/" + output_name + ".ply");
            }  
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            std::cout << "Error happened during writing result." << std::endl;
            std::cout << "Result output path: " << outPath + "/" + output_name + "." + OUTPUT_TYPE + ";" << std::endl;
            std::cout << "In Memory Mesh : Vertices=" << mesh->verticesNum << ", Cells=" << mesh->trianglesNum << ";" << std::endl;
            exit(1);
        }
        index++;
        delete(mesh);
        delete(constructor);
        delete(recorder);
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

    std::string dataDirPath;
    std::string outPath;

    switch (argc)
    {
    case 3:
        dataDirPath = std::string(argv[1]);
        outPath = std::string(argv[2]);
        std::cout << "Data dir: " << dataDirPath << std::endl;
        std::cout << "Out dir: " << outPath << std::endl;
        run(dataDirPath, outPath);
        break;
    case 2:
        dataDirPath = std::string(argv[1]);
        outPath = dataDirPath + "/out";
        std::cout << "Data dir: " << dataDirPath << std::endl;
        run(dataDirPath, outPath);
        break;
    case 1:
    default:
        dataDirPath =
        // "E:/data/multiR/mr_csv";
        // "E:/data/vtk/csv";
        // "E:/data/vtk_11/vtk/oil/h5";
        "C:/Users/11379/Desktop/liu";
        outPath = 
        // "E:/data/multiR/mr_csv/out";
        // "E:/data/vtk/csv/out";
        // "E:/data/vtk_11/vtk/oil/out";
        "C:/Users/11379/Desktop/liu/out";
        run(dataDirPath, outPath);
        break;
    }

    return 0;
}
