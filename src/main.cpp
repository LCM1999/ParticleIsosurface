#include <fstream>
#include <iostream>
#include <filesystem>
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

#include "marching.h"

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
bool NEED_RECORD = false;
int TARGET_FRAME = 0;
// std::string PREFIX = "";
std::string SUFFIX = "";    // CSV, H5
bool WITH_NORMAL;
std::vector<std::string> DATA_PATHES;
std::string OUTPUT_TYPE = "ply";
float RADIUS = 0;
float SMOOTH_FACTOR = 2.0;
float ISO_FACTOR = 1.9;
float ISO_VALUE = 0.0f;
// bool USE_CUDA = false;
bool CALC_P_NORMAL = true;
bool GEN_SPLASH = true;
bool SINGLE_LAYER = false;
bool USE_OURS = true;
bool USE_POLY6 = 0;

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
    float version;
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
        if (readInJSON.contains("USE_ANI"))
        {
            USE_ANI = readInJSON.at("USE_ANI");
        }
        if (readInJSON.contains("USE_OURS"))
        {
            USE_OURS = readInJSON.at("USE_OURS");
        }
        if (readInJSON.contains("USE_POLY6"))
        {
            USE_POLY6 = readInJSON.at("USE_POLY6");
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
        if (readInJSON.contains("SUFFIX"))
        {
            SUFFIX = readInJSON.at("SUFFIX");
        }
        if (SUFFIX.empty() || SUFFIX == "")
        {
            const std::string filePath = readInJSON.at("DATA_FILE");
            std::string data_pathes = filePath;
            parseString(&DATA_PATHES, data_pathes, ",");
        } else {
#ifdef _WIN32
            intptr_t hFile;
            struct _finddata_t fileInfo;
            std::string p;
            if ((hFile = _findfirst(p.assign(dataPath).append("/").append("*").append(SUFFIX).c_str(), &fileInfo)) != -1)
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
        if (readInJSON.contains("ISO_VALUE"))
        {
            ISO_VALUE = readInJSON.at("ISO_VALUE");
        }
        
    }
    else
    {
        std::cout << "Cannot open case!" << std::endl;
    }
}

void loadParticlesFromCSV(std::string &csvPath,
                          std::vector<Eigen::Vector3f> &particles,
                          std::vector<float> &radiuses)
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
        radiuses.clear();
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
        particles.push_back(Eigen::Vector3f(elements[xIdx], elements[yIdx], elements[zIdx]));
        if (!IS_CONST_RADIUS)
        {
            radiuses.push_back(elements[radiusIdx]);
        }
        getline(ifn, line);
    }
}

bool readShonDyParticleData(const std::string &fileName,
                            std::vector<Eigen::Vector3f> &positions,
                            std::vector<float>& radiuses)
{
    if (!std::filesystem::exists(fileName))
    {
        std::cout << "Error: cannot find hdf5 file: " << fileName << std::endl;
        return false;
    }

    // open file
    auto hdf5FileID = H5Fopen(fileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (hdf5FileID < 0)
    {
        std::cout << "Error: cannot read hdf5 file: " << fileName << std::endl;
        return false;
    }

    // read particle positions
    auto nodeSize = size(hdf5FileID, "position");
    const int numberOfNodes = nodeSize / 3;
    std::vector<double> nodes(nodeSize);
    std::vector<double> radiusesdouble(numberOfNodes);
    readDouble(hdf5FileID, "position", nodes);
    positions.resize(numberOfNodes);

    // read particle densities
    if (!IS_CONST_RADIUS)
    {
        readDouble(hdf5FileID, "particleRadius", radiusesdouble);
        radiuses.resize(numberOfNodes);
    }
    
    // convert float data to float
    for (int i = 0; i < numberOfNodes; i++)
    {
        positions[i] =
            Eigen::Vector3f(nodes[3 * i], nodes[3 * i + 1], nodes[3 * i + 2]);
        if (!IS_CONST_RADIUS)
        {
            radiuses[i] = radiusesdouble[i];
        }
    }

    // close file
    H5Fclose(hdf5FileID);
    return true;
}

void runOurs(std::string dataDirPath, std::string outPath)
{
    float frameStart = 0;
    int index = 1;
    std::vector<Eigen::Vector3f> particles;
    std::vector<float> radiuses;
    for (const std::string frame : DATA_PATHES)
    {
        Mesh mesh(int(pow(10, 4)));
        std::cout << "-=   Frame " << (TARGET_FRAME == 0 ? index : TARGET_FRAME) << " " << frame << "   =-"
                  << std::endl;
        std::string dataPath = dataDirPath + "/" + frame;
        frameStart = get_time();

        if (SUFFIX == "")
        {
            SUFFIX = std::filesystem::path(dataPath).filename().extension().string();
        }
        if (".csv" == SUFFIX) {
            loadParticlesFromCSV(dataPath, particles, radiuses);
        } else if (".h5" == SUFFIX) {
            readShonDyParticleData(dataPath, particles, radiuses);
        }

        if (!IS_CONST_RADIUS)
        {
            if (abs(*std::max_element(radiuses.begin(), radiuses.end()) - *std::min_element(radiuses.begin(), radiuses.end())) < 1e-7)
            {
                    IS_CONST_RADIUS = true;
                    RADIUS = radiuses[0];
            }
        }
        printf("Particles Number = %zd\n", particles.size());
        SurfReconstructor* constructor = new SurfReconstructor(particles, radiuses, &mesh, RADIUS);
        Recorder recorder(dataDirPath, frame.substr(0, frame.size() - 4), constructor);
        constructor->Run(ISO_FACTOR, SMOOTH_FACTOR);
        if (NEED_RECORD)
        {
            // recorder.RecordProgress();
            recorder.RecordParticles();
            // recorder.RecordFeatures();
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
                writePlyFile(mesh,
                    outPath + "/" + output_name + ".ply");
            } else if ("obj" == OUTPUT_TYPE || "OBJ" == OUTPUT_TYPE) 
            {
                writeObjFile(mesh,
                    outPath + "/" + output_name + ".obj");    
            } else {
                writePlyFile(mesh,
                    outPath + "/" + output_name + ".ply");
            }  
            std::cout << "Output Done" << std::endl;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            std::cout << "Error happened during writing result." << std::endl;
            std::cout << "Result output path: " << outPath + "/" + output_name + "." + OUTPUT_TYPE + ";" << std::endl;
            std::cout << "In Memory Mesh : Vertices=" << mesh.verticesNum << ", Cells=" << mesh.trianglesNum << ";" << std::endl;
            exit(1);
        }
        index++;
        delete constructor;
    }
}

void runUniform(std::string dataDirPath, std::string outPath)
{
    // float frameStart = 0;
    int index = 1;
    std::vector<Eigen::Vector3f> particles;
    std::vector<float> radiuses;
    for (const std::string frame : DATA_PATHES)
    {
        Mesh mesh(int(pow(10, 4)));
        std::cout << "-=   Frame " << (TARGET_FRAME == 0 ? index : TARGET_FRAME) << " " << frame << "   =-"
                  << std::endl;
        std::string dataPath = dataDirPath + "/" + frame;
        // frameStart = get_time();

        if (SUFFIX == "")
        {
            SUFFIX = std::filesystem::path(dataPath).filename().extension().string();
        }
        if (".csv" == SUFFIX) {
            loadParticlesFromCSV(dataPath, particles, radiuses);
        } else if (".h5" == SUFFIX) {
            readShonDyParticleData(dataPath, particles, radiuses);
        }

        if (!IS_CONST_RADIUS)
        {
            if (abs(*std::max_element(radiuses.begin(), radiuses.end()) - *std::min_element(radiuses.begin(), radiuses.end())) < 1e-7)
            {
                    IS_CONST_RADIUS = true;
                    RADIUS = radiuses[0];
            }
        }
        printf("Particles Number = %zd\n", particles.size());
        std::string output_name = frame.substr(0, frame.find_last_of('.'));
        UniformGrid* uniformGrid = IS_CONST_RADIUS ? new UniformGrid(particles, RADIUS) : new UniformGrid(particles, radiuses);
        uniformGrid->Run(ISO_VALUE, output_name, outPath);
        std::cout << "Output path: " << outPath + "/" + output_name + ".vti"<< std::endl;
        std::cout << "Output Done" << std::endl;
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
        runOurs(dataDirPath, outPath);
        break;
    case 2:
        dataDirPath = std::string(argv[1]);
        outPath = dataDirPath + "/out";
        std::cout << "Data dir: " << dataDirPath << std::endl;
        runOurs(dataDirPath, outPath);
        break;
    case 1:
    default:
        dataDirPath =
        "E:/data/multiR/mr_csv";
        // "E:/data/geo";
        // "E:/data/3s/20231222-water";
        // "E:/data/damBreak3D-27steps";
        // "E:/BaiduNetdiskDownload/MultiResolutionResults/damBreak3D";
        // "E:/data/ring/csv";
        // "E:/data/car_render_test_data_2/Fluid";
        // "E:/data/oil_csv";
        // "E:/data/test";
        // "C:/Users/11379/Desktop/protein";
        outPath = 
        "E:/data/multiR/mr_csv/out";
        // "E:/data/geo/out";
        // "E:/data/3s/20231222-water/out";
        // "E:/data/damBreak3D-27steps/out";
        // "E:/BaiduNetdiskDownload/MultiResolutionResults/damBreak3D/out";
        // "E:/data/ring/csv/out";
        // "E:/data/car_render_test_data_2/Fluid/out";
        // "E:/data/oil_csv/out";
        // "E:/data/test/out";
        // "C:/Users/11379/Desktop/protein/out";
        loadConfigJson(dataDirPath);
        if (USE_OURS)
        {
            runOurs(dataDirPath, outPath);
        } else {
            runUniform(dataDirPath, outPath);
        }
        
        break;
    }

    return 0;
}
