#include <iostream>
#include <fstream>
#include <omp.h>
#include <set>
#include "iso_common.h"
#include "iso_method_ours.h"
#include "surface_reconstructor.h"
#include "global.h"
#include "recorder.h"
#include "json.hpp"

#include "kdtree_neighborhood_searcher.h"

#ifdef _WIN32
#include <windows.h>
#else
#include "unistd.h"
#endif

int OMP_USE_DYNAMIC_THREADS = 0;
int OMP_THREADS_NUM = 16;

// variants for test
bool NEED_RECORD;
std::vector<std::string> CSV_PATHES;
int CSV_TYPE;
float P_RADIUS;

void writeFile(Mesh& m, std::string fn)
{
    FILE* f = fopen(fn.c_str(), "w");
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

/**
 * @brief 导入 json 文件
 * @param {string} controlJsonPath - 文件路径
 * @return {*}
 */
void loadConfigJson(const std::string controlJsonPath)
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
        NEED_RECORD = readInJSON.at("NEED_RECORD");
    }
    else
    {
        std::cout << "Cannot open case!" << std::endl;
    }
}



/**
 * @brief 从文件中导入粒子数据
 * @param {string&} csvPath csv文件路径
 * @param {std::vector<Eigen::Vector3f>&} particles 粒子坐标数组
 * @param {std::vector<float>&} density 粒子密度数组
 * @param {std::vector<float>&} mass 粒子质量数组
 * @return {*}
 */
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


/**
 * @brief 运行重建表面程序
 * @param {string&} csvDirPath csv文件及controlData所在文件夹
 * @return {*}
 */
void testWithCSV(std::string& csvDirPath)
{
    loadConfigJson(csvDirPath + "/controlData.json");
    double frameStart = 0;
    int index = 0;
    std::vector<Eigen::Vector3f> particles;
    std::vector<float> density;
    std::vector<float> mass;
    for (const std::string frame : CSV_PATHES)
    {
        Mesh mesh(P_RADIUS);
        std::cout << "-=   Frame " << index << " " << frame << "   =-" << std::endl;
        std::string csvPath = csvDirPath + "/" + frame;
        frameStart = get_time();
        // 导入数据
        loadParticlesFromCSV(csvPath, particles, density, mass);
        printf("particle size: %zd\n", particles.size());

        SurfReconstructor constructor(particles, density, mass, mesh, P_RADIUS);
        Recorder recorder(csvDirPath, frame.substr(0, frame.size() - 4), &constructor);
        constructor.Run();

        printf("Run finished\n");

        if (NEED_RECORD)
        {
            recorder.RecordProgress();
            recorder.RecordParticles();
        }

        writeFile(mesh, csvDirPath + "/" + frame.substr(0, frame.size() - 4) + ".obj");
        index++;
    }
}

double GenRandomDouble()
{
    return 1.0 / (rand() % 100 + 1) * 100000;
}

void KDTreeTest()
{
    printf("-= Test for KD-Tree =-\n");
    srand(time(0));
    // 创建粒子数组
    std::vector<Eigen::Vector3f> _GlobalParticles;
    std::vector<double> _GlobalRadius;
    for (int i = 0; i < 1000000; ++i)
    {
        _GlobalParticles.push_back({ GenRandomDouble(), GenRandomDouble(), GenRandomDouble() });
    }
    for (unsigned int i = 0; i < _GlobalParticles.size(); ++i)
    {
        _GlobalRadius.push_back(1.0 / (rand() % 100 + 1));
    }
    std::vector<int> p_idx;
    std::vector<double> dis;
    std::vector<Eigen::Vector3f> par;
    Eigen::Vector3f testvec;
    testvec = { GenRandomDouble(), GenRandomDouble(), GenRandomDouble() };
    double test_radius = GenRandomDouble() / 10;
    printf("Target: (%f,%f,%f) r=%f\n", testvec[0], testvec[1], testvec[2], test_radius);

    // 时间测试
    double start_time = get_time(), out_time, brute_time;

    // 基础生成
    KDTreeNeighborhoodSearcher* kd_searcher = new KDTreeNeighborhoodSearcher(&_GlobalParticles, &_GlobalRadius);
    printf("Time generating tree = %f\n", get_time() - start_time);
    //for (int testcase = 0; testcase < 10000; ++testcase)
    //{
    //    testvec = { GenRandomDouble(), GenRandomDouble(), GenRandomDouble() };
    //    double test_radius = GenRandomDouble() * 10;
    //}
    start_time = get_time();
    kd_searcher->GetNeighborhood(testvec, test_radius, &p_idx, &par);
    out_time = get_time() - start_time;

    // 暴力生成
    start_time = get_time();
    std::vector<int> vec_correct_ids;
    for (int i = 0; i < _GlobalParticles.size(); ++i)
    {
        if ((_GlobalParticles[i] - testvec).norm() <= _GlobalRadius[i] + test_radius)
        {
            vec_correct_ids.push_back(i);
        }
    }
    brute_time = get_time() - start_time;

    // 测试结果验证
    std::set<int> correct_ids;
    for (int i = 0; i < _GlobalParticles.size(); ++i)
    {
        if ((_GlobalParticles[i] - testvec).norm() <= _GlobalRadius[i] + test_radius)
        {
            correct_ids.insert(i);
        }
    }
    printf("Neighborhood size: %d\n", int(correct_ids.size()));
    if (correct_ids.size() != p_idx.size())
    {
        printf("Neighborhood Error: size not equal. Expect %d, got %d instead", int(correct_ids.size()), int(p_idx.size()));
        system("pause");
    }
    for (int i = 0; i < p_idx.size(); ++i)
    {
        if (!correct_ids.count(p_idx[i]))
        {
            printf("Neighborhood Error: %d should not in list", p_idx[i]);
            system("pause");
        }
    }
    for (int i = 0; i < p_idx.size(); ++i)
    {
        if (_GlobalParticles[p_idx[i]] != par[i])
        {
            printf("Neighborhood Error: id coordinate not match");
            system("pause");
        }
    }

    printf("Time cost: our = %f  brute force = %f\n", out_time, brute_time);

    printf("KDTree Test Pass\n");
}

int main(int argc, char** argv)
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

    std::cout << std::to_string(argc) << std::endl;

    while (true)
        KDTreeTest();

    if (argc == 2)
    {
        std::string csvDirPath = std::string(argv[1]);
        std::cout << csvDirPath << std::endl;
        testWithCSV(csvDirPath);
    }
    else
    {
        //std::cout << "arguments format error" << "\n";
        std::string csvDirPath = "C:\\csv";
        std::cout << csvDirPath << std::endl;
        testWithCSV(csvDirPath);
    }

    return 0;
}
