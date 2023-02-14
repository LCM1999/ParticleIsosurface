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

void LoadParticlesInfo(std::string& csvPath, std::vector<Eigen::Vector3f>& particles, std::vector<double>& radius)
{
    std::ifstream ifn;
    ifn.open(csvPath.c_str());

    particles.clear();
    radius.clear();

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
        particles.push_back(Eigen::Vector3f(elements[2], elements[3], elements[4]));
        radius.push_back(elements[0]);
        //switch (CSV_TYPE)
        //{
        //    case 0:
        //        particles.push_back(Eigen::Vector3f(elements[0], elements[1], elements[2]));
        //        mass.push_back(elements[6]);
        //        density.push_back(elements[7] + 1000.0f);
        //        break;
        //    case 1:
        //        particles.push_back(Eigen::Vector3f(elements[1], elements[2], elements[3]));
        //        mass.push_back(1.0f);
        //        density.push_back(elements[0] + 1000.0f);
        //        break;
        //    case 2:
        //        particles.push_back(Eigen::Vector3f(elements[0], elements[1], elements[2]));
        //        mass.push_back(1.0f);
        //        density.push_back(elements[3] + 1000.0f);
        //        break;
        //    default:
        //        printf("Unknown type of csv format.");
        //        exit(1);
        //        break;
        //}

        getline(ifn, line);
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
    bool random_mode = false; // 随机还是使用数据集
    if (random_mode)
    {
        for (int i = 0; i < 100000; ++i)
        {
            _GlobalParticles.push_back({ GenRandomDouble(), GenRandomDouble(), GenRandomDouble() });
        }
        for (unsigned int i = 0; i < _GlobalParticles.size(); ++i)
        {
            _GlobalRadius.push_back(1.0 / (rand() % 100 + 1));
        }
    }
    else
    {
        std::string filepath = "C:\\csv with radius\\mr_10.csv";
        LoadParticlesInfo(filepath, _GlobalParticles, _GlobalRadius);
    }

    std::vector<int> result_id, verify_id;
    std::vector<Eigen::Vector3f> result_coordinate, verify_coordinate;
    Eigen::Vector3f test_coordinate;
    double test_radius;

    // 时间测试
    double start_time = get_time(), our_time, brute_time, our_time_all = 0.0, brute_time_all = 0.0;

    KDTreeNeighborhoodSearcher* kd_searcher = new KDTreeNeighborhoodSearcher(&_GlobalParticles, &_GlobalRadius, 0);
    double time_gen_tree = get_time() - start_time;
    printf("Time generating tree = %f\n", time_gen_tree);
    
    std::map<int, int> neighborhood_count;
    std::map<int, double> neighborhood_count_time;
    bool need_brute = false; // 是否需要暴力生成验证
    for (int testcase = 0; testcase < _GlobalParticles.size(); ++testcase)
    {
        verify_id.clear();
        verify_coordinate.clear();
        if (testcase % 1000 == 0)
            std::cout << "Test Case #" << testcase << "\n";
        test_coordinate = _GlobalParticles[testcase];
        test_radius = _GlobalRadius[testcase];

        if (need_brute)
        {
            start_time = get_time();
            for (int i = 0; i < _GlobalParticles.size(); ++i)
            {
                if ((_GlobalParticles[i] - test_coordinate).norm() <= 2 * (_GlobalRadius[i] + test_radius))
                {
                    verify_id.push_back(i);
                }
            }
            brute_time = get_time() - start_time;
            brute_time_all += brute_time;
        }

        start_time = get_time();
        kd_searcher->GetNeighborhood(test_coordinate, test_radius, &result_id, &result_coordinate);
        our_time = get_time() - start_time;
        our_time_all += our_time;

        if (neighborhood_count.count(int(result_id.size())))
        {
            neighborhood_count[int(result_id.size())] = neighborhood_count[int(result_id.size())] + 1;
            neighborhood_count_time[int(result_id.size())] = our_time;
        }
        else
        {
            neighborhood_count[int(result_id.size())] = 1;
            neighborhood_count_time[int(result_id.size())] = neighborhood_count_time[int(result_id.size())] + our_time;
        }

        //printf("Single time cost: our = %f  brute force = %f\n", our_time, brute_time);

        if (need_brute)
        {
            // 测试结果验证
            for (int i = 0; i < result_id.size(); ++i)
            {
                if (_GlobalParticles[result_id[i]] != result_coordinate[i])
                {
                    printf("Neighborhood Error: id coordinate not match\n");
                    printf("Expect %d-(%f,%f,%f), got (%f,%f,%f) instead\n",
                        result_id[i],
                        _GlobalParticles[result_id[i]].x(),
                        _GlobalParticles[result_id[i]].y(),
                        _GlobalParticles[result_id[i]].z(),
                        result_coordinate[i].x(),
                        result_coordinate[i].y(),
                        result_coordinate[i].z());

                    system("pause");
                }
            }
            bool neighborhood_error = false;
            sort(result_id.begin(), result_id.end());
            sort(verify_id.begin(), verify_id.end());
            if (verify_id.size() != result_id.size())
            {
                printf("Neighborhood Error: size not equal. Expect %d, got %d instead\n", int(verify_id.size()), int(result_id.size()));
                neighborhood_error = true;
            }
            if (!neighborhood_error)
                for (int i = 0; i < result_id.size(); ++i)
                {
                    if (result_id[i] != verify_id[i])
                    {
                        printf("Neighborhood Error: not match\n");
                        neighborhood_error = true;
                    }
                }
            if (neighborhood_error)
            {
                printf("=======\nid = %d, coor = (%f,%f,%f), radius = %f\n", testcase,
                    _GlobalParticles[testcase].x(),
                    _GlobalParticles[testcase].y(),
                    _GlobalParticles[testcase].z(),
                    _GlobalRadius[testcase]);
                printf("Expect neighborhood are:\n");
                for (int i = 0; i < verify_id.size(); ++i)
                {
                    printf("id = %d, coordinate = (%f,%f,%f), radius = %f, distance = %f\n", verify_id[i], _GlobalParticles[verify_id[i]].x(), _GlobalParticles[verify_id[i]].y(), _GlobalParticles[verify_id[i]].z(), _GlobalRadius[verify_id[i]], (_GlobalParticles[verify_id[i]] - test_coordinate).norm());
                }
                printf("Got neighborhood are:\n");
                for (int i = 0; i < result_id.size(); ++i)
                {
                    printf("id = %d\n", result_id[i]);
                }
                system("pause");
            }
        }
    }
    std::map<int, int>::iterator iter;
    for (iter = neighborhood_count.begin(); iter != neighborhood_count.end(); ++iter)
    {
        printf("number of neighborhood = %d, number of testcases = %d, average time cost = %f\n", iter->first, iter->second, neighborhood_count_time[iter->first] / iter->second);
    }
    printf("Total time cost: our = %f  brute force = %f\n", our_time_all, brute_time_all);
    printf("Total time cost(including building tree): our = %f  brute force = %f\n", our_time_all + time_gen_tree, brute_time_all);
    printf("KDTree Test Pass\n");
    system("pause");
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
