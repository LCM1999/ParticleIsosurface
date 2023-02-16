#include "kdtree_neighborhood_searcher.h"
#include <vector>
#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <queue>
#include <algorithm>

/**
 * @brief 构造函数
 * @param _particles 粒子坐标数组
 * @param _radius 粒子半径数组
 * @param _split 分组方法，若为正整数则按半径每个分组，最小阈值为该值；若为负数则平均分组
 * @param _index 粒子编号数组，若默认 0 开始则为 nullptr
*/
KDTreeNeighborhoodSearcher::KDTreeNeighborhoodSearcher(std::vector<Eigen::Vector3f>* _particles, std::vector<double>* _radius, int _split, std::vector<int>* _index)
{
    split_method = _split;
    index_map.clear();
    if (_index == nullptr)
    {
        for (unsigned int i = 0; i < _particles->size(); ++i)
        {
            index_map.push_back(int(i));
        }
    }
    else
    {
        if (_index->size() != _particles->size())
        {
            printf("Error: Index given, but size of it doesn't equal to particles'\n错误: 创建邻域搜索时，给出了编号数组，但与粒子数组大小不一致");
            for (unsigned int i = 0; i < _particles->size(); ++i)
            {
                index_map.push_back(int(i));
            }
        }
        else
        {
            for (unsigned int i = 0; i < _particles->size(); ++i)
            {
                index_map.push_back(_index->at(i));
            }
        }
    }
    for (unsigned int i = 0; i < _particles->size(); ++i)
    {
        particles.push_back(ParticleInfo(_particles->at(i), int(i), _radius->at(i)));
    }
    InitializeGroups();
}
/**
 * @brief 粒子分组
*/
void KDTreeNeighborhoodSearcher::InitializeGroups()
{
    sort(particles.begin(), particles.end(), [](ParticleInfo a, ParticleInfo b) { return a.radius < b.radius; });
    std::vector<Eigen::Vector3f> group_particles;
    std::vector<int> group_ids;
    double group_max_radius;
    int group_size;

    if (split_method <= 0)
    {
        group_size = int(sqrt(particles.size())); // 这里设定分组大小为 sqrt(n)，可按需更改
        if (!group_size) group_size = 1;
        for (unsigned int i = 0; i < particles.size() / group_size + 1; ++i)
        {
            unsigned int
                l_range = i * group_size,
                r_range = (i + 1) * group_size - 1;
            if (r_range > particles.size() - 1) r_range = particles.size() - 1;
            group_particles.clear();
            group_ids.clear();
            group_max_radius = particles[r_range].radius;
            for (unsigned int j = l_range; j <= r_range; ++j)
            {
                group_particles.push_back(particles[j].coordinate);
                group_ids.push_back(particles[j].id);
            }
            KDTree* tree = new KDTree(&group_particles, &group_ids);
            groups.push_back(RadiusTreePair(group_max_radius, tree));
            printf("Group size: %d\n", r_range - l_range + 1);
        }
    }
    else if (split_method > 0)
    {
        group_max_radius = particles[0].radius;
        group_particles.push_back(particles[0].coordinate);
        group_ids.push_back(particles[0].id);
        for (int i = 1; i < particles.size(); ++i)
        {
            if (particles[i].radius != group_max_radius && group_particles.size() > split_method)
            {
                KDTree* tree = new KDTree(&group_particles, &group_ids);
                groups.push_back(RadiusTreePair(particles[i].radius, tree));
                printf("Group size:%d\n", (group_particles.size()));
                group_particles.clear();
                group_ids.clear();
                group_max_radius = particles[i].radius;
            }
            group_particles.push_back(particles[i].coordinate);
            group_ids.push_back(particles[i].id);
        }
        if (group_particles.size() != 0)
        {
            KDTree* tree = new KDTree(&group_particles, &group_ids);
            groups.push_back(RadiusTreePair(particles[particles.size() - 1].radius, tree));
            printf("Group size:%d\n", (group_particles.size()));
        }
    }
    // 排序完后，变为原数组，方便后续处理
    sort(particles.begin(), particles.end(), [](ParticleInfo a, ParticleInfo b) { return a.id < b.id; });
}
/**
 * @brief 析构函数
*/
KDTreeNeighborhoodSearcher::~KDTreeNeighborhoodSearcher()
{
    for (int i = 0; i < groups.size(); ++i)
    {
        delete(groups[i].tree);
    }
}
/**
 * @brief 获取与指定球有交集的粒子（邻居）
 * @param target 目标球心
 * @param radius 目标半径
 * @param neighborhood_ids 返回：邻居编号
 * @param neighborhood_coordinates 返回：邻居坐标
*/
void KDTreeNeighborhoodSearcher::GetNeighborhood(Eigen::Vector3f target, double radius, std::vector<int>* neighborhood_ids, std::vector<Eigen::Vector3f>* neighborhood_coordinates)
{
    if (neighborhood_ids != nullptr)
        neighborhood_ids->clear();
    if (neighborhood_coordinates != nullptr)
        neighborhood_coordinates->clear();

    // 第一次筛选，这一步会导致有不是邻居的粒子加入
    std::vector<int> current_group_ids, all_group_ids; // 编号数组，注意这里的编号是搜索器内部编号，不是给出的
    std::vector<Eigen::Vector3f> current_group_coordinates, all_group_coordinates; // 坐标数组

    int i;
    #pragma omp parallel for num_threads(2) private(current_group_ids, current_group_coordinates)
    for (i = 0; i < groups.size(); ++i)
    {
        current_group_ids.clear();
        current_group_coordinates.clear();
        groups[i].tree->GetPointWithinRadius(target, 2 * (radius + groups[i].max_radius) + 1e-6, &current_group_ids, &current_group_coordinates);
        #pragma omp critical (neighborhood_pushback) 
        {
            for (int j = 0; j < current_group_ids.size(); ++j)
            {
                all_group_ids.push_back(current_group_ids[j]);
            }
            for (int j = 0; j < current_group_coordinates.size(); ++j)
            {
                all_group_coordinates.push_back(current_group_coordinates[j]);
            }
        }
    }

    // 第二次筛选，对第一步筛出来的粒子逐个进行检查
    //int error_count = 0;
    for (int i = 0; i < all_group_ids.size(); ++i)
    {
        int cur_id = all_group_ids[i];
        if ((particles[cur_id].coordinate - target).norm() <= 2 * (particles[cur_id].radius + radius))
        {
            if (neighborhood_ids != nullptr)
                neighborhood_ids->push_back(index_map[cur_id]); // 这里需要映射编号
            if (neighborhood_coordinates != nullptr)
                neighborhood_coordinates->push_back(particles[cur_id].coordinate);
        }
        //else
        //{
        //    ++error_count;
        //}
    }
    //if (all_group_ids.size() && error_count)
    //    printf("Pass rate is: %f\n", 1.0 - 1.0 * error_count / int(all_group_ids.size()));
}