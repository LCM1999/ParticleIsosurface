#include "kdtree.h"

KDTree::KDTree()
{
    particles.clear();
    root = nullptr;
    while (!KNearestQueue.empty()) KNearestQueue.pop();
}

KDTree::~KDTree()
{
    DeleteTreeNode(root);
}

/**
 * @brief 递归删除 KD-Tree 结点
 * @param node 目前结点
*/
void KDTree::DeleteTreeNode(KDTreeNode* node)
{
    if (node->son[0] != nullptr) DeleteTreeNode(node->son[0]);
    if (node->son[1] != nullptr) DeleteTreeNode(node->son[1]);
    delete(node);
}

/**
 * @brief 构造函数
 * @param _particles 结点坐标
 * @param _index 结点编号，若提供编号请保证个数一致，且编号不会重复；若不提供则从 0 开始依次编号
*/
KDTree::KDTree(std::vector<Eigen::Vector3f>* _particles, std::vector<int>* _index)
{
    if (_particles == nullptr) return;
    bool use_index = true;
    if (_index == nullptr || _index->size() != _particles->size())
    {
        // 若没提供编号，或编号与结点个数不一致，则不使用
        // 这里没有检验编号唯一等操作
        use_index = false;
    }
    for (unsigned int i = 0; i < _particles->size(); ++i)
    {
        if (use_index)
            particles.push_back((ParticleIDPair){ _particles->at(i), int(_index->at(i)) });
        else
            particles.push_back((ParticleIDPair) { _particles->at(i), int(i) });
    }
    root = BuildKDTree(0, particles.size() - 1);
    while (!KNearestQueue.empty()) KNearestQueue.pop();
}

/**
 * @brief 建立 KD-Tree
 * @param range_left 建立左边界
 * @param range_right 建立右边界
 * @return 根节点
*/
KDTreeNode* KDTree::BuildKDTree(int range_left, int range_right)
{
    if (particles.empty() || range_left > range_right)
    {
        return nullptr;
    }

    SplitAxis split_axis = NOT_SET;
    ParticleIDPair split_point;
    Eigen::Vector3f region_min, region_max;
    GetSplitPoint(range_left, range_right, split_point, split_axis, region_min, region_max);
    int mid = (range_left + range_right) / 2;
    KDTreeNode* split = new KDTreeNode();
    split->particle = split_point;
    split->axis = split_axis;
    split->reigon_min = region_min;
    split->reigon_max = region_max;
    split->id_min = range_left;
    split->id_max = range_right;
    split->son[0] = BuildKDTree(range_left, mid - 1);
    split->son[1] = BuildKDTree(mid + 1, range_right);
    return split;
}

/**
 * @brief 获取分割点
 * @param range_left 左边界
 * @param range_right 右边界
 * @param split_point 分割点（返回）
 * @param split_axis 分割方向（返回）
 * @param region_min 范围最小坐标（返回）
 * @param region_max 范围最大坐标（返回）
*/
void KDTree::GetSplitPoint(int range_left, int range_right, ParticleIDPair& split_point, SplitAxis& split_axis, Eigen::Vector3f& region_min, Eigen::Vector3f& region_max)
{
    // 计算方差
    double ave_x = 0, var_x = 0, ave_y = 0, var_y = 0, ave_z = 0, var_z = 0;
    for (int i = range_left; i <= range_right; ++i)
    {
        ave_x += particles.at(i).coordinate[0];
        ave_y += particles.at(i).coordinate[1];
        ave_z += particles.at(i).coordinate[2];
    }
    ave_x /= range_right - range_left + 1;
    ave_y /= range_right - range_left + 1;
    ave_z /= range_right - range_left + 1;
    for (int i = range_left; i <= range_right; ++i)
    {
        var_x += (particles.at(i).coordinate[0] - ave_x) * (particles.at(i).coordinate[0] - ave_x);
        var_y += (particles.at(i).coordinate[1] - ave_y) * (particles.at(i).coordinate[1] - ave_y);
        var_z += (particles.at(i).coordinate[2] - ave_z) * (particles.at(i).coordinate[2] - ave_z);
    }
    var_x /= range_right - range_left + 1;
    var_y /= range_right - range_left + 1;
    var_z /= range_right - range_left + 1;

    // 选择方差最大的方向进行分割，同时获取分割点
    if (var_x >= var_y && var_x >= var_z)
    {
        split_axis = X;
    }
    else if (var_y >= var_x && var_y >= var_z)
    {
        split_axis = Y;
    }
    else
    {
        split_axis = Z;
    }

    switch (split_axis)
    {
        case X:
            sort(particles.begin() + range_left, particles.begin() + range_right + 1,
                [](ParticleIDPair a, ParticleIDPair b) { return a.coordinate[0] < b.coordinate[0]; });
            break;
        case Y:
            sort(particles.begin() + range_left, particles.begin() + range_right + 1,
                [](ParticleIDPair a, ParticleIDPair b) { return a.coordinate[1] < b.coordinate[1]; });
            break;
        case Z:
            sort(particles.begin() + range_left, particles.begin() + range_right + 1,
                [](ParticleIDPair a, ParticleIDPair b) { return a.coordinate[2] < b.coordinate[2]; });
            break;
        default:
            break;
    }
    split_point = particles.at((range_left + range_right) / 2);
    // 获取区域最大最小坐标
    for (int i = 0; i < 3; ++i)
    {
        region_min[i] = INFINITY;
        region_max[i] = -INFINITY;
    }
    for (int i = range_left; i <= range_right; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            region_min[j] = std::min(region_min[j], particles.at(i).coordinate[j]);
            region_max[j] = std::max(region_max[j], particles.at(i).coordinate[j]);
        }
    }
}

/**
 * @brief 将优先队列的数据传回vector
 * @param nearest_particle_ids 结点编号
 * @param nearest_particles 结点坐标
 * @param distance 结点距离
 * @param k 返回元素个数，可选
*/
void KDTree::SaveQueueData(std::vector<int>* nearest_particle_ids, std::vector<Eigen::Vector3f>* nearest_particles, std::vector<double>* distance, unsigned int k)
{
    if (k > 0 && nearest_particle_ids != nullptr) nearest_particle_ids->reserve(k);
    if (k > 0 && nearest_particles != nullptr) nearest_particles->reserve(k);
    if (k > 0 && distance != nullptr) distance->reserve(k);
    while (!KNearestQueue.empty())
    {
        std::pair<double, ParticleIDPair> node = KNearestQueue.top();
        KNearestQueue.pop();
        if (nearest_particle_ids != nullptr) nearest_particle_ids->push_back(node.second.id);
        if (nearest_particles != nullptr) nearest_particles->push_back(node.second.coordinate);
        if (distance != nullptr) distance->push_back(node.first);
    }
    if (nearest_particle_ids != nullptr) std::reverse(nearest_particle_ids->begin(), nearest_particle_ids->end());
    if (nearest_particles != nullptr) std::reverse(nearest_particles->begin(), nearest_particles->end());
    if (distance != nullptr) std::reverse(distance->begin(), distance->end());
}

/**
 * @brief 获取距离目标点最近的 k 个结点
 * @param target 目标结点
 * @param k 最近结点个数
 * @param nearest_particles 最近结点的编号（返回），不需要则使用 nullptr
 * @param nearest_particles 最近结点的坐标（返回），不需要则使用 nullptr
 * @param distance 最近结点的距离（返回），不需要则使用 nullptr
 * @return 最近结点的个数
*/
int KDTree::GetKNearest(const Eigen::Vector3f& target, unsigned int k, std::vector<int>* nearest_particle_ids, std::vector<Eigen::Vector3f>* nearest_particles, std::vector<double>* distance)
{
    while (KNearestQueue.size()) KNearestQueue.pop();
    if (particles.size() < k) k = particles.size();
    GetKNearestSearch(root, target, k);
    SaveQueueData(nearest_particle_ids, nearest_particles, distance, k);
    return k;
}

/**
 * @brief 递归查找 k 个最近结点
 * @param node 目前结点
 * @param target 目标坐标
 * @param k 最近结点数
*/
void KDTree::GetKNearestSearch(KDTreeNode* node, const Eigen::Vector3f& target, unsigned int k)
{
    if (node == nullptr)
    {
        return;
    }
    // 递归找到最近结点
    if (target[node->axis] < node->particle.coordinate[node->axis])
        GetKNearestSearch(node->son[0], target, k);
    else
        GetKNearestSearch(node->son[1], target, k);
    // 计算是否要加入最近列表内
    double dis = (node->particle.coordinate - target).norm();
    if (KNearestQueue.size() < k)
    {
        KNearestQueue.push(std::make_pair(dis, node->particle));
    }
    else if (KNearestQueue.top().first > dis)
    {
        KNearestQueue.pop();
        KNearestQueue.push(std::make_pair(dis, node->particle));
    }
    // 检查是否需要考虑另一个子树
    double cross_area_dis = abs(target[node->axis] - node->particle.coordinate[node->axis]);
    if (KNearestQueue.top().first > cross_area_dis)
    {
        if (target[node->axis] < node->particle.coordinate[node->axis])
            GetKNearestSearch(node->son[1], target, k);
        else
            GetKNearestSearch(node->son[0], target, k);
    }
}

/**
 * @brief 获取目标点指定半径以内的所有结点
 * @param target 目标结点
 * @param radius 指定半径
 * @param nearest_particles 最近结点的编号（返回），不需要则使用 nullptr
 * @param nearest_particles 最近结点的坐标（返回），不需要则使用 nullptr
 * @param distance 最近结点的距离（返回），不需要则使用 nullptr
 * @return 最近结点的个数
*/
int KDTree::GetPointWithinRadius(const Eigen::Vector3f& target, double radius, std::vector<int>* nearest_particle_ids, std::vector<Eigen::Vector3f>* nearest_particles, std::vector<double>* distance)
{
    while (KNearestQueue.size()) KNearestQueue.pop();
    GetPointWithinRadiusSearch(root, target, radius);
    int queue_size = KNearestQueue.size();
    SaveQueueData(nearest_particle_ids, nearest_particles, distance, queue_size);
    return queue_size;
}

/**
 * @brief 返回 KD-Tree 结点指定长方体区域与球的相交情况
 * @param node KD-Tree 结点
 * @param o 球心坐标
 * @param radius 指定半径
 * @return -1 为不相交，0 为相交（包括相切），1 为球包括了长方体（包括内接）
*/
int KDTree::GetCircleTreeIntersectCondition(KDTreeNode* node, const Eigen::Vector3f& o, double radius)
{
    if (node == nullptr)
    {
        return -1;
    }
    /*
        计算是否相交的方法：
        参考：https://blog.csdn.net/noahzuo/article/details/52037151
        设球心是 O，长方体中心是 A，长方体边界结点是 B，半径是 r
        目标是计算出 O 到矩形的最短距离
        为了简化，可以利用对称，使 O，B 在 A 的第一象限方向
        计算 BO 向量，若某方向值为负，则最近距离为垂直到面
        设 B'O 向量为 BO 向量中，若值 >=0 则保留，<0 则置为 0 的向量
        则 |B'O| 为最短距离，判读 |B'O| 与 r 大小即可
    */
    Eigen::Vector3f AB, AO, BO, BO_dis;
    AB = (node->reigon_max - node->reigon_min) / 2;
    AO = o - (node->reigon_max + node->reigon_min) / 2;
    for (int i = 0; i < 3; ++i)
    {
        if (AO[i] < 0) AO[i] = -AO[i];
    }
    BO = AO - AB;
    for (int i = 0; i < 3; ++i)
    {
        if (BO[i] < 0) BO_dis[i] = 0;
        else BO_dis[i] = BO[i];
    }
    if (BO_dis.norm() > radius)
    {
        // 相离
        return -1;
    }
    else if ((AO + (node->reigon_max - node->reigon_min) / 2).norm() <= radius)
    {
        /*
            判断长方体是否在球内：
            由于限定了第一象限，所以圆心距离最远的点在 B 对称的点
            判断该点是否在球内即可
        */
        return 1;
    }
    else
    {
        // 其他情况为只相交
        return 0;
    }
}

/**
 * @brief 递归查找半径内点
 * @param node 目前结点
 * @param target 目标坐标
 * @param radius 指定半径
*/
void KDTree::GetPointWithinRadiusSearch(KDTreeNode* node, const Eigen::Vector3f& target, double radius)
{
    if (node == nullptr)
    {
        return;
    }
    int condition = GetCircleTreeIntersectCondition(node, target, radius);
    if (condition == 1)
    {
        // 范围内的所有结点都在球内，加入所有点后返回即可
        for (int i = node->id_min; i <= node->id_max; ++i)
        {
            KNearestQueue.push(std::make_pair((particles[i].coordinate - target).norm(), particles[i]));
        }
        return;
    }
    else if (condition == 0)
    {
        // 有部分重合，判断自身，之后交给左右子树进行搜索
        if ((node->particle.coordinate - target).norm() <= radius)
        {
            KNearestQueue.push(std::make_pair((node->particle.coordinate - target).norm(), node->particle));
        }
        GetPointWithinRadiusSearch(node->son[0], target, radius);
        GetPointWithinRadiusSearch(node->son[1], target, radius);
    }
    else
    {
        // 没有重合，则子树也不会有重合，返回
        return;
    }
}