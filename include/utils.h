#pragma once
#include <vector>
#include <string.h>
#include <Eigen/Dense>


template <class T>
int sign(Eigen::Vector<T, 4> &x)
{
    return x[3] > 0 ? 1 : -1;
}

void parseString(std::vector<std::string> *commList, const std::string& input, std::string sep);

template <class T>
void parseStringToElements(std::vector<T>* elements, std::string& input, std::string sep)
{
    size_t pos = input.find(sep);

    while (pos != input.npos)
    {
        if (typeid(T) == typeid(int))
        {
            elements->push_back(stoi(input.substr(0, pos)));
        }
        else if (typeid(T) == typeid(float))
        {
            elements->push_back(stof(input.substr(0, pos)));
        }
        else
        {
            return;
        }
        input = input.substr(pos + 1);
        pos = input.find(sep);
    }

    if (!input.empty())
    {
        if (typeid(T) == typeid(int))
        {
            elements->push_back(stoi(input));
        }
        else if (typeid(T) == typeid(float))
        {
            elements->push_back(stof(input));
        }
        else
        {
            return;
        }
        input.clear();
    }
}

