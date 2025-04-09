#pragma once
#include <vector>
#include <string>
//#include <Eigen/Dense>


int sign(const float &x)
{
	return x > 0 ? 1 : -1;
}

void printMem();

void parseString(std::vector<std::string> *commList, const std::string& input, std::string sep);

void replaceAll(std::string &str, const std::string &olds, const std::string &news);

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

