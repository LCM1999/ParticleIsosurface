#include "utils.h"
#include <iostream>

#ifdef _WIN32
#include <Windows.h>
#include <Psapi.h>

void printMem() {
	PROCESS_MEMORY_COUNTERS pmc;
	if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
	{
		std::cout << float(pmc.WorkingSetSize) / 1024.0 / 1024.0 << "MB" << std::endl;
	}
}

#endif

void parseString(std::vector<std::string> *commList, const std::string& input, std::string sep)
{
	std::string comm;

	for (unsigned int i = 0; i < input.size(); i++)
	{
		const char ch = input[i];

		bool added = false;

		for (unsigned int s = 0; s < sep.size(); s++)
		{
			if (ch == sep[s])
			{
				commList->push_back(comm);
				comm = "";
				added = true;
				break;
			}
		}

		if (!added)
		{
			comm += ch;
		}
	}
	
	if (comm != "")
		commList->push_back(comm);
}

void replaceAll(std::string &str, const std::string &olds, const std::string &news)
{
	if (olds.empty())
		return;
	std::string::size_type iFind = 0;
	while (true)
	{
		iFind = str.find(olds, iFind);
		if (std::string::npos == iFind)
			break;
		str.replace(iFind, olds.size(), news);
		iFind += news.size();
	}
}
