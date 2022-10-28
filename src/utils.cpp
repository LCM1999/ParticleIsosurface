#include "utils.h"


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
