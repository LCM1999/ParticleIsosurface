#pragma once

#include <string>
#include <vector>

class TNode;
class SurfReconstructor;

class Recorder
{
private:
    std::string _Output_Dir;
    std::string _Output_Prefix;
    SurfReconstructor* constructor;
public:
    Recorder(std::string& output_dir, std::string& output_prefix, SurfReconstructor* surf_constructor);
    ~Recorder(){};

    void RecordProgress(const int& index);
    void RecordParticles(const int& index);
};


