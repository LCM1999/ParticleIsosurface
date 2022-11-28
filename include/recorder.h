#pragma once

#include <string>
#include <vector>

class TNode;
class SurfReconstructor;

class Recorder
{
private:
    std::string _Output_Dir;
    std::string _Frame_Name;
    SurfReconstructor* constructor;
public:
    Recorder(std::string& output_dir, std::string& frame_name, SurfReconstructor* surf_constructor);
    ~Recorder(){};

    void RecordProgress();
    void RecordParticles();
};


