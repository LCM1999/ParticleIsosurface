#include <filesystem>
#include <iostream>

#include <pvdReader.hpp>

bool readShonDyParticlesPVD(
    const std::string dir_path,
    const std::string pvd_file,
    bool &is_const_radius,
    double &radius,
    std::vector<std::string> &files
) {
    std::string pvd_path = dir_path + "/" + pvd_file;
    if (! std::filesystem::exists(pvd_path))
    {
        std::cout << "Error: cannot find PVD file: " << pvd_path << std::endl;
        return false;
    }

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(pvd_path.c_str());
    if (!result)
    {
        std::cout << "Load result: " << result.description() << std::endl;
        return false;
    }
    
    std::cout << doc.first_child().name() << std::endl;
    
    return true;
}