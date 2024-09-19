#include "utility.h"

std::vector<unsigned char> load_file(const std::string& file_path) {
    std::ifstream in(file_path, std::ios::in | std::ios::binary);
    if (!in.is_open()) return {};

    in.seekg(0, std::ios::end);
    size_t file_length = in.tellg();

    std::vector<unsigned char> file_data;
    if (file_length > 0) {
        in.seekg(0, std::ios::beg);
        file_data.resize(file_length);

        in.read((char*)&file_data[0], file_length);
    }

    in.close();
    return file_data;
}