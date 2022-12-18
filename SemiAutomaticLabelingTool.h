#ifndef __SemiAutomaticLabelingTool__H
#define __SemiAutomaticLabelingTool__H

#include <string>
#include <vector>
#include <filesystem>

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

class SemiAutomaticLabel
{
public:
    SemiAutomaticLabel();
    ~SemiAutomaticLabel();
    void start();

private:
    void read_cfg_file(std::string cfg_file);
    static bool isBothSpace(char const &lhs, char const &rhs);
    void read_labels_file();
    void print_labels();
    void remove_json_file();
    bool check_use_frame_range(std::string mode);
    void generate_colors();
    void load_labeled_data(cv::Mat frame, std::filesystem::path save_txt_path);
    std::string remove_space(std::string line);
    std::vector<int> point2xyminmax(cv::Rect2i p);
    std::vector<int> point2xyminmax(std::vector<int> p);
    std::vector<int> clip(std::vector<int> p, int w, int h);
    void remove_labeled_data(std::filesystem::path save_txt_path, std::vector<int> pointxy, std::string choiced_class_name, int w, int h);

    float compute_iou(std::vector<int> box1, std::vector<int> box2);
    int compute_overlap(std::vector<int> box1, std::vector<int> box2);
    int compute_area(std::vector<int> box);

    std::vector<float> to_yolo_point(std::vector<int> p, bool from_xyminmax, int w, int h);
    void write_point2txt(std::vector<float> yolo_point, std::string choiced_class_name, std::filesystem::path save_txt_path);

private:
    std::filesystem::path out_dir;
    std::filesystem::path video_path;
    std::filesystem::path labels_file;

    bool read_from_video;
    std::filesystem::path frame_dir_path;

    bool write_txt;
    bool remove_json;
    bool show_video;
    bool delete_one_class;

    bool get_frame_range;
    std::vector<int> frame_range;
    std::string start_mode;

    std::vector<std::string> names;
    std::vector<std::vector<int>> colors;
};

#endif