#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <yaml-cpp/yaml.h>

#include <unistd.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "SemiAutomaticLabelingTool.h"

using namespace cv;
using namespace std;

SemiAutomaticLabel::SemiAutomaticLabel()
{
    this->read_cfg_file("./config_LabelTool.yaml");
}

SemiAutomaticLabel::~SemiAutomaticLabel()
{
}

void SemiAutomaticLabel::read_cfg_file(string cfg_path)
{
    /*
    Read config file
    */

    YAML::Node config = YAML::LoadFile(cfg_path);

    this->out_dir = filesystem::path(config["PATH"]["OUTPUT_DIR"].as<string>());
    this->video_path = filesystem::path(config["PATH"]["video_path"].as<string>());
    this->labels_file = filesystem::path(config["PATH"]["labels_file"].as<string>());

    this->read_from_video = config["FRAME"]["read_from_video"].as<bool>();
    this->frame_dir_path = filesystem::path(config["FRAME"]["frame_dir_path"].as<string>());

    this->write_txt = config["OPTION"]["write_txt"].as<bool>();
    this->remove_json = config["OPTION"]["remove_json"].as<bool>();
    this->show_video = config["OPTION"]["show_video"].as<bool>();
    this->delete_one_class = config["OPTION"]["delete_one_class"].as<bool>();

    this->get_frame_range = config["ACTION"]["get_frame_range"].as<bool>();
    this->frame_range = config["ACTION"]["frame_range"].as<vector<int>>();
    this->start_mode = config["ACTION"]["start_mode"].as<string>();

    return;
}

bool SemiAutomaticLabel::isBothSpace(char const &lhs, char const &rhs)
{
    return lhs == rhs && iswspace(lhs);
}

void SemiAutomaticLabel::read_labels_file()
{
    /*
    Load labels file
    */

    if (access(this->labels_file.c_str(), 0))
    {
        cout << "`labels_file` Not exists: " << this->labels_file << endl;
        exit(1);
    }

    ifstream ifs(this->labels_file, ios::in);
    if (!ifs.is_open())
    {
        cout << "Fail to open `labels_file`" << endl;
        exit(1);
    }
    string line;
    while (getline(ifs, line))
    {
        line = this->remove_space(line);

        if (line != "")
            this->names.push_back(line);
    }
    ifs.close();
    this->print_labels();
    return;
}

void SemiAutomaticLabel::print_labels()
{
    cout << "Labels: " << endl;
    int i = 0;
    for (auto it : this->names)
    {
        if (i == 0)
            cout << it;
        else
            cout << " " << it;
        ++i;
    }
    cout << endl;

    return;
}

void SemiAutomaticLabel::remove_json_file()
{
    /*
    When you use the DarkMark to check the label, will generate json file automatically.
    Set `remove_json = True` to remove the json file.
    */

    for (auto &file : filesystem::directory_iterator(this->out_dir))
    {
        if (file.path().string().find(".json") != string::npos)
            filesystem::remove(file.path());
    }
    return;
}

bool SemiAutomaticLabel::check_use_frame_range(string mode)
{
    if (mode == "")
        return this->get_frame_range && this->frame_range.size() == 2;
    return this->get_frame_range && this->frame_range.size() == 2 && this->start_mode == mode;
}

void SemiAutomaticLabel::generate_colors()
{
    vector<int> color;

    for (int i = 0; i < 10; ++i)
    {
        color.clear();
        for (int j = 0; j < 3; ++j)
            color.push_back(rand() % 256);
        this->colors.push_back(color);
    }
    return;
}

vector<string> SemiAutomaticLabel::get_line(string line)
{
    int curr = 0;
    int pos = 0;
    string sub_;
    vector<string> new_line;
    while (true)
    {
        pos = line.find("\r", curr);
        if (pos != -1)
        {
            sub_ = line.substr(curr, pos - curr);
            new_line.push_back(sub_);
        }
        else
            break;
        curr = pos + 1;
    }
    return new_line;
}

void SemiAutomaticLabel::load_labeled_data(Mat frame, filesystem::path save_txt_path)
{
    /*
    Read labeled file and draw box to the frame.
    */

    ifstream ifs(save_txt_path, ios::in);

    if (!ifs.is_open())
    {
        cout << "Fail to open: " << save_txt_path << endl;
        exit(1);
    }
    string line, item;
    int curr, pos;
    int choiced_class_name;
    vector<float> box;

    int h = frame.rows;
    int w = frame.cols;

    int color_len = this->colors.size();
    vector<int> color;

    for (int i = 0; getline(ifs, line); ++i)
    {
        auto it = unique(line.begin(), line.end(), this->isBothSpace);
        line.erase(it, line.end());

        if (line == "")
            continue;

        vector<string> lines = this->get_line(line);

        int c = 0;
        for (string new_line : lines)
        {
            color = this->colors[c % color_len];

            curr = 0;
            pos = 0;

            box.clear();
            for (int j = 0; pos != string::npos && pos != -1; ++j)
            {
                pos = new_line.find(" ", curr);
                item = new_line.substr(curr, pos - curr);
                if (j == 0)
                    choiced_class_name = stoi(item);
                else if (j == 1 || j == 3)
                    box.push_back(stof(item) * w);
                else if (j == 2 || j == 4)
                    box.push_back(stof(item) * h);

                curr = pos + 1;
            }

            int xmin = (int)(box[0] - box[2] / 2);
            int ymin = (int)(box[1] - box[3] / 2);
            int xmax = (int)(box[0] + box[2] / 2);
            int ymax = (int)(box[1] + box[3] / 2);

            rectangle(frame, Point(xmin, ymin), Point(xmax, ymax),
                      Scalar(color[0], color[1], color[2]),
                      3);
            putText(frame, this->names[choiced_class_name], Point(xmin, ymin - 10), FONT_HERSHEY_DUPLEX, 1,
                    Scalar(color[0], color[1], color[2]),
                    1, LINE_AA);
            ++c;
        }
    }
    ifs.close();

    return;
}

string SemiAutomaticLabel::remove_space(string line)
{
    auto it = unique(line.begin(), line.end(), this->isBothSpace);
    line.erase(it, line.end());

    if (line.find(" ") != string::npos)
        line = line.replace(line.find(" "), 1, "");

    return line;
}

vector<int> SemiAutomaticLabel::point2xyminmax(Rect2i p)
{
    /*
    Transfer point [xmin, ymin, w, h] to [xmin, ymin, xmax, ymax]
    */

    vector<int> result;
    result.push_back((int)p.x);
    result.push_back((int)p.y);
    result.push_back((int)(p.x + p.width));
    result.push_back((int)(p.y + p.height));

    return result;
}

vector<int> SemiAutomaticLabel::point2xyminmax(vector<int> p)
{
    /*
    Transfer point [xmin, ymin, w, h] to [xmin, ymin, xmax, ymax]
    */

    vector<int> result;
    result.push_back((int)p[0]);
    result.push_back((int)p[1]);
    result.push_back((int)(p[0] + p[2]));
    result.push_back((int)(p[1] + p[3]));

    return result;
}

vector<int> SemiAutomaticLabel::clip(vector<int> p, int w, int h)
{
    vector<int> result;
    result.push_back(min(max(p[0], 0), w - 1));
    result.push_back(min(max(p[1], 0), h - 1));
    result.push_back(min(max(p[2], 0), w - 1));
    result.push_back(min(max(p[3], 0), h - 1));

    return result;
}

void SemiAutomaticLabel::remove_labeled_data(filesystem::path save_txt_path, vector<int> pointxy, string choiced_class_name, int w, int h)
{
    /*
    When you press 'r', a `delete` box will be drawn and
        all boxes touching the `delete` box will be deleted.

    if set `delete_one_class = True` only the corresponding
        label class will be deleted.
    */

    vector<string> update_line;
    ifstream ifs(save_txt_path, ios::in);

    if (!ifs.is_open())
    {
        cout << "Fail to open: " << save_txt_path << endl;
        exit(1);
    }
    string line, item;
    int curr, pos;
    int choiced_class_name_old;
    vector<float> box;

    for (int i = 0; getline(ifs, line); ++i)
    {
        auto it = unique(line.begin(), line.end(), this->isBothSpace);
        line.erase(it, line.end());

        if (line == "")
            continue;

        vector<string> lines = this->get_line(line);

        for (string new_line : lines)
        {
            curr = 0;
            pos = 0;

            box.clear();

            for (int j = 0; pos != string::npos && pos != -1; ++j)
            {
                pos = new_line.find(" ", curr);
                item = new_line.substr(curr, pos - curr);

                if (j == 0)
                    choiced_class_name_old = stoi(item);
                else if (j == 1 || j == 3)
                    box.push_back(stof(item) * w);
                else if (j == 2 || j == 4)
                    box.push_back(stof(item) * h);

                curr = pos + 1;
            }

            int xmin = (int)(box[0] - box[2] / 2);
            int ymin = (int)(box[1] - box[3] / 2);
            int xmax = (int)(box[0] + box[2] / 2);
            int ymax = (int)(box[1] + box[3] / 2);

            vector<int> box_({xmin, ymin, xmax, ymax});

            float iou = this->compute_iou(pointxy, box_);

            if (this->delete_one_class)
            {
                if (iou > 0 && this->names[choiced_class_name_old] == choiced_class_name)
                {
                }
                else
                    update_line.push_back(new_line);
            }
            else
            {
                if (iou == 0)
                    update_line.push_back(new_line);
            }
        }
    }
    ifs.close();

    if (update_line.size() > 0)
    {
        ofstream ofs(save_txt_path);
        if (!ofs.is_open())
        {
            cout << "Fail to open: " << save_txt_path << endl;
            exit(1);
        }

        for (string x : update_line)
            ofs << x + "\r\n";
        ofs.close();
    }

    // Exists
    else if (!access(save_txt_path.c_str(), 0))
        filesystem::remove(save_txt_path);

    return;
}

float SemiAutomaticLabel::compute_iou(vector<int> box1, vector<int> box2)
{
    int overlap = this->compute_overlap(box1, box2);
    int union_ = this->compute_area(box1) + this->compute_area(box2) - overlap;

    float iou = (float)overlap / (float)union_;

    return iou;
}

int SemiAutomaticLabel::compute_overlap(vector<int> box1, vector<int> box2)
{
    int xmin = max(box1[0], box2[0]);
    int ymin = max(box1[1], box2[1]);
    int xmax = min(box1[2], box2[2]);
    int ymax = min(box1[3], box2[3]);

    vector<int> box({xmin, ymin, xmax, ymax});

    int area = this->compute_area(box);

    return area;
}

int SemiAutomaticLabel::compute_area(vector<int> box)
{
    int w = max(box[2] - box[0], 0);
    int h = max(box[3] - box[1], 0);

    return w * h;
}

vector<float> SemiAutomaticLabel::to_yolo_point(vector<int> p, bool from_xyminmax, int w, int h)
{
    /*
    Transfer to [cx, cy, w, h]
    */

    vector<int> temp;
    if (from_xyminmax)
        temp = p;
    else
        temp = this->point2xyminmax(p);

    float w_ = (float)(temp[2] - temp[0]);
    float h_ = (float)(temp[3] - temp[1]);

    vector<float> result;

    result.push_back((float)(temp[0] + temp[2]) / 2 / w);
    result.push_back((float)(temp[1] + temp[3]) / 2 / h);
    result.push_back(w_ / w);
    result.push_back(h_ / h);

    return result;
}

void SemiAutomaticLabel::write_point2txt(vector<float> yolo_point, string choiced_class_name, filesystem::path save_txt_path)
{
    /*
    Record box infomation to txt file
    */

    vector<string> update_line;
    string line;

    // Exists
    if (!access(save_txt_path.c_str(), 0))
    {
        ifstream ifs(save_txt_path, ios::in);

        if (!ifs.is_open())
        {
            cout << "Fail to open: " << save_txt_path << endl;
            exit(1);
        }

        while (getline(ifs, line))
            update_line.push_back(line);
        ifs.close();
    }

    int idx = 0;
    bool found = false;
    for (string x : this->names)
    {
        if (x == choiced_class_name)
        {
            found = true;
            break;
        }
        ++idx;
    }

    if (found)
    {
        line = format("%d %f %f %f %f\r\n", idx, yolo_point[0], yolo_point[1], yolo_point[2], yolo_point[3]);
        update_line.push_back(line);
    }
    else
    {
        cout << choiced_class_name << " Not found in names" << endl;
        exit(1);
    }

    ofstream ofs(save_txt_path);
    if (!ofs.is_open())
    {
        cout << "Fail to open: " << save_txt_path << endl;
        exit(1);
    }

    for (string x : update_line)
        ofs << x;
    ofs.close();

    return;
}

void SemiAutomaticLabel::start()
{
    bool tracking = false;
    bool removeItem = false;
    int display_time = 1;

    Ptr<Tracker> tracker = TrackerCSRT::create();

    this->read_labels_file();
    string choiced_class_name = "";

    this->generate_colors();

    filesystem::path mid_path = this->video_path.parent_path();
    if (this->video_path.string().find("./") != string::npos)
        mid_path = filesystem::path(this->video_path.parent_path().string().replace(this->video_path.string().find("./"), 2, ""));
    out_dir = this->out_dir / mid_path / this->video_path.stem();

    // `out_dir` not exists
    if (access(this->out_dir.c_str(), 0))
    {
        this->show_video = false;
        this->get_frame_range = false;
        cout << "First time will create folder and catch every frame" << endl
             << "Set:" << endl
             << "\tshow_vid: " << (this->show_video ? "True" : "False") << endl
             << "\tget_frame_range: " << (this->get_frame_range ? "True" : "False") << endl;
        filesystem::create_directories(this->out_dir);
    }

    cout << "Output Path: " << this->out_dir << endl;

    if (this->remove_json)
        this->remove_json_file();

    if (this->check_use_frame_range(""))
    {
        if (this->frame_range[0] < 2 || (this->frame_range[0] >= this->frame_range[1] && this->frame_range[1] != -1) || (this->frame_range[1] <= 2 && this->frame_range[1] != -1))
        {
            cout << endl
                 << "`frame_range` invalid" << endl
                 << "Support" << endl
                 << "start from: 2" << endl
                 << "end to: -1" << endl
                 << "Example: [2, 100]" << endl
                 << "Example: [2, -1]" << endl
                 << "Example: [10, -1]" << endl;
            exit(1);
        }
        cout << "Use frame range: "
             << "[" << this->frame_range[0] << ", " << this->frame_range[1] << "]" << endl;
    }

    VideoCapture cap;
    int frames_total = 0;

    if (this->read_from_video)
    {
        if (access(this->video_path.c_str(), 0))
        {
            cout << "`video_path` Not exists!" << endl;
            exit(1);
        }

        cout << "Read from video: " << this->video_path << endl;
        cap.open(this->video_path);

        if (!cap.isOpened())
        {
            cout << "Cannot open camera";
            exit(1);
        }
    }
    else
    {
        if (access(this->frame_dir_path.c_str(), 0))
        {
            cout << "`frame_dir_path` Not exists!" << endl;
            exit(1);
        }
        cout << "Read from frame" << endl;

        for (auto &file : filesystem::directory_iterator(this->frame_dir_path))
        {
            if (file.path().string().find(".jpg") != string::npos)
                ++frames_total;
        }
    }

    filesystem::path target_names_path = this->out_dir / this->video_path.replace_extension("names").filename();
    // Exists
    if (!access(target_names_path.c_str(), 0))
        filesystem::remove(target_names_path);
    filesystem::copy(this->labels_file, target_names_path);

    int frame_id = 0;

    stringstream ss;
    filesystem::path save_img_path;
    filesystem::path save_txt_path;

    bool ret;
    Mat frame;
    int h, w;
    int keyName;

    while (true)
    {
        ++frame_id;
        string frame_id_str;
        ss.clear();
        ss << setw(6) << setfill('0') << frame_id;
        ss >> frame_id_str;

        save_img_path = this->out_dir / (frame_id_str + ".jpg");
        save_txt_path = this->out_dir / (frame_id_str + ".txt");

        if (this->read_from_video)
        {
            ret = cap.read(frame);
            if (!ret)
            {
                cout << "End video" << endl;
                break;
            }
            if (access(save_img_path.c_str(), 0))
                imwrite(save_img_path, frame);
        }
        else
        {
            if (frame_id > frames_total)
                break;
            frame = imread(save_img_path);
        }

        if (this->check_use_frame_range(""))
        {
            if (frame_id == this->frame_range[0] - 1)
            {
                resize(frame, frame, Size(1366, 768));
                putText(frame, to_string(frame_id), Point(70, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 1, LINE_AA);

                if (this->show_video)
                {
                    imshow(this->video_path, frame);
                    waitKey(0);
                }
            }
            else if (frame_id < this->frame_range[0])
                continue;
            else if (this->frame_range[1] != -1 && frame_id > this->frame_range[1])
                break;
        }

        resize(frame, frame, Size(1366, 768));
        putText(frame, to_string(frame_id), Point(70, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 1, LINE_AA);

        h = frame.rows;
        w = frame.cols;

        // Exists
        if (!access(save_txt_path.c_str(), 0))
        {
            if (filesystem::file_size(save_txt_path) == 0)
                filesystem::remove(save_txt_path);
            else
                this->load_labeled_data(frame, save_txt_path);
        }

        keyName = waitKey(display_time);

        // Exit
        if (keyName == 'q')
            break;

        // Pause video
        else if (keyName == ' ')
            waitKey(0);

        // Draw 1 box
        else if (keyName == 'a' || (this->check_use_frame_range("a") && frame_id == this->frame_range[0]))
        {
            this->start_mode = "";
            string input_msg = "==================================================\n";
            cout << input_msg << endl;
            this->print_labels();
            cout << "Input Class Name: ";
            cin >> choiced_class_name;

            choiced_class_name = this->remove_space(choiced_class_name);

            while (find(this->names.begin(), this->names.end(), choiced_class_name) == this->names.end())
            {
                cout << choiced_class_name << " not in names please input again" << endl;
                cout << input_msg << endl;
                this->print_labels();
                cout << "Input Class Name: ";
                cin >> choiced_class_name;

                choiced_class_name = this->remove_space(choiced_class_name);
            }

            auto area = selectROI(this->video_path, frame, false, false);
            tracker = TrackerCSRT::create();
            tracker->init(frame, area);
            tracking = true;
        }

        // Cancel tracking
        else if (keyName == 'c')
        {
            tracking = false;
            removeItem = false;
        }

        // Slow down
        else if (keyName == '1')
            display_time = min(display_time * 10, 1000);

        // Normal speed
        else if (keyName == '2')
            display_time = 1;

        // Speed up
        else if (keyName == '3')
            display_time = max((int)(display_time / 10), 1);

        else if (keyName == 'r' || (this->check_use_frame_range("r") && frame_id == this->frame_range[0]))
        {
            this->start_mode = "";
            removeItem = true;
            if (this->delete_one_class)
            {
                string input_msg = "==================================================\n";
                cout << input_msg << endl;
                this->print_labels();
                cout << "Input Class Name( delete ): ";
                cin >> choiced_class_name;

                choiced_class_name = this->remove_space(choiced_class_name);

                while (find(this->names.begin(), this->names.end(), choiced_class_name) == this->names.end())
                {
                    cout << choiced_class_name << " not in names please input again" << endl;
                    cout << input_msg << endl;
                    this->print_labels();
                    cout << "Input Class Name( delete ): ";
                    cin >> choiced_class_name;

                    choiced_class_name = this->remove_space(choiced_class_name);
                }
            }

            auto area = selectROI(this->video_path, frame, false, false);
            tracker = TrackerCSRT::create();
            tracker->init(frame, area);
            tracking = true;
        }

        if (tracking)
        {
            Rect2i point;
            bool success = tracker->update(frame, point);
            if (success)
            {
                vector<int> pointxy = this->point2xyminmax(point);
                pointxy = this->clip(pointxy, w, h);

                // Exists
                if (removeItem && !access(save_txt_path.c_str(), 0))
                    this->remove_labeled_data(save_txt_path, pointxy, choiced_class_name, w, h);

                vector<float> yolo_point = this->to_yolo_point(pointxy, true, w, h);

                rectangle(frame, Point(pointxy[0], pointxy[1]), Point(pointxy[2], pointxy[3]),
                          Scalar(0, 0, 255),
                          3);

                string plot_msg = (!removeItem) ? choiced_class_name : "Delete";
                putText(frame, plot_msg, Point(pointxy[0], pointxy[1] - 10), FONT_HERSHEY_DUPLEX, 1,
                        Scalar(0, 0, 255),
                        1, LINE_AA);

                if (this->write_txt && !removeItem)
                    this->write_point2txt(yolo_point, choiced_class_name, save_txt_path);
            }
        }

        if (this->show_video)
        {
            imshow(this->video_path, frame);
            if (this->check_use_frame_range(" ") && frame_id == this->frame_range[0])
            {
                this->start_mode = "";
                waitKey(0);
            }
        }
    }
    if (this->read_from_video)
        cap.release();
    destroyAllWindows();
}

int main()
{
    SemiAutomaticLabel tool;
    tool.start();

    return 0;
}