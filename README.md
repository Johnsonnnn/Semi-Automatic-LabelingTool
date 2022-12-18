# Semi-Automatic-LabelingTool

## Download
```bash
git clone https://github.com/Johnsonnnn/Semi-Automatic-LabelingTool.git
```

## Requirement
* **OpenCV**

### 1. Install Dependencies
```bash
cd Semi-Automatic-LabelingTool

# Python
bash install_python_dependency.sh

# C++
bash install_ubuntu_dependency.sh
```

## Prepare

### 1. Modify
1. `config_LabelTool.yaml` in line 3 and 4
```yaml
video_path:        "./Videos/test.mp4"
labels_file:       "./coco.names"
```
### 2. Option
File: `config_LabelTool.yaml`

```yaml
OUTPUT_DIR:        "./Output"
video_path:        "./Videos/06.mp4"
labels_file:       "./coco.names"

read_from_video:   When True, will read from video, otherwise read from saved image frames.
frame_dir_path:    When `read_from_video` is False, saved image frames will be read from this folder.

write_txt:         True
remove_json:       True
show_video:        True
delete_one_class:  False

get_frame_range:   False
frame_range:       [2, 100]
start_mode:        "r"
```

## Usage

### Python
```bash
cd Semi-Automatic-LabelingTool

python SemiAutomaticLabelingTool.py
```

### C++
```bash
cd Semi-Automatic-LabelingTool

mkdir build
cd build
cmake ..
make -j$(nproc)
cd ..

bash run_LabelingTool.sh
```
