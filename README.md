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
video_path:        "./Videos/test.mp4"
labels_file:       "./coco.names"

read_from_video:   True     --> Will read from video, otherwise read from saved image frames.
frame_dir_path:    When `read_from_video` is False, saved image frames will be read from this folder.

write_txt:         True     --> The drawn box information and class name will be saved.
remove_json:       True     --> The json file will be remove.
                                If you use `DarkMark` you will need.
show_video:        True     --> Show the video.
delete_one_class:  True     --> When you use the hotkey 'r' only delete specific classes that touch `delete box`
                   False    --> All objects that touch the `delete box` will be deleted.
                   
get_frame_range:   True     --> Start and end at a specific frame.
frame_range:       [2, 100] --> Is [Start, End]
                                Support range: [2, -1]
                                  Start must be greater than or equal to 2
                                  End must be greater than start, -1 means until the End of the Video.
start_mode:        "r"      --> When `get_frame_range` is True, the specified HotKey will be activated on the first frame.
                                Support ["a" or "r" or " "] ---> " " means "space"
                                see more in "HotKey"
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
### HotKey
```text
* Support speed mode(Slow, Slow-1, Slow-2, Normal)
1 --> Slow down
2 --> Normal speed
3 --> Speed up

space --> Suspend

a --> Create a class and draw a box (Input class name is required)
q --> Exit
c --> Cancel tracking
r --> Draw a `Delete box`
      When option `delete_one_class = False`, all objects that touch the `delete box` will be deleted.
      When option `delete_one_class = True`, only delete specific classes that touch `delete box`
```




