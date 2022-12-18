import cv2
import yaml
import shutil
import numpy as np
from pathlib import Path
from numpy import ndarray
from yaml.loader import SafeLoader
from typing import Dict, List, Tuple, Union

class SemiAutomaticLabel:
    def __init__(self, cfg_path: str) -> None:
        self.args = self.read_cfg_file(cfg_path)
        self.set_params()
    
    @staticmethod
    def read_cfg_file(cfg_path: str) -> Dict:
        """
        Read config file
        """

        with open(cfg_path, 'r') as f:
            args = yaml.load(f, Loader=SafeLoader)

        return args
    
    @staticmethod
    def read_labels_file(labels_file: str) -> List:
        """
        Load labels file
        """

        with open(labels_file, 'r') as f:
            names = [x.strip() for x in f.readlines()]

        return names

    def set_params(self) -> None:
        """
        Transfor config file to variable
        """

        for k, v in zip(['OUTPUT_DIR', 'video_path', 'labels_file'], ['out_dir', 'video_path', 'labels_file']):
            setattr(self, v, Path(self.args['PATH'][k]))

        for k in ['read_from_video', 'frame_dir_path']:
            setattr(self, k, self.args['FRAME'][k])
        
        self.frame_dir_path = Path(self.frame_dir_path)

        for k in ['write_txt', 'remove_json', 'show_video', 'delete_one_class']:
            setattr(self, k, self.args['OPTION'][k])

        for k in ['get_frame_range', 'frame_range', 'start_mode']:
            setattr(self, k, self.args['ACTION'][k])

    @staticmethod
    def remove_json_file(out_dir: Path) -> None:
        """
        When you use the DarkMark to check the label, will generate json file automatically.
        Set `remove_json = True` to remove the json file.
        """
        [x.unlink() for x in out_dir.iterdir() if '.json' in x.name]

    def check_use_frame_range(self, mode: str = None) -> bool:
        if mode is None:
            return self.get_frame_range and self.frame_range is not None
        else:
            return self.get_frame_range and self.frame_range is not None and self.start_mode == mode

    def load_labeled_data(self, frame_: ndarray, txt_path: Path) -> None:
        """
        Read labeled file and draw box to the frame.
        """

        with open(str(txt_path), 'r') as f:
            for i, line in enumerate(f.readlines()):
                temp = line.strip().split()
                choiced_class_name, box = int(temp[0]), np.array(temp[1:], dtype=np.float32)
                box[[0, 2]] *= self.w
                box[[1, 3]] *= self.h
                cx, cy, w_, h_ = box

                xmin = int(cx - w_ / 2)
                ymin = int(cy - h_ / 2)
                xmax = int(cx + w_ / 2)
                ymax = int(cy + h_ / 2)

                cv2.rectangle(frame_, (xmin, ymin), (xmax, ymax), [
                            int(x) for x in self.colors[i % len(self.colors)]], 3)
                cv2.putText(frame_, self.names[choiced_class_name], (xmin, ymin - 10), cv2.FONT_HERSHEY_DUPLEX, 1, [
                            int(x) for x in self.colors[i % len(self.colors)]], 1, cv2.LINE_AA)

    @staticmethod
    def point2xyminmax(p: Tuple) -> List[int]:
        """
        Transfer point [xmin, ymin, w, h] to [xmin, ymin, xmax, ymax]
        """

        xmin = int(p[0])
        ymin = int(p[1])
        xmax = int(p[0] + p[2])
        ymax = int(p[1] + p[3])

        return [xmin, ymin, xmax, ymax]

    def clip(self, p: List[int]) -> List[int]:
        """
        Set limit to [0, w - 1] and [0, h - 1]
        """

        p[0] = np.clip(p[0], 0, self.w - 1)
        p[1] = np.clip(p[1], 0, self.h - 1)
        p[2] = np.clip(p[2], 0, self.w - 1)
        p[3] = np.clip(p[3], 0, self.h - 1)

        return p
    
    def remove_labeled_data(self, save_txt_path: Path, pointxy: List[int], choiced_class_name: str) -> None:
        """
        When you press 'r', a `delete` box will be drawn and
            all boxes touching the `delete` box will be deleted.

        if set `delete_one_class = True` only the corresponding
            label class will be deleted.
        """

        update_line = list()
        with open(str(save_txt_path), 'r') as f:
            for line in f.readlines():
                temp = line.split()
                choiced_class_name_old = int(temp[0])

                box = np.array([float(x) for x in temp[1:]], np.float32)
                box[[0, 2]] *= self.w
                box[[1, 3]] *= self.h
                box = box.astype(np.int32)
                xmin = box[0] - box[2] // 2
                ymin = box[1] - box[3] // 2
                xmax = box[0] + box[2] // 2
                ymax = box[1] + box[3] // 2

                box_ = [xmin, ymin, xmax, ymax]
                iou = self.compute_iou(pointxy, box_)

                if self.delete_one_class:
                    if iou > 0 and self.names[choiced_class_name_old] == choiced_class_name:
                        pass
                    else:
                        update_line.append(line)
                else:
                    if iou == 0:
                        update_line.append(line)

        if len(update_line) > 0:
            with open(str(save_txt_path), 'w') as f:
                for line in update_line:
                    f.write(line)
        elif save_txt_path.exists():
            save_txt_path.unlink()

    def compute_iou(self, box1: Union[ndarray, List[int]], box2: Union[ndarray, List[int]]) -> float:
        if not isinstance(box1, np.ndarray):
            box1 = np.array(box1)
        if not isinstance(box2, np.ndarray):
            box2 = np.array(box2)

        overlap = self.compute_overlap(box1, box2)
        union = self.compute_area(box1) + self.compute_area(box2) - overlap

        return overlap / union

    def compute_overlap(self, box1: Union[ndarray, List[int]], box2: Union[ndarray, List[int]]) -> int:
        xmin = np.maximum(box1[0], box2[0])
        ymin = np.maximum(box1[1], box2[1])
        xmax = np.minimum(box1[2], box2[2])
        ymax = np.minimum(box1[3], box2[3])

        box = np.array([xmin, ymin, xmax, ymax])
        area = self.compute_area(box)

        return area

    def compute_area(self, box: Union[ndarray, List[int]]) -> int:
        w = np.maximum(box[2] - box[0], 0)
        h = np.maximum(box[3] - box[1], 0)

        return w * h

    def to_yolo_point(self, p: List[int], from_xyminmax: bool = False) -> List[float]:
        """
        Transfer to [cx, cy, w, h]
        """

        if from_xyminmax:
            xmin, ymin, xmax, ymax = p
        else:
            xmin, ymin, xmax, ymax = self.point2xyminmax(p)
        w_ = xmax - xmin
        h_ = ymax - ymin

        return [(xmin + xmax) / 2 / self.w, (ymin + ymax) / 2 / self.h, w_ / self.w, h_ / self.h]

    def write_point2txt(self, yolo_point: List[float], choiced_class_name: str, save_txt_path: Path) -> None:
        """
        Record box infomation to txt file
        """

        with open(str(save_txt_path), 'a') as f1:
            f1.write(f'{self.names.index(choiced_class_name)} {yolo_point[0]} {yolo_point[1]} {yolo_point[2]} {yolo_point[3]}\n')

    def start(self) -> None:
        tracking = False
        removeItem = False
        display_time = 1

        self.tracker = cv2.legacy.TrackerCSRT_create()

        assert self.labels_file.exists(), '`labels_file` Not exists!'
        self.names = self.read_labels_file(self.labels_file)

        choiced_class_name = ''

        self.colors = [np.random.randint(0, high=200, size=(3,)) for x in range(10)]

        self.video_path = Path(self.video_path)
        self.out_dir = Path(self.out_dir) / self.video_path.parents[0] / self.video_path.stem

        if not self.out_dir.exists():
            self.show_video = False
            self.get_frame_range = False
            print('First time will create folder and catch every frame')
            print('Set:')
            print(f'\tshow_vid: {self.show_video}')
            print(f'\tget_frame_range: {self.get_frame_range}')

        self.out_dir.mkdir(parents=True, exist_ok=True)
        print('Output Path:', self.out_dir)

        if self.remove_json:
            self.remove_json_file(self.out_dir)

        if self.check_use_frame_range():
            if self.frame_range[0] < 2 or (self.frame_range[0] >= self.frame_range[1] and self.frame_range[1] != -1) or (self.frame_range[1] <= 2 and self.frame_range[1] != -1):
                print('\n`frame_range` invalid')
                print('Support')
                print('start from: 2')
                print('end to: -1')
                print('Example: [2, 100]')
                print('Example: [2, -1]')
                print('Example: [10, -1]')
                exit()

            print('Use frame range:', self.frame_range)

        if self.read_from_video:
            assert self.video_path.exists(), '`video_path` Not exists!'

            print(f'Read from video: {self.video_path}')
            cap = cv2.VideoCapture(str(self.video_path))

            if not cap.isOpened():
                print("Cannot open camera")
                exit()
        else:
            assert self.frame_dir_path.exists(), '`frame_dir_path` Not exists!'

            print('Read from frame')
            all_data = self.frame_dir_path.iterdir()
            frames_total = len([x for x in all_data if '.jpg' in str(x)])

        shutil.copyfile(self.labels_file, self.out_dir / f'{self.video_path.stem}.names')

        frame_id = 0

        while True:
            frame_id += 1
            frame_id_str = str(frame_id).zfill(6)
            save_img_path = self.out_dir / f'{frame_id_str}.jpg'
            save_txt_path = self.out_dir / f'{frame_id_str}.txt'

            if self.read_from_video:
                ret, frame = cap.read()
                if not ret:
                    print("End video")
                    break
                if not save_img_path.exists():
                    cv2.imwrite(str(save_img_path), frame)
            else:
                if frame_id > frames_total:
                    break
                frame = cv2.imread(str(save_img_path))

            if self.check_use_frame_range():
                if frame_id == self.frame_range[0] - 1:
                    frame = cv2.resize(frame, (1366, 768))
                    cv2.putText(frame, str(frame_id), (70, 50),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    if self.show_video:
                        cv2.imshow(str(self.video_path), frame)
                        cv2.waitKey(0)

                elif frame_id < self.frame_range[0]:
                    continue

                elif self.frame_range[1] != -1 and frame_id > self.frame_range[1]:
                    break

            frame = cv2.resize(frame, (1366, 768))
            cv2.putText(frame, str(frame_id), (70, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            self.h, self.w, _ = frame.shape

            if save_txt_path.exists():
                if save_txt_path.stat().st_size == 0:
                    save_txt_path.unlink()
                else:
                    self.load_labeled_data(frame, save_txt_path)

            keyName = cv2.waitKey(display_time)

            # Exit
            if keyName == ord('q'):
                break

            # Pause video
            elif keyName == ord(' '):
                cv2.waitKey(0)

            # Draw 1 box
            elif keyName == ord('a') or (self.check_use_frame_range('a') and frame_id == self.frame_range[0]):
                self.start_mode = ''
                input_msg = f'{"=" * 50}\n{self.names}\nInput Class Name:'
                choiced_class_name = input(input_msg).strip()
                while choiced_class_name not in self.names:
                    print(f'{choiced_class_name} not in names please input again')
                    choiced_class_name = input(input_msg).strip()
                area = cv2.selectROI(str(self.video_path), frame, showCrosshair=False, fromCenter=False)
                del self.tracker
                self.tracker = cv2.legacy.TrackerCSRT_create()
                self.tracker.init(frame, area)
                tracking = True

            # Cancel tracking
            elif keyName == ord('c'):
                tracking = False
                removeItem = False

            # Slow down
            elif keyName == ord('1'):
                display_time = np.minimum(display_time * 10, 1000)

            # Normal speed
            elif keyName == ord('2'):
                display_time = 1

            # Speed up
            elif keyName == ord('3'):
                display_time = np.maximum(display_time // 10, 1)

            elif keyName == ord('r') or (self.check_use_frame_range('r') and frame_id == self.frame_range[0]):
                self.start_mode = ''
                removeItem = True
                input_msg = f'{"=" * 50}\n{self.names}\ninput class name( delete ):'
                if self.delete_one_class:
                    choiced_class_name = input(input_msg).strip()
                    while choiced_class_name not in self.names:
                        print(f'{choiced_class_name} not in names please input again')
                        choiced_class_name = input(input_msg).strip()

                area = cv2.selectROI(str(self.video_path), frame, showCrosshair=False, fromCenter=False)
                del self.tracker
                self.tracker = cv2.legacy.TrackerCSRT_create()
                self.tracker.init(frame, area)
                tracking = True

            if tracking:
                success, point = self.tracker.update(frame)
                if success:
                    pointxy = self.point2xyminmax(point)
                    pointxy = self.clip(pointxy)

                    if removeItem and save_txt_path.exists():
                        self.remove_labeled_data(save_txt_path, pointxy, choiced_class_name)

                    yolo_point = self.to_yolo_point(pointxy, True)

                    cv2.rectangle(frame, (pointxy[0], pointxy[1]), (pointxy[2], pointxy[3]), (0, 0, 255), 3)
                    cv2.putText(
                        frame, choiced_class_name if not removeItem else 'Delete', (pointxy[0], pointxy[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    if self.write_txt and not removeItem:
                        self.write_point2txt(yolo_point, choiced_class_name, save_txt_path)

            if self.show_video:
                cv2.imshow(str(self.video_path), frame)
                if self.check_use_frame_range(' ') and frame_id == self.frame_range[0]:
                    self.start_mode = ''
                    cv2.waitKey(0)

        if self.read_from_video:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    cfg_file_path = './config_LabelTool.yaml'
    labelTool = SemiAutomaticLabel(cfg_file_path)
    labelTool.start()

