import torch
import numpy as np
import os
import cv2
from os.path import join
import pickle
import json

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


class GTBBox:
    def __init__(self, folder="data/wildtrack/annotations_positions") -> None:
        
        self.folder = folder
        annots = os.listdir(folder)
        self.gtbbox = {}
        self.pids = {}
        for annot in annots: # 00000001.json, 00000002.json, ... frame ID
            annot_file = os.path.join(folder, annot)
            annot_data = read_json(annot_file)
            frame_id = int(annot.split(".")[0])
            if frame_id not in self.gtbbox:
                self.gtbbox[frame_id] = {}
            if frame_id not in self.pids:
                self.pids[frame_id] = {}
                
            for p in annot_data: # 0, 1, 2, ... person ID
                for view in p["views"]: # 1,3,4,5,6,7 (camera id)
                    view_id = int(view["viewNum"])+1 # index starts from 0 in the json file
                    pid = int(p["personID"])
                    xmin = int(view["xmin"])
                    ymin = int(view["ymin"])
                    xmax = int(view["xmax"])
                    ymax = int(view["ymax"])
                    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
                        continue
                    bbox = np.array([xmin, ymin, xmax, ymax, 0.99])
                    if view_id not in self.gtbbox[frame_id]:
                        self.gtbbox[frame_id][view_id] = []
                    if view_id not in self.pids[frame_id]:
                        self.pids[frame_id][view_id] = []
                    self.gtbbox[frame_id][view_id].append(np.array(bbox))
                    self.pids[frame_id][view_id].append(pid)
                    
        for frame_id in self.pids:
            for view in self.pids[frame_id]:
                assert len(self.pids[frame_id][view]) == len(set(self.pids[frame_id][view])), "Duplicate PIDs in the same frame and view!"
                self.pids[frame_id][view] = np.array(self.pids[frame_id][view])
                self.gtbbox[frame_id][view] = np.array(self.gtbbox[frame_id][view])
        
        
        
        print(f"Loaded {len(self.gtbbox)} frames")
        print(f"Loaded {len(self.gtbbox[frame_id])} views")
        

    def __call__(self, images, imgnames):
        squeeze = False
        # imgname = 'data/wildtrack\\images\\1\\00000000.jpg'
        if not isinstance(images, list):
            images = [images]
            imgnames = [imgnames]
            squeeze = True
        
        cam_ids = [int(imgname.split("\\")[-2]) for imgname in imgnames]
        frame_ids = [int(imgname.split("\\")[-1].split(".")[0]) for imgname in imgnames]
        
        gt = [self.gtbbox[frame_id][cam_id] for frame_id,cam_id in zip(frame_ids, cam_ids)]
        pid = [self.pids[frame_id][cam_id] for frame_id,cam_id in zip(frame_ids, cam_ids)]
        if squeeze:
            gt = gt[0]
            
        return {"bbox": gt,
                "person_ids": pid}

    

