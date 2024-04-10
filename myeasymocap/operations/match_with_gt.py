import numpy as np
from easymocap.mytools.debug_utils import log, mywarn, myerror
from .iterative_triangulate import iterative_triangulate
from easymocap.mytools.triangulator import project_points, batch_triangulate

class MatchAndTrackWithGT():
    def __init__(self, cfg_match, cfg_track) -> None:
        self.cfg_match = cfg_match
        self.cfg_track = cfg_track
        
    
    def __call__(self, cameras, keypoints, meta, person_ids):
        #print('Match&Triangulate')
        #print('Cameras:', cameras)
        #print('Keypoints:', [keypoint.shape for keypoint in keypoints])
        #print('Meta:', meta)
        #print('Person IDs:', [person_ids.shape for person_ids in person_ids])
        
        
        persons = {}
        for view_id, pids in enumerate(person_ids):
            for id, person_id in enumerate(pids):
                if person_id not in persons:
                    persons[person_id] = {}
                persons[person_id][view_id] = keypoints[view_id][id]
        
        Pall = cameras['P']
        keypoints3d = []
        results = []
        for person_id, person in persons.items():
            #print('Person ID:', int(person_id))
            #print('Person:', person)
            
            keypoints_ = []
            cameras_ = []
            for view_id, keypoint in person.items():
                keypoints_.append(keypoint)
                cameras_.append(Pall[view_id])
            
            keypoints_ = np.stack(keypoints_)
            cameras_ = np.stack(cameras_)
        

            keypoints3d_ = batch_triangulate(keypoints_, cameras_, min_view=self.cfg_match.triangulate.min_view_body)
            #print('Keypoints3D:', keypoints3d_)
            keypoints3d.append(keypoints3d_)
            results.append({'id': int(person_id), 'keypoints3d': keypoints3d_, 'ages': 1})
            
        keypoints3d = np.stack(keypoints3d)
        pids = [p['id'] for p in results]
       #
        return {'results': results, 'keypoints3d': keypoints3d, 'pids': pids}
    
    