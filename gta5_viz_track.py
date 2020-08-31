import os
import cv2
import json
import sys
import argparse
import numpy as np

#converting current log.json files to coco format
def to_coco():
    with open(log_path) as json_file:
        data = json.load(json_file)            
        new_dict = dict()
        new_dict = {'annotations': []} 
        
        dct = {e['id']: [] for e in data}
        
        count = 0
        for annot in data:        
            if annot['id'] in dct.keys():
                for frame in annot['hist']:
                    frame_height = frame['frame_height']*1080
                    frame_width = frame['frame_width']*1920
                    frame_x = frame['frame_x']*1920
                    frame_y = frame['frame_y']*1080
                    bbox = [frame_x, frame_y, frame_width, frame_height]
                    bbox[2] = bbox[0] + bbox[2]
                    bbox[3] =  bbox[1] + bbox[3]

                    new_dict['annotations'].append({
                        "class_id": annot['id'],
                        "bbox": bbox,
                        "frameId": frame['frameId'],
                        "attributes": {'speed': frame['speed'], 
                                    'time': frame['time'],
                                    'bot_x': frame['bot_x']*500,
                                    'bot_y': frame['bot_y']*800,
                                    'frameFirstDet_height': frame['frameFirstDet_height'], 
                                    'frameFirstDet_width': frame['frameFirstDet_width'], 
                                    'frameFirstDet_x': frame['frameFirstDet_x'], 
                                    'frameFirstDet_y': frame['frameFirstDet_y'], 
                                    'ground_point_x': frame['ground_point_x'], 
                                    'ground_point_y': frame['ground_point_y'],}
                            })
                    count+=len(frame)
            else:
                continue
        print(dct[24])
        with open(args.folder_path+'/new_dict.json', 'w') as json_file:
            json.dump(new_dict, json_file)

#visualisation 
def viz_track():
    mp4state = False

    for element in folders:
        if ".mp4" in element:
            video_file_path = args.folder_path + os.sep + element
            mp4state = True
            
    if mp4state is False:
        sys.exit("no video file found")

    dct = dict()
    rect_color = (252, 3, 169)
    class_color = (3, 115, 252)
    track_color = (252, 148, 3)
    stopped_color = (93, 255, 61)

    cv2.namedWindow("window", cv2.WINDOW_NORMAL)      
    cv2.resizeWindow("window", 1280, 1024)

    cv2.namedWindow("topDown", cv2.WINDOW_NORMAL)      
    cv2.resizeWindow("topDown", 800, 500)
    
    cv2.namedWindow("ground_point", cv2.WINDOW_NORMAL)      
    cv2.resizeWindow("ground_point", 300, 250)

    with open(args.folder_path+'/new_dict.json') as json_file:
        data = json.load(json_file)
        dct = {e['frameId']: [] for e in data['annotations']}
        
        for annot in data['annotations']:
            if annot['frameId'] in dct.keys():
                element = {
                    'track_id': annot['class_id'], 
                    'bbox': annot['bbox'], 
                    'speed': round(annot['attributes']['speed'], 2), 
                    'topdown_xy': [annot['attributes']['bot_x'], annot['attributes']['bot_y']], 
                    'ground_point': [annot['attributes']['ground_point_x'], annot['attributes']['ground_point_y']]}
                dct[annot['frameId']].append(element)
            else:
                continue
        
        cap = cv2.VideoCapture(video_file_path)
        counter = 1
        img = np.zeros((500,800,3), np.uint8)
        img2 = np.zeros((100,120,3), np.uint8)
        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            
            if video_on:
                cv2.rectangle(frame, (0, 0, 1920, 1080), (229, 255, 254), -1)
            
            if counter in dct.keys():
                annots = dct[counter]
                for annot in annots:                        
                    dets = [int(x) for x in annot['bbox']]
                    cv2.rectangle(frame, (dets[0], dets[1]), (dets[2], dets[3]), rect_color, 2)

                    track_id = str(annot['track_id'])
                    cx = dets[0] + 12
                    cy = dets[1] - 10
                    cv2.putText(frame, f"track_id - {track_id}", (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 1, track_color, 2)
                    speed = str(annot['speed'])
                    cx = dets[0] + 30
                    cy = dets[1] - 40                        
                    cv2.putText(frame, f"speed - {speed}", (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 1, class_color, 2)

                    if annot['speed'] < 1 and annot['speed'] > 0:
                        cx = dets[0] + 60
                        cy = dets[1] - 70                        
                        cv2.putText(frame, f"stopped", (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 1, stopped_color, 2)
                        #cv2.circle(img,(topdown_xy[0], topdown_xy[1]), 1, stopped_color, -1)
                        #cv2.circle(img2,(ground_point[0], ground_point[1]), 1, stopped_color, -1)
                        #cv2.putText(img, f"stopped", (topdown_xy[0], topdown_xy[1]), cv2.FONT_HERSHEY_DUPLEX, 1, stopped_color, 2)

                    topdown_xy = [int(x) for x in annot['topdown_xy']]
                    cv2.circle(img,(topdown_xy[0], topdown_xy[1]), 3, (0,0,255), -1)

                    ground_point = [int(x) for x in annot['ground_point']]
                    cv2.circle(img2,(ground_point[0], ground_point[1]), 1, (0,0,255), 1)
                    

            cv2.imshow('window', frame)
            cv2.imshow('topDown', img)
            cv2.imshow('ground_point', img2)
            
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            counter += 1

    cv2.destroyAllWindows()

def is_stoped(speed):
    if speed < 1:
        return 1

def topDown_viz():
    print("test_dopdown")              


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test annotation')
    parser.add_argument('--folder_path', type=str, required=True, help="path to folder with video and unzipped annotations")
    parser.add_argument('--video', type=bool, required=False, help="path to folder with video and unzipped annotations")

    args = parser.parse_args()
    log_path = args.folder_path + os.sep + 'log.json'
    folders = os.listdir(args.folder_path)
    if args.video:
        video_on = 1
    else:
        video_on = 0

    to_coco()
    viz_track()