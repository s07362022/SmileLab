# -*- coding: utf-8 -*-
from distutils import extension
import cv2
from os import makedirs, cpu_count
from tqdm import tqdm
import numpy as np
import os
import sys
import multiprocessing
from itertools import repeat, count
from contextlib import ContextDecorator
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

try:
    import os
    vipshome = r'F:\source\vips\vips-dev-8.11\bin'
    os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
    import pyvips
except:
    print("improt pyvips fail.")


# Config
level = 2
wsi_root_path = r'F:\nuck\dataset\rawData\Folder 18 LPI'               # WSI Path 
patch_png_save_path = r'F:\nuck\dataset\patch'         # Save path
uuid = '18-25950A3'                                                                  # WSI Filename
ext = '.mrxs'                                                                      # WSI Type
patch_size = 512
stride = 512
borderthreshold = 255                                   



class Timer(ContextDecorator):
    def __init__(self, message, time_log_path):
        self.message = message
        self.time_log_path = time_log_path
        self.end = None

    def elapsed_time(self):
        self.end = perf_counter()
        return self.end - self.start

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if not self.end:
            self.elapsed_time()
        print("{} : {}".format(self.message, self.end - self.start))
        
        with open(self.time_log_path, 'a+') as f:
            out_str = ','.join([str(self.message),str(self.end - self.start)])
            f.write(out_str)
            f.close()


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype='float32') / 255.0
    R, G, B = background
    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')

def detect_background(img, threshold):
    img = np.array(img)
    if img.mean() > threshold:
        return False
    return True


# +
def save(datas):    
    file_name, file_path, img,  start_loc = datas
    x, y = start_loc
    cv2.imwrite(f'{file_path}/{file_name}-{x}-{y}.png', img[:,:,::-1])
          
            
def run(start_locs, file_name, tiff_path, level, patch_size, borderthreshold, folder_patch):
    datas = []
    
    slide = pyvips.Image.new_from_file(tiff_path, level= level)
    region = pyvips.Region.new(slide)
    
    
    for start_loc in tqdm(start_locs):
        x, y = start_loc
        
        slide_fetch = region.fetch(int(x), int(y), patch_size, patch_size)
        patch = np.ndarray(buffer=slide_fetch, 
                           dtype=np.uint8,
                           shape=[patch_size, patch_size, slide.bands]) 
        patch = rgba2rgb(patch)
        if detect_background(patch.copy(), borderthreshold):  
            datas.append([file_name, folder_patch, patch, start_loc])
            
        if len(datas) >= 300:
            with ThreadPoolExecutor() as e:
                for r in e.map(save, datas):
                    if r is not None:
                        print(r)
            datas = []
        
    if len(datas):
        with ThreadPoolExecutor() as e:
            for r in e.map(save, datas):
                if r is not None:
                    print(r)

def get_start_loc(slide, patch_size, stride_size, num_workers):
    w, h = slide.width, slide.height
    start_loc_data = [(sx, sy)
                    for sy in range(0, h-patch_size, stride_size)
                    for sx in range(0, w-patch_size, stride_size)]
    
    chunk_size = len(start_loc_data) // num_workers
    start_loc_list_iter = [start_loc_data[i:i+chunk_size]
                           for i in range(0, len(start_loc_data), chunk_size)]
    
    return start_loc_list_iter, chunk_size           

def start(paramaters):
    slide = pyvips.Image.new_from_file(paramaters['tiff_path'], level = paramaters['level'])
    start_loc_list_iter, chunk_size = get_start_loc(
        slide, paramaters['patch_size'], paramaters['stride_size'], paramaters['process_num_workers'])
    with Timer(paramaters['file_name'], paramaters['time_log_path']) as timer:
        with ProcessPoolExecutor(max_workers=paramaters['process_num_workers']) as executor:
            executor.map(
                run,
                start_loc_list_iter,
                repeat(paramaters['file_name']),
                repeat(paramaters['tiff_path']),
                repeat(paramaters['level']),
                repeat(paramaters['patch_size']),
                repeat(paramaters['borderthreshold']),
                repeat(paramaters['folder_patch']),
            )
# -



if __name__ == '__main__':
    
    paramaters = {
        'time_log_path': f'{patch_png_save_path}/{uuid}_time_log_cut_patch_png.txt',
        'tiff_path':  os.path.join(wsi_root_path, uuid+ext),
        'file_name': uuid,
        'folder_patch': f'{patch_png_save_path}/patch/{uuid}',
        'patch_size': patch_size,
        'borderthreshold': borderthreshold,
        'stride_size': stride,
        'process_num_workers': cpu_count(),
        'level': int(level)
    }
    print(f"level: {level}, patch size: {paramaters['patch_size']}, stride size: {paramaters['stride_size']}")
    makedirs(paramaters['folder_patch'], exist_ok=True)

    start(paramaters)
