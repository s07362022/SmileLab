# import pyvips
try:
    import os
    vipshome = r'F:\source\vips\vips-dev-8.11\bin'
    os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
    import pyvips
except:
    print("improt pyvips fail.")
import numpy as np
import json
import cv2
import time

dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}
def numpy2vips(np_array):
    height, width = np_array.shape
    linear_array = np_array.reshape(width * height * 1)
    image = pyvips.Image.new_from_memory(linear_array.data, width, height, 1,
                                      dtype_to_format[str(np_array.dtype)])
    return image
#20-00005-Masson
def genMask(file):
    mask_path = 'Z:\\[StudentResearch]\\113年_李念祖\\各位的DATA\\肝臟纖維化_宏瑜\\' + file + '-Masson.mrxs'
    annotation_path = 'Z:\\[StudentResearch]\\113年_李念祖\\各位的DATA\\肝臟纖維化_宏瑜\\' + file + '-Masson.json'
    GT_path = 'Z:\\[StudentResearch]\\113年_李念祖\\各位的DATA\\肝臟纖維化_宏瑜\\' + file + 'test_mask.tiff'
    vips_mask = pyvips.Image.new_from_file(mask_path,level=2)
    w, h = vips_mask.width, vips_mask.height
    # GT= np.zeros((h, w, 1), dtype=np.uint8)
    # empty = pyvips.Image.black(w, h)
    GT= np.zeros((w, h, 1), dtype='uint8') #buffer=vips_mask.write_to_memory()
    # GT = np.ndarray(buffer=empty.write_to_memory(), dtype=np.uint8, shape=[vips_mask.height, vips_mask.width, vips_mask.bands])
    annotation_list = []
    with open(annotation_path) as f:
        annotation_list = json.load(f)['annotation']
    
    for annotation in annotation_list:
        if annotation['name'] == 'lumen':
            cv2.fillPoly(GT, [np.array(annotation['coordinates']) // 4], color=0)
        elif annotation['name'] == 'fibrosis':
            cv2.fillPoly(GT, [np.array(annotation['coordinates']) // 4], color=255)
    _, bin_GT = cv2.threshold(GT, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    opened_GT = cv2.morphologyEx(bin_GT, cv2.MORPH_OPEN, kernel)
    vips_GT = numpy2vips(opened_GT)
    vips_GT.tiffsave(GT_path, compression='deflate', tile=True, bigtiff=True,
                     pyramid=True, miniswhite=False, squash=False)

file = '20-00005'
print('[INFO] start converting ' + file + ' annotation into mask ...')
t_start = time.time()
genMask(file)
t_end = time.time()
print('[INFO] convert ' + file + ' done')
print('Ground truth generation time: {:.1f} seconds'.format(t_end - t_start))