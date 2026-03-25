from PIL import Image
import numpy as np
import glob
import os.path
import random as rd
import cv2

# Load a random image from the dataset
def load_random_image(i,path_source, size):

    raw_img_path = rd.choice(glob.glob(os.path.join(path_source, "IR", '*.g'))+glob.glob(os.path.join(path_source, "IR", '*.bmp'))) 
    print(raw_img_path)
    ref_img_path= os.path.join(path_source, "VIS",raw_img_path.split('/')[-1] )
    img = Image.open(raw_img_path)
    img2 = Image.open(ref_img_path)

    img_grey = img.resize(size)                 
    img_data = np.asarray(img_grey)

    img2_grey = img2.resize(size)                 
    img2_data = np.asarray(img2_grey)
 
    return img_data,raw_img_path,img2_data,ref_img_path


def save_to_file(index, image_raw_1, image_raw_2, path_dest, gt_4pt_shift,path,path2,src_input1,src_input2,dst):
    
    raw_path = path_dest
    print( raw_path +'//input1' )
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)

    
    raw_input1_path = raw_path +'/input_vis' 
    raw_input2_path = raw_path +'/input_ir'
    raw_label_path = raw_path +'/label'
    pt4_shift_path = raw_path + '/shift'
    
    if not os.path.exists(raw_input1_path):
        os.makedirs(raw_input1_path)
    if not os.path.exists(raw_input2_path):
        os.makedirs(raw_input2_path)
    if not os.path.exists(raw_label_path):
        os.makedirs(raw_label_path)
    if not os.path.exists(pt4_shift_path):
        os.makedirs(pt4_shift_path)


    raw_input1_path = raw_path +'//input_ir//' + index + '.png'
    raw_input2_path = raw_path +'//input_vis//'+ index + '.png'
    raw_label_path = raw_path +'//label//'+ index + '.png'
    pt4_shift_path = raw_path +'//shift//'+ index + '.npy'
    
    image1 = Image.fromarray(image_raw_1.astype('uint8')).convert('RGB')
    image2 = Image.fromarray(image_raw_2.astype('uint8')).convert('RGB')
    image1.save(raw_input1_path)
    image2.save(raw_input2_path)
    np.save(pt4_shift_path, gt_4pt_shift)



    

# Function to generate dataset
def generate_dataset(path_source, path_dest, height, width, data, box, scale):
    # print(data)
    for count in range(0, data):
        print(count)
        img, path,img2,path2 = load_random_image(count,path_source, [width, height])
        img=img.astype(np.uint16)
        img2=img2.astype(np.uint16)
        src_input1 = np.zeros([4, 2])
        dst = np.zeros([4, 2])

        oo=1
        while oo==1:
            offset = np.empty(8, dtype=np.int8)
            offset[0] = rd.randint(-box/scale,0)
            offset[1] = rd.randint(-box/scale,0)
            offset[2] = rd.randint(0, box/scale)
            offset[3] = rd.randint(-box/scale,0)

            offset[4] = rd.randint(-box/scale,0)
            offset[5] = rd.randint(0, box/scale)

            offset[6] = rd.randint(0, box/scale)
            offset[7] = rd.randint(0, box/scale)
            x_start=max(offset[0]*-1,offset[4]*-1)
            y_start=max(offset[1]*-1,offset[3]*-1)
            x_end=max(offset[2],offset[6])
            y_end=max(offset[5],offset[7])

            if y_start+1<height-box-y_end-1 and x_start+1 < width-box-x_end-1:
                oo=0
            else:
                oo=1
        src_input1[0][0] = rd.randint(x_start+1 ,width-box-x_end-1)
        src_input1[0][1] = rd.randint(y_start+1, height-box-y_end-1)
               
        # Upper right
        src_input1[1][0] = src_input1[0][0] + box
        src_input1[1][1] = src_input1[0][1]
        # Lower left
        src_input1[2][0] = src_input1[0][0]
        src_input1[2][1] = src_input1[0][1] + box
        # Lower right
        src_input1[3][0] = src_input1[1][0]
        src_input1[3][1] = src_input1[2][1] 
        # Upper left
        dst[0][0] = src_input1[0][0] + offset[0]
        dst[0][1] = src_input1[0][1] + offset[1]
        # Upper righ
        dst[1][0] = src_input1[1][0] + offset[2]
        dst[1][1] = src_input1[1][1] + offset[3]
        # Lower left
        dst[2][0] = src_input1[2][0] + offset[4]
        dst[2][1] = src_input1[2][1] + offset[5]
        # Lower right
        dst[3][0] = src_input1[3][0] + offset[6]
        dst[3][1] = src_input1[3][1] + offset[7]


        source = np.zeros([4, 2])
        target = np.zeros([4, 2])
        source[0][0] = 0
        source[0][1] = 0
        source[1][0] = source[0][0] + box
        source[1][1] = source[0][1]
        source[2][0] = source[0][0]
        source[2][1] = source[0][1] + box
        source[3][0] = source[1][0]
        source[3][1] = source[2][1]


        target[0][0] = dst[0][0] - src_input1[0][0]
        target[0][1] = dst[0][1] - src_input1[0][1]
        target[1][0] = dst[1][0] - src_input1[0][0]
        target[1][1] = dst[1][1] - src_input1[0][1]
        target[2][0] = dst[2][0] - src_input1[0][0]
        target[2][1] = dst[2][1] - src_input1[0][1]
        target[3][0] = dst[3][0] - src_input1[0][0]
        target[3][1] = dst[3][1] - src_input1[0][1]


        gt_4pt_shift = np.zeros((8,1), dtype = np.float32)
        for i in range(4):
            gt_4pt_shift[2*i] = target[i][0] - source[i][0]
            gt_4pt_shift[2*i+1] = target[i][1] - source[i][1]
        h, status = cv2.findHomography(dst, src_input1)
        img_warped_ref = np.asarray(cv2.warpPerspective(img2, h, (width, height))).astype(np.uint8)
        
        # Generate input1
        x1 = int(src_input1[0][0])
        y1 = int(src_input1[0][1])
        image_raw_1 = img[y1:y1+box, x1:x1+box]
        
        # Generate input2
        x2 = int(src_input1[0][0])
        y2 = int(src_input1[0][1])

        image_raw_2 = img_warped_ref[y2:y2+box, x2:x2+box,...]
        save_to_file(str(count+1).zfill(6), image_raw_1, image_raw_2, path_dest, gt_4pt_shift,path,path2,src_input1,src_input1,dst)

raw_image_path = 'RoadScene'       
box_size = 320
height = 440
width = 480
scale = 5

dataset_size =1105
generate_image_path = 'roadscene_dataset'  
generate_dataset(raw_image_path, generate_image_path, height, width, dataset_size, box_size, scale)
