from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import numpy as np

def sort_images(image_list):
    sorted_list=[]

    mouse_list = []
    for image_path in image_list:
        if image_path.endswith(".png"):
            mouse = image_path.split("/")[-1].split("_")[0]
            mouse_list.append(mouse)

    all_mice = np.unique(np.array(mouse_list))

    for mouse in all_mice:
        mouse_image_list = []

        for image_path in image_list:
            if image_path.endswith(".png"):
                mouse_i = image_path.split("/")[-1].split("_")[0]
                if mouse == mouse_i:
                    mouse_image_list.append(image_path)

        for j in range(0, 100):
            for image_path in mouse_image_list:
                if image_path.endswith(".png"):
                    day = int(image_path.split("/")[-1].split("_")[1].split("D")[1])
                    if day == j:
                        sorted_list.append(image_path)

    return sorted_list







def image_list2pdf(image_list, save_path, res=100):
    image_list = sort_images(image_list)

    im_list=[]
    first = True
    for im_path in image_list:
        if first:
            im1 = open_image(im_path)
            first = False
        else:
            im_list.append(open_image(im_path))

    pdf_filename = save_path+"/combined.pdf"
    im1.save(pdf_filename, "PDF", resolution=res, save_all=True, append_images=im_list)

def open_image(im_path):
    rgba = Image.open(im_path)
    rgb = Image.new('RGB', rgba.size, (255, 255, 255))  # white background
    rgb.paste(rgba, mask=rgba.split()[3])               # paste using alpha channel as mask
    return rgb

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    path = "/mnt/datastore/Harry/Cohort8_may2021/summary/combined_grid_cells_figures"
    image_list = [f.path for f in os.scandir(path) if f.is_file()]
    image_list2pdf(image_list, save_path=path)

    path = "/mnt/datastore/Harry/Cohort7_october2020/summary/combined_grid_cells_figures"
    image_list = [f.path for f in os.scandir(path) if f.is_file()]
    #image_list2pdf(image_list, save_path=path)



if __name__ == '__main__':
    main()
