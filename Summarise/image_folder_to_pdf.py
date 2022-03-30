from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

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


def image_list2pdf_position_cells(image_list, df, save_path, label=False, res=100):
    image_list.sort()

    for p_class in ["P", "PSA", "PS", "PA"]:
        subset_df = df[df["final_model_o_b"] == p_class]

        im_list=[]
        first = True
        for im_path in image_list:
            # check if the cell is part of the p_class
            cluster_id = int(im_path.split("/")[-1].split("_rate_map_Cluster_")[1].split("_")[0])
            session_id = im_path.split("/")[-1].split("_rate_map_Cluster_")[0]
            df_cell = subset_df[(subset_df["cluster_id"] == cluster_id) & (subset_df["session_id"] == session_id)]

            if len(df_cell)==1:
                p_class_cell = df_cell["final_model_o_b"].iloc[0]
                if p_class_cell == p_class:
                    if first:
                        im1 = open_image(im_path, label=label)
                        first = False
                    else:
                        im_list.append(open_image(im_path,label=label))

        pdf_filename = save_path+"_combined_"+p_class+".pdf"
        im1.save(pdf_filename, "PDF", resolution=res, save_all=True, append_images=im_list)




def image_list2pdf(image_list, save_path, label=False, res=100):
    image_list.sort()
    #image_list = sort_images(image_list)

    im_list=[]
    first = True
    for im_path in image_list:
        if first:
            im1 = open_image(im_path, label=label)
            first = False
        else:
            im_list.append(open_image(im_path,label=label))

    pdf_filename = save_path+"combined.pdf"
    im1.save(pdf_filename, "PDF", resolution=res, save_all=True, append_images=im_list)

def open_image(im_path, label, margin=10):
    rgba = Image.open(im_path)
    width, height = rgba.size
    draw = ImageDraw.Draw(rgba)
    textwidth, textheight = draw.textsize(im_path.split("/")[-1])
    x = width - textwidth - margin
    y = height - textheight - margin
    if label:
        draw.text((x, y), im_path.split("/")[-1])
    draw.text((0, 0),im_path.split("/")[-1],(0,0,0)) # this will draw text with Blackcolor and 16 size

    rgb = Image.new('RGB', rgba.size, (255, 255, 255))  # white background
    rgb.paste(rgba, mask=rgba.split()[3])               # paste using alpha channel as mask
    return rgb

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    path = "/mnt/datastore/Harry/Cohort8_may2021/summary/combined_grid_cells_figures"
    #image_list = [f.path for f in os.scandir(path) if f.is_file()]
    #image_list2pdf(image_list, save_path=path)

    path = "/mnt/datastore/Harry/Cohort7_october2020/summary/combined_grid_cells_figures/"
    image_list = [f.path for f in os.scandir(path) if f.is_file()]
    #image_list2pdf(image_list, save_path=path)

    path = "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Plots/Figures/average_firing_rate_maps"
    image_list = [f.path for f in os.scandir(path) if f.is_file()]
    #image_list2pdf(image_list, save_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Plots/Figures", label=True)

    path = "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Plots/Figures/instant_firing_rates"
    image_list = [f.path for f in os.scandir(path) if f.is_file()]
    #image_list2pdf(image_list, save_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Plots/Figures/instant_rates", label=True)


    # for looking at P cells
    df = pd.read_csv('/mnt/datastore/Harry/test_recording/AllMice_LinearModelResults.txt', sep="\t")
    path = "/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Plots/Figures/instant_firing_rates"
    image_list = [f.path for f in os.scandir(path) if f.is_file()]
    image_list2pdf_position_cells(image_list, df, save_path="/mnt/datastore/Harry/CurrentBiology_2022/Ramp_Plots/Figures/instant_rates", label=True)



if __name__ == '__main__':
    main()
