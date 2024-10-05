import os.path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import openslide
import valis.slide_io
from valis import registration
from valis.micro_rigid_registrar import MicroRigidRegistrar
from valis import registration, valtils

"""
不能创建新文件夹，只能把空文件夹中当做目标文件夹
统一使用ome.tiff格式，其他格式先转换成ome.tiff
"""

demo_slide_src_dir = "/home/jmc/PycharmProjects/valis-main/examples/example_datasets/ihc"
demo_results_dst_dir = "/home/jmc/PycharmProjects/valis-main/examples/demo/expected_results"
demo_registered_slide_dst_dir = "/home/jmc/PycharmProjects/valis-main/examples/demo/expected_results/registered_slides"

parent_dir = "/home/jmc/tools/scale-25pc/"
dataset_src_dir = os.path.join(parent_dir, "dataset_liver")
results_dst_dir = os.path.join(parent_dir, "expected_results")


def democonvert2ome(src_dir, dst_dir):
    # Convert the slides in slide_src_dir to ome.tiff format
    start = time.time()
    for i in range(1, 12):
        if i == 6 or i == 8:
            src_dir = src_dir + str(i) + "_CD31.jpg"
            dst_dir = dst_dir + str(i) + "_CD31.ome.tiff"
        elif i % 2 == 1:
            src_dir = src_dir + str(i) + "_PAS.jpg"
            dst_dir = dst_dir + str(i) + "_PAS.ome.tiff"
        else:
            src_dir = src_dir + str(i) + "_aSMA.jpg"
            dst_dir = dst_dir + str(i) + "_aSMA.ome.tiff"
        valis.slide_io.convert_to_ome_tiff(src_dir, dst_dir, 0)
    end = time.time()
    elapsed = end - start
    print("elapsed time is " + str(elapsed / 60) + " minutes")



def newest_find_tissue_cnt(tif_path, wsiType):
    tif_slide = openslide.OpenSlide(tif_path)
    wsi_width, wsi_high = tif_slide.dimensions
    print(wsi_width, wsi_high)
    low_thresh_val = 200
    if (wsiType == 0):
        first_thresh_val = 251.5
        second_thresh_val = 247.0
    elif (wsiType == 1):
        first_thresh_val = 254.0
        second_thresh_val = 248.0
    else:
        raise Exception("请确定输入图像的格式类型")

    # 寻找左边界
    print("寻找左边界")
    scan_step = 6144
    left_boundery = 0
    right_boundery = wsi_width
    up_boundery = 0
    down_boundery = wsi_high

    epoch = 0

    while epoch <= 1:
        num_w = (right_boundery - left_boundery) // scan_step
        num_h = (down_boundery - up_boundery) // scan_step
        print(num_w, num_h)
        for i in range(num_w - 1):
            flag = False
            for j in range(num_h):
                checkimg = np.array(
                    tif_slide.read_region((left_boundery + scan_step * i, up_boundery + scan_step * j), 0,
                                          (scan_step, scan_step)))
                checkmean = np.mean(np.array(checkimg))
                # print(checkmean)

                if (checkmean < first_thresh_val and checkmean > low_thresh_val):
                    nextCheckimg = np.array(
                        tif_slide.read_region((left_boundery + scan_step * (i + 1), up_boundery + scan_step * j), 0,
                                              (scan_step, scan_step)))
                    if (np.mean(np.array(nextCheckimg)) < second_thresh_val):
                        print(np.mean(np.array(nextCheckimg)))
                        flag = True
                        break
            if (flag):
                left_boundery = left_boundery + i * scan_step
                break

        print("左边界：", left_boundery)
        print("i=", i, " j=", j)

        # 寻找右边界
        print("寻找右边界")
        for i in range(0, num_w - 2):
            flag = False
            for j in range(num_h):
                checkimg = np.array(
                    tif_slide.read_region((right_boundery - scan_step * (i + 1), up_boundery + scan_step * j), 0,
                                          (scan_step, scan_step)))
                checkmean = np.mean(np.array(checkimg))
                # print(checkmean)

                if (checkmean < first_thresh_val and checkmean > low_thresh_val):
                    nextCheckimg = np.array(
                        tif_slide.read_region((right_boundery - scan_step * (i + 2), up_boundery + scan_step * j), 0,
                                              (scan_step, scan_step)))
                    if (np.mean(np.array(nextCheckimg)) < second_thresh_val):
                        print(np.mean(np.array(nextCheckimg)))
                        flag = True
                        break
            if (flag):
                right_boundery = right_boundery - i * scan_step
                break

        print("右边界：", right_boundery)
        print("i=", i, " j=", j)

        # 寻找上下边界
        if (right_boundery < left_boundery):
            raise Exception("右边界小于左边界")

        print("寻找上边界")
        for i in range(0, num_h - 1):
            flag = False
            for j in range(0, (right_boundery - left_boundery) // scan_step):
                checkimg = np.array(
                    tif_slide.read_region((left_boundery + scan_step * j, up_boundery + scan_step * i), 0,
                                          (scan_step, scan_step)))
                checkmean = np.mean(np.array(checkimg))
                # print(checkmean)

                if (checkmean < first_thresh_val and checkmean > low_thresh_val):
                    nextCheckimg = np.array(
                        tif_slide.read_region((left_boundery + scan_step * j, up_boundery + scan_step * (i + 1)), 0,
                                              (scan_step, scan_step)))
                    if (np.mean(np.array(nextCheckimg)) < second_thresh_val):
                        print(np.mean(np.array(nextCheckimg)))
                        flag = True
                        break
            if (flag):
                up_boundery = up_boundery + i * scan_step
                break

        print("上边界：", up_boundery)
        print("i=", i, " j=", j)

        # 下边界
        print("寻找下边界")
        for i in range(0, num_h - 2):
            flag = False
            for j in range(0, (right_boundery - left_boundery) // scan_step):
                checkimg = np.array(
                    tif_slide.read_region((left_boundery + scan_step * j, down_boundery - scan_step * (i + 1)), 0,
                                          (scan_step, scan_step)))
                checkmean = np.mean(np.array(checkimg))
                # print(checkmean)

                if (checkmean < first_thresh_val and checkmean > low_thresh_val):
                    nextCheckimg = np.array(
                        tif_slide.read_region((left_boundery + scan_step * j, down_boundery - scan_step * (i + 2)), 0,
                                              (scan_step, scan_step)))
                    if (np.mean(np.array(nextCheckimg)) < second_thresh_val):
                        print(np.mean(np.array(nextCheckimg)))
                        flag = True
                        break
            if (flag):
                down_boundery = down_boundery - i * scan_step
                break

        print("下边界：", down_boundery)
        print("i=", i, " j=", j)
        epoch = epoch + 1
        print("第" + str(epoch) + "轮边界结果")
        print(left_boundery, right_boundery, up_boundery, down_boundery)
    print(left_boundery, right_boundery, up_boundery, down_boundery)
    print(left_boundery, up_boundery, right_boundery - left_boundery, down_boundery - up_boundery)

    return int(left_boundery), int(up_boundery), int(right_boundery) - int(left_boundery), int(down_boundery) - int(
        up_boundery)


def convert2ome(src_dir, dst_dir):
    # 将 liver dataset 转换成 ome.tiff format
    start = time.time()
    for i in range(1, 3):
        src_dir = "/home/jmc/tools/scale-25pc/origin_dataset_liver/he/" + str(i) + ".tif"
        dst_dir = "/home/jmc/tools/scale-25pc/dataset_liver/he/" + str(i) + ".ome.tiff"
        print(src_dir)
        print(dst_dir)
        # xywh = newest_find_tissue_cnt(src_dir, i - 1)
        valis.slide_io.convert_to_ome_tiff(src_dir, dst_dir, 0, None)

    # for i in range(2, 3):
    #     src_dir = "/home/jmc/tools/scale-25pc/origin_dataset_liver/he/" + str(i) + ".tif"
    #     dst_dir = "/home/jmc/tools/scale-25pc/dataset_liver/he/" + str(i) + ".ome.tiff"
    #     print(src_dir)
    #     print(dst_dir)
    #     xywh2 = newest_find_tissue_cnt(src_dir, 1)
    #     valis.slide_io.convert_to_ome_tiff(src_dir, dst_dir, 0, None, xywh2)

    end = time.time()
    elapsed = end - start
    print("converting elapsed time is " + str(elapsed / 60) + " minutes")


def reading_ome(src_dir):
    # 读取 ome.tiff 格式的文件
    reader = valis.slide_io.VipsSlideReader(src_dir)
    image = reader.slide2image(level=0)
    plt.imshow(image)
    plt.show()


def reading_sliedes(src_dir):
    # downs ample liver dataset using slide2vips and return pyvips.Image
    slide_src_f = "/home/jmc/tools/scale-25pc/dataset_liver/1.ome.tiff"
    series = 0
    # 获取reader
    reader_cls = valis.slide_io.get_slide_reader(slide_src_f, series=series)
    reader = reader_cls(slide_src_f, series=series)

    # 获取每一层的图像大小(width,height)
    pyramid_level_sizes_wh = reader.metadata.slide_dimensions

    # 获取每个像素的physical units
    pixel_physical_size_xyu = reader.metadata.pixel_physical_size_xyu

    # 获取通道名
    channel_names = reader.metadata.channel_names

    # 获取原始xml元数据
    original_xml = reader.metadata.original_xml

    # 以numpy数组的形式获取图像金字塔第三层
    img = reader.slide2image(level=1)

    # 以pyvips.Image格式获取全分辨率图像
    full_rez_vips = reader.slide2vips(level=0)

    # 第0层切分ROI，以numpy格式返回
    rio_img = reader.slide2image(level=0, xywh=(100, 100, 500, 500))
    plt.imshow(rio_img)
    plt.show()


"""
   平均convert时长22m
   ## BUGGY 使用liver数据集没有输出 --- 文件夹命名问题                                    
"""


# registration to liver dataset
# liver_src_dir = "/home/jmc/tools/scale-25pc/origin_dataset_liver/ihc/"
# liver_dst_dir = "/home/jmc/tools/scale-25pc/expected_results/liver_registered_slides/"
# registered_dst_dir = "/home/jmc/tools/scale-25pc/expected_results/res/"
#
def register_and_save(src_dir, dst_dir):
    start = time.time()
    registrar = registration.Valis(src_dir, dst_dir)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()  # slides ————> image
    img = "/home/jmc/tools/scale-25pc/origin_dataset_liver/ihc/1.ome.tiff"

    load_registrar = valis.registration.load_registrar(
        "/home/jmc/tools/scale-25pc/expected_results/miceKidney_registered_slides/data/_registrar.pickle")
    reader = valis.slide_io.VipsSlideReader(src_dir + "1_PAS.ome.tiff")
    image = reader.slide2image(level=0)
    reg = valis.registration.Slide(src_dir + "1_PAS.ome.tiff", image, load_registrar, reader)
    reg.warp_and_save_slide(dst_dir + "res1_PAS", crop="overlap")  # image ————> slides
    stop = time.time()
    elapsed = stop - start
    print("registration time is " + str(elapsed / 60) + " minutes")
    registration.kill_jvm()


def register_and_saves(src_dir, dst_dir):
    registrar = registration.Valis(src_dir, dst_dir)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    # load  registrar
    # registrar = valis.registration.load_registrar(os.path.join(results_dst_dir, "data/_registrar.pickle"))
    registered_slide_dst_dir = os.path.join(registrar.dst_dir, "registered_slides", registrar.name)
    registrar.warp_and_save_slides(dst_dir=registered_slide_dst_dir, Q=100, compression="lzw", crop="overlap")
    registration.kill_jvm()


def cnames_from_filename(src_f):
    """Get channel names from file name
    Note that the DAPI channel is not part of the filename
    but is always the first channel.

    """
    f = valtils.get_name(src_f)
    return ["DAPI"] + f.split(" ")


def register_and_merges(src_dir, dst_dir):
    registrar = registration.Valis(src_dir, dst_dir)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    # load  registrar
    # registrar = valis.registration.load_registrar(os.path.join(results_dst_dir, "data/_registrar.pickle"))
    registered_slide_dst_dir = os.path.join(registrar.dst_dir, "registered_slides", registrar.name)
    channel_name_dict = {f: cnames_from_filename(f) for
                         f in registrar.original_img_list}
    registrar.warp_and_merge_slides(dst_f=registered_slide_dst_dir, channel_name_dict=channel_name_dict,
                                    drop_duplicates=True, src_f_list=registrar.original_img_list)
    registration.kill_jvm()


def micro_rigid_register_and_saves(src_dir, dst_dir, micro_reg_fraction=0.25):
    registrar = registration.Valis(src_dir, dst_dir, micro_rigid_registrar_cls=MicroRigidRegistrar)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    micro_non_rigid_registrar, micro_error_df = registrar.register_micro(max_non_rigid_registration_dim_px=2000)
    registered_slide_dst_dir = os.path.join(registrar.dst_dir, "registered_slides", registrar.name)
    registrar.warp_and_save_slides(dst_dir=registered_slide_dst_dir, Q=100, compression="lzw")

    # # 计算非刚性配准时 max_non_rigid_registration_dim_px 需要的大小
    # img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
    # min_max_size = np.min([np.max(d) for d in img_dims])
    # img_areas = [np.multiply(*d) for d in img_dims]
    # max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
    # micro_reg_size = np.floor(min_max_size * micro_reg_fraction).astype(int)
    #
    # # Perform high resolution non-rigid registration using 25% full resolution
    # micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px=micro_reg_size)
    # registered_slide_dst_dir = os.path.join(registrar.dst_dir, "registered_slides", registrar.name)
    # registrar.warp_and_save_slides(dst_dir=registered_slide_dst_dir, Q=100, compression="lzw")

    # We can also plot the high resolution matches using `Valis.draw_matches`:
    matches_dst_dir = os.path.join(registrar.dst_dir, "hi_rez_matches")
    registrar.draw_matches(matches_dst_dir)

    registration.kill_jvm()


def convert_ome_to_tif(src, dst):
    reader = valis.slide_io.VipsSlideReader(src)
    image = reader.slide2image(level=0)


if __name__ == "__main__":
    origin_ihc_src_dir = os.path.join(parent_dir, "origin_dataset_liver/ihc")
    origin_he_src_dir = os.path.join(parent_dir, "origin_dataset_liver/he")

    ihc_convert_dst_dir = os.path.join(parent_dir, "dataset_liver/ihc")
    he_convert_dst_dir = os.path.join(parent_dir, "dataset_liver/he")

    ihc_src_dir = os.path.join(dataset_src_dir, "ihc")
    he_src_dir = os.path.join(dataset_src_dir, "he")
    mixed_src_dir = os.path.join(dataset_src_dir, "mixed")

    ihc_dst_dir = os.path.join(results_dst_dir, "ihc_registration")
    he_dst_dir = os.path.join(results_dst_dir, "he2_registration")
    mixed_dst_dir = os.path.join(results_dst_dir, "liver_registration")

    print(origin_he_src_dir, he_convert_dst_dir)

    convert2ome(origin_he_src_dir, he_convert_dst_dir)
    # print(he_src_dir, he_dst_dir)
    register_and_saves(he_src_dir, he_dst_dir)
    # register_and_merges(ihc_src_dir, ihc_dst_dir)
    # micro_rigid_register_and_saves(ihc_src_dir, ihc_dst_dir)
    # convert_ome_to_tif("/home/jmc/tools/scale-25pc/expected_results/he_registration/he/registered_slides/he/1.ome.tiff", "/home/jmc/tools/scale-25pc/dataset_liver/converted/1.tif") # 爆内存
