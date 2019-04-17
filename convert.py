import os
import cv2
import datetime
import json
import getArea
import numpy as np
import shutil

CELL_DATASET_ORIGINAL_IMG_PATH = 'original/'
CELL_DATASET_TRAIN_IMG_PATH = 'train/'
CELL_DATASET_VAL_IMG_PATH = 'val/'
ANNOTATIONS_NAME = "via_region_data_40.json"


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }

    return image_info


def create_annotation_info(annotation_id, image_id, category_id, is_crowd,
                           area, bounding_box, segmentation):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,  # float
        "bbox": bounding_box,  # [x,y,width,height]
        "segmentation": segmentation  # [polygon]
    }

    return annotation_info


def get_segmenation(coord_x, coord_y):
    seg = []
    for x, y in zip(coord_x, coord_y):
        seg.append(x)
        seg.append(y)
    return [seg]


def convert(VIA_ORIGINAL_ANNOTATIONS_NAME, imgdir, annpath):
    """
    :param imgdir: directory for your images
    :param annpath: path for your annotations
    :return: coco_output is a dictionary of coco style which you could dump it into a json file
    as for keywords 'info','licenses','categories',you should modify them manually
    """

    # get the name(type) and supercategory(super_type) from VIA ANNOTATION
    annotations = json.load(open(VIA_ORIGINAL_ANNOTATIONS_NAME))
    annotations = list(annotations.values())  # don't need the dict keys
    annotations = [a for a in annotations if a['regions']]
    name_supercategory_dict = {}
    for a in annotations:
        names = [r['region_attributes']['type'] for r in a['regions']]
        supercategories = [r['region_attributes']['super_type'] for r in a['regions']]
        for index, name in enumerate(names):
            if name not in name_supercategory_dict.keys():
                name_supercategory_dict[name] = supercategories[index]

    coco_output = {}
    coco_output['info'] = {
        "description": "Cell Dataset",
        "url": "https://github.com/chang010453/",
        "version": "0.1.0",
        "year": 2019,
        "contributor": "DennisChang",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    coco_output['licenses'] = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    # coco_output['categories'] = [{
    #     'id': 1,
    #     'name': "中性杆状核粒细胞",
    #     'supercategory': "粒細胞系",
    #     },
    #     {
    #         'id': 2,
    #         'name': '中性中幼粒细胞',
    #         'supercategory': '粒細胞系',
    #     },
    #                 .
    #                 .
    #                 .
    # ]

    # get the coco category from dict
    coco_output['categories'] = []
    for i in range(len(name_supercategory_dict)):
        category = {'id': i + 1,
                    'name': list(name_supercategory_dict)[i],
                    'supercategory': name_supercategory_dict[list(name_supercategory_dict)[i]]
                    }
        coco_output['categories'].append(category)
    coco_output['images'] = []
    coco_output['annotations'] = []
    ##########################################################################################################

    ann = json.load(open(annpath))
    # annotations id start from zero
    ann_id = 0
    # in VIA annotations, [key]['filename'] are image name
    for img_id, key in enumerate(ann.keys()):
        filename = ann[key]['filename']
        img = cv2.imread(imgdir + filename)
        # make image info and storage it in coco_output['images']
        image_info = create_image_info(img_id, os.path.basename(filename), img.shape[:2])
        coco_output['images'].append(image_info)

        regions = ann[key]["regions"]
        # for one image ,there are many regions,they share the same img id
        for region in regions:
            cate = region['region_attributes']['type']
            # cate must in categories
            assert cate in [i['name'] for i in coco_output['categories']]
            # get the cate_id
            cate_id = 0
            for category in coco_output['categories']:
                if cate == category['name']:
                    cate_id = category['id']
            ####################################################################################################

            iscrowd = 0
            points_x = region['shape_attributes']['all_points_x']
            points_y = region['shape_attributes']['all_points_y']
            area = getArea.GetAreaOfPolyGon(points_x, points_y)

            min_x = min(points_x)
            max_x = max(points_x)
            min_y = min(points_y)
            max_y = max(points_y)
            box = [min_x, min_y, max_x - min_x, max_y - min_y]
            segmentation = get_segmenation(points_x, points_y)

            # make annotations info and storage it in coco_output['annotations']
            ann_info = create_annotation_info(ann_id, img_id, cate_id, iscrowd, area, box, segmentation)
            coco_output['annotations'].append(ann_info)
            ann_id = ann_id + 1

    return coco_output


# automatic split train and val
def train_val_split(annos_name, original_dir, train_dir, val_dir, move):
    annotations = json.load(open(annos_name))
    annotations = list(annotations.values())

    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]

    # get images in annotation
    total_images = [a['filename'] for a in annotations]

    # image index that will move to val
    val_index = np.random.choice(len(annotations), size=len(annotations) // 5, replace=False).tolist()
    train_index = [i for i in range(len(annotations))]
    for i in val_index:
        train_index.remove(i)

    # create train, val annos
    val_annos = {}
    train_annos = {}
    # move images to train, val folder
    if move:
        shutil.rmtree(val_dir)
        os.mkdir(val_dir)
        shutil.rmtree(train_dir)
        os.mkdir(train_dir)
        for i in val_index:
            shutil.copyfile(original_dir + total_images[i],
                            val_dir + total_images[i])
            val_annos[str(i)] = annotations[i]
        for i in train_index:
            shutil.copyfile(original_dir + total_images[i],
                            train_dir + total_images[i])
            train_annos[str(i)] = annotations[i]
    # not move images to train, val folder
    else:
        for i in val_index:
            val_annos[str(i)] = annotations[i]
        for i in train_index:
            train_annos[str(i)] = annotations[i]

    return train_annos, val_annos


if __name__ == '__main__':
    # get VIA annotations
    VIA_train_annos, VIA_val_annos = train_val_split(ANNOTATIONS_NAME, CELL_DATASET_ORIGINAL_IMG_PATH,
                                                     CELL_DATASET_TRAIN_IMG_PATH, CELL_DATASET_VAL_IMG_PATH, move=True)

    # save VIA annotations
    with open(CELL_DATASET_TRAIN_IMG_PATH + 'VIA_train_annos.json', 'w') as outfile:
        json.dump(VIA_train_annos, outfile, sort_keys=True, indent=4)

    with open(CELL_DATASET_VAL_IMG_PATH + 'VIA_val_annos.json', 'w') as outfile:
        json.dump(VIA_val_annos, outfile, sort_keys=True, indent=4)

    # convert VIA annotations to COCO annotations
    COCO_train_annos = convert(ANNOTATIONS_NAME, CELL_DATASET_TRAIN_IMG_PATH,
                               CELL_DATASET_TRAIN_IMG_PATH + 'VIA_train_annos.json')
    COCO_val_annos = convert(ANNOTATIONS_NAME, CELL_DATASET_VAL_IMG_PATH,
                             CELL_DATASET_VAL_IMG_PATH + 'VIA_val_annos.json')

    with open(CELL_DATASET_TRAIN_IMG_PATH + 'COCO_train_annos.json', 'w') as outfile:
        json.dump(COCO_train_annos, outfile, sort_keys=True, indent=4)

    with open(CELL_DATASET_VAL_IMG_PATH + 'COCO_val_annos.json', 'w') as outfile:
        json.dump(COCO_val_annos, outfile, sort_keys=True, indent=4)
