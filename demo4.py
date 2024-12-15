from pycocotools.coco import COCO


# 物体検出用のアノテーション情報
anno_path = "./data/coco/annotations/instances_8.json"
coco_ins = COCO(anno_path)

# アノテーション情報の可視化
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

fig = plt.figure(figsize=(20,80))


# 指定した画像 ID に対応する画像の情報を取得する。
for i in range(8):
    img_id = coco_ins.getImgIds()[i]
    img_info = coco_ins.loadImgs(img_id)
    w = img_info[0]['width']
    h = img_info[0]['height']
    aspect = w / h
    img_path = f"./data/coco/train2017/{img_info[0]['file_name']}"

# 指定した画像IDに対応するアノテーションIDを取得する。
    anno_ids_ins = coco_ins.getAnnIds(img_id)

# 指定したアノテーション ID に対応するアノテーションの情報を取得する。
    annos_ins = coco_ins.loadAnns(anno_ids_ins)
    ax = fig.add_subplot(4, 2, i + 1, title = f"img_id = {img_id}")

# 画像とアノテーション結果の描写
    img = plt.imread(img_path)
    ax.imshow(img, aspect = aspect)
    coco_ins.showAnns(annos_ins)

plt.savefig("sin.jpg")