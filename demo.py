import json

def main():
    f0 = open('/data1/kawano/ota/mmdetection/data/coco/annotations/instances_train2017.json')
    d0 = json.load(f0)
    i0 = d0["images"][0:8]
    a0 = d0["annotations"]
    ids = [i0[i]["id"]for i in range(8)]
    anns_to_images = []
    l = []
    for ann in a0:
        if ann["image_id"] in ids:
            if ann["image_id"] == 309022:
                l.append(ann["bbox"])
            anns_to_images.append(ann)
            

    breakpoint()
    d2 = {'info': d0['info'],
          'licenses': d0['licenses'],
          'categories': d0['categories'],
          'images': i0,
          'annotations': anns_to_images}

    f2 = open('instances_8.json', 'w')
    json.dump(d2, f2)

if __name__ == '__main__':
    main()