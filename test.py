from mmdet.apis import init_detector, inference_detector, DetInferencer

# Initialize the DetInferencer
inferencer = DetInferencer('./configs/yolox/yolox_s_8xb8-300e_coco.py', "/data1/kawano/ota/mmdetection_1/work_dirs/yolox_s_simota_8xb8-300e_coco/epoch_300.pth")
img = "/data1/kawano/ota/mmdetection/demo/demo.jpg"
# Perform inference
result = inferencer(img, out_dir= "./test1")

inferencer =  DetInferencer('./configs/yolox/yolox_s_8xb8-300e_coco.py', "/data3/yamamura/mmdet_custom/mmdetection-kawano/work_dirs/yolox_s_ota_8xb8-300e_coco/epoch_300.pth")

result = inferencer(img, out_dir= "./test2")