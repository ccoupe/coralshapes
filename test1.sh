#! /bin/bash
python3 detect_image.py   --model coral/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite   --labels coral/coco_labels.txt --input test_image/person.jpg  --out foo.jpg
