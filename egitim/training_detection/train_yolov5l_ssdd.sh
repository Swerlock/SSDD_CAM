python val.py \
  --weights runs/train/YOLOv5l_Original_SSDD/weights/best.pt \
  --data data/ssdd_custom.yaml \
  --img 640 \
  --task test \
  --device 0
