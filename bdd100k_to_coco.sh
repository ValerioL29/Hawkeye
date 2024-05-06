# det:
#
# python3 -m bdd100k.label.to_coco -m det \                             
#     -i data/det_20_labels_trainval/det_val.json \
#     -o converted_data/det_20_labels_trainval/det_val.json \
#     --nproc 1

# box_track:
#
# python3 -m bdd100k.label.to_coco -m box_track \
#     -i base_data/box_track_labels_trainval/val \       
#     -o coco_data/box_track_labels_trainval/val/box_track_val.json \
#     --nproc 1

# seg_track:
#
# python3 -m bdd100k.label.to_coco -m seg_track \
#     -i base_data/seg_track_labels_trainval/rles/val \
#     -o coco_data/seg_track_labels_trainval/val/seg_track_val.json \
#     --nproc 1
# -mb base_data/seg_track_labels_trainval/bitmasks/train \

# ins_seg:
#
# python3 -m bdd100k.label.to_coco -m ins_seg \
#     -i base_data/bdd100k_ins_seg_labels_trainval/rles/ins_seg_val.json \
#     -o data/coco_data/ins_seg_labels_trainval/val/ins_seg_val.json \
#     --nproc 1
python3 -m bdd100k.label.to_coco -m ins_seg \
    -i base_data/bdd100k_drivable_labels_trainval/rles/drivable_train.json \
    -o data/coco_data/drivable_labels_trainval/train/drivable_train.json \
    --nproc 1

# pan_seg:
# 
# python3 -m bdd100k.label.to_coco_panseg \
#     -i base_data/bdd100k_pan_seg_labels_trainval/bitmasks/train \
#     -o data/coco_data/pan_seg_labels_trainval/train/pan_seg_train.json \
#     -pb data/coco_data/pan_seg_labels_trainval/masks/train/ \
#     --nproc 1
