# Hawkeye Dataset

check `data_preparations.ipynb`, place all labels in the a folder like this -

```
base_data
├── 100k
├── 10k
├── bdd100k_box_track_20_labels_trainval
├── bdd100k_det_20_labels_trainval
├── bdd100k_drivable_labels_trainval
├── bdd100k_ins_seg_labels_trainval
├── bdd100k_lane_labels_trainval
├── bdd100k_pan_seg_labels_trainval
├── bdd100k_seg_track_20_labels_trainval
├── bdd100k_sem_seg_labels_trainval
├── image_labels
└── tmp_coco_data

13 directories, 0 files
```

The `base_data` dir can be a symlink to the actual folder placing all the label files.

```bash
ln -s /path/to/your/labels ./base_data
```
