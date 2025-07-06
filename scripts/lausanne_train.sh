python utils/image_downsample.py data/lausanne_center/images --factor 4
echo 'Done: python utils/image_downsample.py data/lausanne_center/images --factor 4'
python utils/estimate_dataset_depths.py data/lausanne_center -d 4
echo 'Done: python utils/estimate_dataset_depths.py data/lausanne_center -d 4'
python main.py fit --config configs/lausanne_9000_coarse.yaml --n lausanne_coarse_9000
echo 'Done: python main.py fit --config configs/lausanne_9000_coarse.yaml --n lausanne_coarse_9000'
nohup python utils/partition_citygs.py --config_path configs/lausanne_train.yaml --force --origin auto
echo 'Done: nohup python utils/partition_citygs.py --config_path configs/lausanne_train.yaml --force --origin auto'
python utils/train_citygs_partitions.py -n lausanne_train -p lausanne_train
echo 'Done: python utils/train_citygs_partitions.py -n lausanne_train -p lausanne_train'
python utils/merge_citygs_ckpts.py outputs/lausanne_train
echo 'Done: python utils/merge_citygs_ckpts.py outputs/lausanne_train'
python main.py test --config outputs/lausanne_train/config.yaml --save_val --test_speed
echo 'Done: python main.py test --config outputs/lausanne_train/config.yaml --save_val --test_speed'
python utils/gs2d_mesh_extraction.py outputs/lausanne_train --voxel_size 0.01 --sdf_trunc 0.04
echo 'Done: python utils/gs2d_mesh_extraction.py outputs/lausanne_train --voxel_size 0.01 --sdf_trunc 0.04'
python tools/vectree_lightning.py --model_path outputs/lausanne_train --save_path outputs/lausanne_train/vectree --sh_degree 3 --gs_dim 3
echo 'Done: python tools/vectree_lightning.py --model_path outputs/lausanne_train --save_path outputs/lausanne_train/vectree --sh_degree 3 --gs_dim 3'
