dataset_name="refcoco" # "refcoco", "refcoco+", "refcocog_g", "refcocog_u"
config_name="MiCA_base.yaml"
gpu=2
split_name="testB" # "val", "testA", "testB" 
model_path='/path/to/best_model.pth'  # set your checkpoint path
# Evaluation on the specified of the specified dataset
filename=$dataset_name"_$(date +%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=$gpu \
python3 \
-u test.py \
--config config/$dataset_name/$config_name \
--path $model_path \
--opts TEST.test_split $split_name \
            TEST.test_lmdb datasets/lmdb/$dataset_name/$split_name.lmdb \
            DATA.dataset $dataset_name \
            DATA.mask_root datasets/masks/$dataset_name
