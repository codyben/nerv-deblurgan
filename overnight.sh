# python3 train_nerv.py -e 40   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1     --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1      --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv     -b 1  --lr 0.0005 --norm none --act swish --dump_images
# python3 train_nerv.py -e 40   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1     --outf bunny_ab_db --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1      --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv     -b 1  --lr 0.0005 --norm none  --act swish    --weight /home/cody/CSC7760/NeRV/output/bunny_ab/bunny/embed1.25_40_512_1_fc_9_16_26__exp1.0_reduce2_low96_blk1_cycle1_gap1_e40_warm8_b1_conv_lr0.0005_cosine_Fusion6_Strd5,2,2,2,2_SinRes_actswish_/model_train_best.pth --eval_only  --dump_images


# python3 train_nerv.py -e 40   --lower-width 96 --num-blocks 1 --dataset dog --frame_gap 1     --outf dog_db_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1      --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv     -b 1  --lr 0.0005 --norm none  --act swish    --weight /home/cody/CSC7760/NeRV/output/dog_normal_ab/dog/embed1.25_40_512_1_fc_9_16_26__exp1.0_reduce2_low96_blk1_cycle1_gap1_e40_warm8_b1_conv_lr0.0005_cosine_Fusion6_Strd5,2,2,2,2_SinRes_actswish_/model_train_best.pth --eval_only  --dump_images

python3 train_nerv.py -e 40   --lower-width 96 --num-blocks 1 \
--dataset dynamic_bike --frame_gap 1     --outf bike_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1      \
--single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv     -b 1  --lr 0.0005 --norm none --act swish --dump_images
