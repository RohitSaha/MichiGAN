image=('56000' '56002' '56006' '56008' '57011' '67172')
reference=('56000' '56002' '56006' '56008' '57011' '67172')
#label=('56021' '56031' '56011' '56075' '57011' '67172')
#orient=('56320' '57120' '56099' '56091' '57011' '67172')

for img in ${image[@]}; do
    for lab in ${reference[@]}; do
            python inference.py --name /home/ubuntu/checkpoint/test --gpu_ids 0 --inference_ref_name $lab --inference_tag_name $img --inference_orient_name $lab --netG spadeb --which_epoch 49 --use_encoder --noise_background --expand_mask_be --expand_th 5 --use_ig --load_size 128 --crop_size 128 --add_feat_zeros --data_dir /home/ubuntu/MichiGAN_FFHQ
    done
done
