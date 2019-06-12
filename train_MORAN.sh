GPU=0
CUDA_VISIBLE_DEVICES=${GPU} \
python main.py \
	--train_nips /home/lz/reg_dataset/NIPS2014/NIPS2014 \
	--train_cvpr /home/lz/reg_dataset/CVPR2016/CVPR2016 \
	--valroot /home/lz/reg_dataset/svt_p_645 \
	--workers 2 \
	--batchSize 64 \
	--niter 10 \
	--lr 1 \
	--cuda \
	--experiment output/ \
	--displayInterval 100 \
	--valInterval 1000 \
	--saveInterval 40000 \
	--adadelta \
	--BidirDecoder