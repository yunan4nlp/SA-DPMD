export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

touch log
nohup python -u driver/Train.py --config ddp.cfg.bert.NS > log 2>&1 &
tail -f log
