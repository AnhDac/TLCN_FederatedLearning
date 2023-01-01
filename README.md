###  Set Up
uninstall torch, torchvision
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
 
### Run
#### Easy run: 
python main_fed.py

#### Run with option params:
python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 20 --gpu 0 --local_ep 10 --local_bs 20 --num_users 50 

#### Run with a special version python (Ex: 3.10):  
py -3.10 main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 20 --gpu 0 --local_ep 10 --local_bs 20 --num_users 50 
