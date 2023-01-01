## Federated Learning Application

- Link Word and Powerpoint Report: https://drive.google.com/drive/folders/14RbzLmWIDp3hEvh_jj1UHTU4fWykglhs?usp=sharing

### Student Made:
+ Nguyễn Anh Đắc - 19133020
+ Nguyễn Thanh Tân Kỷ - 19133031

### Descroption
- Đây là một ứng dụng của Federated Learning vào mô hình học máy. 
- Ứng dụng thuật toán CNN kết hợp triển khia Federated Learning vào mô hình để thực hiện nhận dạng chữ số viết tay từ tập diệu MNIST.
- MNIST: là một bộ dữ liệu bao gồm các hình ảnh viết tay được chuẩn hóa và cắt ở giữa. Nó có hơn 60.000 hình ảnh đào tạo và 10.000 hình ảnh thử nghiệm. Đây là một trong những bộ dữ liệu được sử dụng nhiều nhất cho mục đích học tập và thử nghiệm. Để tải và sử dụng tập dữ liệu, chúng ta có thể nhập bằng cú pháp bên dưới sau khi cài đặt gói torchvision:
> torchvision.datasets.MNIST()

###  Set Up 
> uninstall torch, torchvision
> pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
 
### How to Run
#### Easy run: 
> python main_fed.py

#### Run with option params:
> python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 20 --gpu 0 --local_ep 10 --local_bs 20 --num_users 50 

#### Run with a special version python (Ex: 3.10):  
> py -3.10 main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 20 --gpu 0 --local_ep 10 --local_bs 20 --num_users 50 

- After run, we save best model with name + information of some params, it have type: "namefile".pt. This file be use for recognize image in my application basic.
### Run application basic
> python Recognize_number.py
- select image and see the result.

#### References
- Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561
