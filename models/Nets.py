from torch import nn
import torch
import torch.nn.functional as F

#module
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        #Conv2d:  Áp dụng tích chập 2D trên đầu vào
        self.convol1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.convol2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout2d: Xóa ngẫu nhiên trên toàn bộ kênh (một kênh là bản đồ tính năng 2D)
        self.conv2_drop = nn.Dropout2d()
        self.linear1 = nn.Linear(320, 50)
        self.linear2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        #max_pool2d:  Áp dụng tổng hợp tối đa 2D trên tín hiệu đầu vào bao gồm một số mặt phẳng đầu vào.
        x = F.relu(F.max_pool2d(self.convol1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.convol2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        #relu: Áp dụng thành phần hàm đơn vị tuyến tính đã chỉnh lưu.
        x = F.relu(self.linear1(x))
        x = F.dropout(x, training=self.training)
        x = self.linear2(x)
        return x

