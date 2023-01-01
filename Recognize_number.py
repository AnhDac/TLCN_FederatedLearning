import io
import cv2
import torch
from PIL import ImageTk, Image
from torchvision import transforms
from tkinter.filedialog import Open, SaveAs
from tkinter import Frame, Tk, BOTH, Text, Menu, END

model = torch.jit.load('model_cnn_glEps40_lcEps10_lcBs10.pt')
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

class Main(Frame):
    
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
  
    def initUI(self):
        self.parent.title("Recognize Number")
        self.pack(fill=BOTH, expand=1)
  
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)
  
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open", command=self.onOpen)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=fileMenu)
    def onOpen(self):
        global ftypes
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
  
        if fl != '':
            global imgin
            imgin = cv2.imread(fl,cv2.IMREAD_COLOR)

            with open(fl, 'rb') as f:
                image_bytes = f.read()

                tensor = transform_image(image_bytes=image_bytes)
                tensor=tensor.to(device)
                output = model.forward(tensor)
                
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, classes = torch.max(probs, 1)
                print( 'Class: ',classes.item())
                print( ' at confidence score:{0:.2f}'.format(conf.item()))

            img2 = cv2.resize(imgin,(200,200),interpolation=cv2.INTER_AREA)
            cv2.putText(img2,str(classes.item())+"",(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("ImageIn", img2)


root = Tk()
Main(root)
root.geometry("480x480+100+100")
root.mainloop()
