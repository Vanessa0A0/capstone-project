# capstone-project
# 目錄
[保存和加載模型](#保存和加載模型)

[遷移學習 6 個步驟](#資源4)

[微調模型的方法](#資料1)

[Transfer Learning的方法](#資料2)

[資料3](#資料3)

## 保存和加載模型
[來源](https://www.tensorflow.org/tutorials/keras/save_and_load)
不是主要就不整理 但提供參考一下 需要的話我再整理詳細一點

## 資料1
微調模型的方法
[來源](https://kknews.cc/zh-tw/news/q2zpa5y.html)
##### **特徵提取**

我們可以將預訓練模型當做特徵提取裝置來使用。具體的做法是，將輸出層去掉，然後將剩下的整個網絡當做一個固定的特徵提取機，從而應用到新的數據集中。

##### **採用預訓練模型的結構**

我們還可以採用預訓練模型的結構，但先將所有的權重隨機化，然後依據自己的數據集進行訓練。

##### **訓練特定層，凍結其他層**

另一種使用預訓練模型的方法是對它進行部分的訓練。具體的做法是，將模型起始的一些層的權重保持不變，重新訓練後面的層，得到新的權重。在這個過程中，我們可以多次進行嘗試，從而能夠依據結果找到frozen layers和retrain layers之間的最佳搭配。

###### 如何使用與訓練模型
是由數據集大小和新舊數據集(預訓練的數據集和我們要解決的數據集)之間數據的相似度來決定的。

**場景三：數據集大，數據相似度不高**(與pre-trained model的訓練數據相比而言)  

在這種情況下，因為我們有一個很大的數據集，所以神經網絡的訓練過程將會比較有效率。然而，因為實際數據與預訓練模型的訓練數據之間存在很大差異，採用預訓練模型將不會是一種高效的方式。

因此最好的方法還是將預處理模型中的權重全都初始化後在新數據集的基礎上重頭開始訓練。

**場景四：數據集大，數據相似度高**

這就是最理想的情況，採用預訓練模型會變得非常高效。最好的運用方式是保持模型原有的結構和初始權重不變，隨後在新數據集的基礎上重新訓練。
![[Pasted image 20230504233748.png]]

## 資料2
Transfer Learning的方法
[來源](https://chiachun0818.medium.com/%E5%BF%AB%E9%80%9F%E7%90%86%E8%A7%A3pre-trained-model-transfer-learning%E4%B9%8B%E9%96%93%E5%B7%AE%E7%95%B0-%E4%B8%A6%E4%B8%94%E5%AF%A6%E4%BD%9Cpytorch%E6%8F%90%E4%BE%9B%E7%9A%84pre-trained-model-4a246a38463b)

進行遷移學習的方式有兩種：

1. 特徵擷取 (Feature Extraction)：凍結除了全連接層或者輸出層以外的權重，只訓練沒凍結的即可。

2. 微調 (Fine-tuning)：不採用隨機的權重初始，而是使用先前訓練好的權重當作初始值，在這邊可以全部都訓練也可以只訓練某一個部分，甚至說針對網路架構去做更動，也都算是Fine-tuning的範疇。

##### 使用Pre-Trained Model進行Inference
在導入模型的同時你可以定義是否要用預訓練好的模型還是只是單純的模型架構
~~~python
import torchvision.models as models
target_model = models.mobilenet_v2( )   # 沒導入預訓練的權重
target_model = models.mobilenet_v2(pretrained=True) # 導入預訓練的權重
~~~
官方的文件中有說建議是以下列的參數作為正規化的參數：
~~~python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
~~~
主要執行的部分我寫成透過命令列改變參數的方式，所以導入了argparse這個套件，並且將主要執行的程式寫在run_pretrained_model的副函式中，其中我提供了7個常見的模型選擇，然後自訂義要辨識的圖片、還有需要導入模型對應的標籤檔：
~~~python
if __name__=="__main__":
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='mobilenet', help="resnet18, alexnet, vgg16, densenet, inception, googlenet")
parser.add_argument('--image', default='image.jpg', help="Image Path")
parser.add_argument('--label', default='imagenet_classes.txt', help="Label Path")
args = parser.parse_args()
run_pretrained_model(args)
~~~

主要執行的函式如下，第一步是導入模型、標籤，接著設定PyTorch提供的資料處理torchvistion.transforms，要注意的是它吃的是PIL格式，一開始我用OpenCV讀圖檔，但當然可以一開始就用PIL，不過我提供給大家一個轉換的範例，在Inference前記得要將模型的模式調成eval (驗證模式)，透過np.argmax找到最大的數值的位置，再將對應的Label顯示出來
~~~python
def run_pretrained_model(args):
### 打印參數
print_args(args)
### 導入預訓練模型、標籤
target_model = choose_model(args.model)ls_label = read_txt(args.label)
### 先定義數據處理的部分
transform = T.Compose([ T.Resize(256),T.CenterCrop(224),T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
### 透過 OpenCV 導入圖片
path =   args.imageimg =   cv2.imread(f"{path}")
### PyTorch 吃PIL圖
img =   cv2.cvtColor(img, cv2.COLOR_BGR2RGB)img_pil   = Image.fromarray(img)img_tensor = transform(img_pil).unsqueeze(0)   
# 將數據加一維，模擬batch_size
### 推論 Infernece
target_model.eval()
predict   = target_model(img_tensor).squeeze(0)
### 找到最大的值 find max value
idx =   np.argmax(predict.detach().numpy())
print("\n\n辨識結果:   {}".format(ls_label[idx]))
~~~
其餘的程式碼分別是選擇模型 ( choose_model )、讀取txt檔 ( read_txt )、打印argparser的內容 ( print_args )：
~~~python
def choose_model(n):
trg_model = None
if  n=='resnet18':
trg_model = models.resnet18(pretrained=True)
elif   n=='alexnet':
trg_model = models.alexnet(pretrained=True)
elif   n=='vgg16':
trg_model = models.vgg16(pretrained=True)
elif   n=='densenet':
trg_model   = models.vgg16(pretrained=True)
elif   n=='inception':
trg_model = models.inception_v3(pretrained=True)
elif   n=='googlenet':
trg_model = models.googlenet(pretrained=True)
else:
trg_model = models.mobilenet_v2(pretrained=True)
return   trg_modeldef read_txt(path):
f =   open(path, 'r')
total =   f.readlines()
print(f'classes number:{len(total)}')
f.close()
return   total
def print_args(args):
for arg   in vars(args):
print (arg, getattr(args, arg))
~~~

詳細看原網址 有附github

## 資料3
[來源](https://github.com/vietanhdev/blazeface_keras#how-to-fine-tune-one-of-the-trained-models-on-your-own-dataset)
很完整但英文QWQ 我再翻譯一下

## 資源4
遷移學習 6 個步驟
[來源](https://www.v7labs.com/blog/transfer-learning-guide#h5)

##### 1.獲取預訓練模型
##### 2.創建基礎模型
##### 3.凍結圖層
##### 4.添加新的可訓練層
##### 5.訓練新層
##### 6.微調你的模型
