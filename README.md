# capstone-project
# 目錄
[保存和加載模型](#保存和加載模型)

[遷移學習 6 個步驟](#資源4)

[微調模型的方法](#資料1)

[Transfer Learning的方法](#資料2)

[blazeface_keras/weight_sampling_tutorial](#資料3)

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

### 資料3
blazeface_keras/weight_sampling_tutorial.ipynb
[來源](https://github.com/vietanhdev/blazeface_keras#how-to-fine-tune-one-of-the-trained-models-on-your-own-dataset)
## 0. Our example
考慮以下示例。您有一個關於道路交通對象的數據集。讓此數據集包含以下感興趣的對像類的註釋：

['car', 'truck', 'pedestrian', 'bicyclist', 'traffic_light', 'motorcycle', 'bus', 'stop_sign']

也就是說，您的數據集包含 8 個對像類的註釋。

您現在想要在此數據集上訓練 SSD300。但是，與其經歷所有從頭開始訓練新模型的麻煩，不如使用在 MS COCO 上訓練過的經過全面訓練的原始 SSD300 模型，並在您的數據集上對其進行微調。

問題是：在 MS COCO 上訓練的 SSD300 預測了 80 個不同的類別，但您的數據集只有 8 個類別。 MS COCO 模型的分類層的權重張量沒有正確的形狀適合您的模型，該模型應該只學習 8 個類。


### 選項 1：忽略我們只需要 8 個類的事實
~~~python
了解原理用 不算重點
~~~

可能不那麼明顯但完全明顯的選擇是：我們可以忽略這樣一個事實，即經過訓練的 MS COCO 模型預測了 80 個不同的類別，但我們只想在 8 個類別上對其進行微調。==我們可以簡單地將註釋數據集中的 8 個類映射到 MS COCO 模型預測的 80 個索引中的任意 8 個索引。==我們數據集中的類 ID 可以是索引 1-8，它們可以是索引[0, 3, 8, 1, 2, 10, 4, 6, 12]，或 80 個中的任何其他 8 個。

這會奏效，甚至不是一個糟糕的選擇。由於 80 個類別中只有 8 個會得到訓練，該模型在預測其他 72 個類別時可能會逐漸變差，但無論如何我們都不關心它們，至少現在不關心。如果我們意識到我們現在想要預測超過 8 個不同的類別，那麼我們的模型在這個意義上將是可擴展的。我們想要添加的任何新類都可以只獲取任何一個剩餘的空閒索引作為其 ID。我們不需要更改模型的任何內容，只需對數據集進行相應的註釋即可。

儘管如此，在這個例子中我們不想走這條路。==我們不想攜帶過於復雜的分類器層的計算開銷，其中 90% 無論如何我們都不會使用==，但仍然需要在每次前向傳遞中計算它們的整個輸出。


### 選項 2：忽略那些導致問題的權重
~~~python
了解原理用 不算重點
~~~


我們可以構建一個具有 8 個類的新 SSD300，並將 MS COCO SSD300 的權重加載到除分類層之外的所有層。那行得通嗎？是的，那行得通。唯一的衝突是分類層的權重，我們可以通過簡單地忽略它們來避免這種衝突。雖然這個解決方案很簡單，但它有一個明顯的缺點：如果我們沒有為我們新的 SSD300 模型的分類層加載經過訓練的權重，那麼它們將被隨機初始化。我們仍然可以從所有其他層的訓練權重中獲益，==但分類器層需要從頭開始訓練。==

不是世界末日，但我們喜歡預訓練的東西，因為它為我們節省了大量的訓練時間。那麼我們還能做什麼呢？

### 選項 3：對導致問題的權重進行二次抽樣
~~~python
跟接下來步驟有關
~~~
與選項 2 中那樣丟棄有問題的權重不同，我們還可以對它們進行子採樣。如果 MS COCO 模型的分類層的權重張量沒有適合我們新模型的形狀，我們將製作 他們有正確的形狀。這樣我們仍然可以從這些分類層中的預訓練權重中獲益。似乎比選項 2 好得多。

這個例子的偉大之處在於：MS COCO 恰好包含我們關心的所有八個類。因此，當我們對分類層的權重張量進行子採樣時，我們不會隨機進行。相反，我們將從張量中挑選那些負責對我們關心的 8 個類進行分類的元素。

然而，即使數據集中的類別與任何完全訓練模型中的類別完全不同，使用完全訓練模型的權重仍然很有意義。任何經過訓練的權重始終是比隨機初始化更好的訓練起點，即使您的模型將在完全不同的對像類上進行訓練。

當然，如果你碰巧遇到相反的問題，你的數據集有更多的 類而不是你想要微調的訓練模型，那麼你可以簡單地在相反的方向做同樣的事情：而不是對分類層權重進行子採樣，然後你會上採樣 他們。工作方式與我們將在下面執行的操作相同。

~~~python
import h5py
import numpy as np
import shutil

from misc_utils.tensor_sampling_utils import sample_tensors
~~~

## 1.加載訓練好的權重文件並複制一份

首先，我們將加載包含我們需要的訓練權重的 HDF5 文件（源文件）。在我們的例子中，這是“VGG_coco_SSD_300x300_iter_400000.h5”（此 repo 的 README 中提供了下載鏈接），這是在 MS COCO 上訓練的原始 SSD300 模型的權重。

然後，我們將製作該權重文件的副本。該副本將是我們的輸出文件（目標文件）。
~~~python
# TODO: 設置要加載的源權重文件的路徑。
weights_source_path = '../../trained_weights/SSD/VGG_coco_SSD_300x300_iter_400000.h5'

# TODO: 設置目標權重文件的路徑和名稱
#       你想創建的。
weights_destination_path = '../../trained_weights/SSD/VGG_coco_SSD_300x300_iter_400000_subsampled_8_classes.h5'

#複製權重文件。
shutil.copy(weights_source_path, weights_destination_path)
  ~~~
~~~python
# 加載源權重文件和我們製作的副本。
# 我們將以只讀模式加載原始權重文件，這樣我們就不會搞砸任何東西。

weights_source_file = h5py.File(weights_source_path, 'r')
weights_destination_file = h5py.File(weights_destination_path)
  ~~~
  
## 2. 找出我們需要子採樣的權重張量

接下來，我們需要弄清楚我們需要對哪些權重張量進行子採樣。如上所述，除分類層外，所有層的權重都很好，我們不需要更改這些層的任何內容。

那麼SSD300有哪些分類層呢？他們的名字是：

~~~python
classifier_names = ['conv4_3_norm_mbox_conf',
                    'fc7_mbox_conf',
                    'conv6_2_mbox_conf',
                    'conv7_2_mbox_conf',
                    'conv8_2_mbox_conf',
                    'conv9_2_mbox_conf']
~~~


## 3. 找出要挑選的切片

以下部分是可選的。我將查看一個分類層並解釋我們想要做什麼，僅供您理解。如果您不關心這一點，請直接跳到下一節。

我們知道我們想要對哪些權重張量進行子採樣，但我們仍然需要決定我們想要保留這些張量的哪些（或至少多少）元素。讓我們來看看第一個分類器層，“conv4_3_norm_mbox_conf”。它的兩個權重張量 kernel 和 bias 具有以下形狀：

~~~python
conv4_3_norm_mbox_conf_kernel = weights_source_file[classifier_names[0]][classifier_names[0]]['kernel:0']
conv4_3_norm_mbox_conf_bias = weights_source_file[classifier_names[0]][classifier_names[0]]['bias:0']

print("Shape of the '{}' weights:".format(classifier_names[0]))
print()
print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)
~~~
Shape of the 'conv4_3_norm_mbox_conf' weights:

kernel:	 (3, 3, 512, 324)
bias:	 (324,)

所以最後一個軸有 324 個元素。這是為什麼？

-   MS COCO 有 80 個類，但該模型還有一個“背景”類，因此有效地製作了 81 個類。
    
-   “conv4_3_norm_mbox_loc”層為每個空間位置預測 4 個框，因此“conv4_3_norm_mbox_conf”層必須為這 4 個框中的每一個預測 81 個類中的一個。
    

這就是最後一個軸有 4 * 81 = 324 個元素的原因。

那麼我們想要這一層的最後一個軸有多少個元素？

讓我們做與上面相同的計算：

-   我們的數據集有 8 個類，但我們的模型也將有一個“背景”類，因此實際上有 9 個類。
    
-   我們需要為每個空間位置的四個框中的每一個預測這 9 個類別中的一個。
    

這使得 4 * 9 = 36 個元素。

現在我們知道我們要在最後一個軸中保留 36 個元素，並保持所有其他軸不變。但是我們想要原始 324 個元素中的哪 36 個元素呢？

我們應該隨機選擇它們嗎？如果我們數據集中的對像類與 MS COCO 中的類完全無關，那麼隨機選擇這 36 個元素就可以了（下一節也會介紹這種情況）。但在我們的特定示例中，隨機選擇這些元素是一種浪費。由於 MS COCO 恰好包含我們需要的 8 個類，因此我們不會隨機抽樣，而是只採用那些經過訓練的元素來預測我們的 8 個類。

以下是我們感興趣的 MS COCO 中 9 個類的索引：
~~~
[0, 1, 2, 3, 4, 6, 8, 10, 12]
~~~
上面的索引代表 MS COCO 數據集中的以下類：
~~~
['background', 'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic_light', 'stop_sign']
~~~
我是如何找到這些指數的？我只是在 MS COCO 數據集的註釋中查找了它們。

雖然這些是我們想要的類，但我們不希望它們按此順序排列。在我們的數據集中，類恰好按照本筆記本頂部所述的以下順序排列：
~~~
['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'traffic_light', 'motorcycle', 'bus', 'stop_sign']
~~~
例如， '紅綠燈' 在我們的數據集中是類 ID 5，但在 SSD300 MS COCO 模型中是類 ID 10。所以我實際想要選擇上面 9 個索引的順序是這樣的：
~~~
[0, 3, 8, 1, 2, 10, 4, 6, 12]
~~~
所以在 324 個元素中的每 81 個中，我想選擇上面的 9 個元素。這為我們提供了以下 36 個索引：

在 [7] 中：
~~~python
n_classes_source = 81
classes_of_interest = [0, 3, 8, 1, 2, 10, 4, 6, 12]

subsampling_indices = []
for i in range(int(324/n_classes_source)):
    indices = np.array(classes_of_interest) + i * n_classes_source
    subsampling_indices.append(indices)
subsampling_indices = list(np.concatenate(subsampling_indices))

print(subsampling_indices)
  ~~~
~~~
[0, 3, 8, 1, 2, 10, 4, 6, 12, 81, 84, 89, 82, 83, 91, 85, 87, 93, 162, 165, 170, 163, 164, 172, 166 , 168, 174, 243, 246, 251, 244, 245, 253, 247, 249, 255]
~~~
  

這些是我們要從偏置向量和內核張量的最後一個軸中選取的 36 個元素的索引。

這是 'conv4_3_norm_mbox_conf' 層。當然，我們還沒有實際對這一層的權重進行子採樣，我們只是想出了要保留哪些元素。下一節中的這段代碼將對所有分類器層執行子採樣。


## 4. 對分類器權重進行子採樣

本節中的代碼遍歷源權重文件的所有分類器層，並對每個分類器層執行以下步驟：

1.  從源權重文件中獲取內核和偏置張量。
    
2.  計算最後一個軸的子採樣索引。內核的前三個軸保持不變。
    
3.  用我們新創建的子採樣內核和偏置張量覆蓋目標權重文件中相應的內核和偏置張量。
    

第二步執行上一節中解釋的內容。

如果你想上採樣 最後一個軸而不是對其進行子採樣，只需設置classes_of_interest 下面的變量到你想要的長度。添加的元素將隨機初始化或可選地用零初始化。查看文檔sample_tensors() 了解詳情。
~~~python

# TODO：設置源權重文件中的類數。注意這個數字必須包括 背景類，所以對於 MS COCO 的 80 個類，這必須是 80 + 1 = 81。

n_classes_source = 81
# TODO：設置要為子採樣權重張量選擇的類的索引。
# 如果你想隨機抽取一定數量的類，你可以設置
# `classes_of_interest` 為整數而不是下面的列表。無論哪種方式，不要忘記
# 包含背景類。也就是說，如果你設置一個整數，並且你想要`n`個正類，
# 那麼你必須設置 `classes_of_interest = n + 1`。

classes_of_interest = [0, 3, 8, 1, 2, 10, 4, 6, 12]
# classes_of_interest = 9 
# 如果您想隨機對最後一個軸進行子採樣而不是提供索引列表，請取消註釋。
for name in classifier_names:
     # 從源 HDF5 權重文件中獲取該層的訓練權重。
    kernel = weights_source_file[name][name]['kernel:0'].value
    bias = weights_source_file[name][name]['bias:0'].value
    # 獲取內核的形狀。我們對子採樣感興趣
    # 最後一個維度，'o'


    height, width, in_channels, out_channels = kernel.shape
    
    # Compute the indices of the elements we want to sub-sample.
    # Keep in mind that each classification predictor layer predicts multiple
    # bounding boxes for every spatial location, so we want to sub-sample
    # the relevant classes for each of these boxes.
    # 計算我們想要子採樣的元素的索引。
    # 請記住，每個分類預測器層預測多個
    # 每個空間位置的邊界框，所以我們想要子樣本
    # 每個框的相關類。


    if isinstance(classes_of_interest, (list, tuple)):
        subsampling_indices = []
        for i in range(int(out_channels/n_classes_source)):
            indices = np.array(classes_of_interest) + i * n_classes_source
            subsampling_indices.append(indices)
        subsampling_indices = list(np.concatenate(subsampling_indices))
    elif isinstance(classes_of_interest, int):
        subsampling_indices = int(classes_of_interest * (out_channels/n_classes_source))
    else:
        raise ValueError("`classes_of_interest` must be either an integer or a list/tuple.")
    
     # 對kernel 和bias進行子採樣 Sub-sample 。
    # 下面使用的 `sample_tensors()` 函數提供了廣泛的
    # 文檔，所以如果你想知道，請不要猶豫閱讀它
    # 這裡到底發生了什麼。
    new_kernel, new_bias = sample_tensors(weights_list=[kernel, bias],
                                          sampling_instructions=[height, width, in_channels, subsampling_indices],
                                          axes=[[3]],# 一個bias維度對應於最後一個kernel維度。

                                          init=['gaussian', 'zeros'],
                                          mean=0.0,
                                          stddev=0.005)
    
    # 從目標文件中刪除舊的權重。
    del weights_destination_file[name][name]['kernel:0']
    del weights_destination_file[name][name]['bias:0']
    # 為Sub-sample權重創建新數據集。
    weights_destination_file[name][name].create_dataset(name='kernel:0', data=new_kernel)
    weights_destination_file[name][name].create_dataset(name='bias:0', data=new_bias)

# 確保在該 sub-routine退出之前將所有數據寫入我們的輸出文件
weights_destination_file.flush()
~~~
就是這樣，我們完成了。

讓我們快速檢查一下 'conv4_3_norm_mbox_conf' 目標權重文件中的圖層：
~~~python

conv4_3_norm_mbox_conf_kernel = weights_destination_file[classifier_names[0]][classifier_names[0]]['kernel:0']
conv4_3_norm_mbox_conf_bias = weights_destination_file[classifier_names[0]][classifier_names[0]]['bias:0']

print("Shape of the '{}' weights:".format(classifier_names[0]))
print()
print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)

  ~~~

“conv4_3_norm_mbox_conf”權重的形狀：

kernel:	 (3, 3, 512, 36)
bias:	 (36,)

  

好的！正是我們想要的，最後一個軸上有 36 個元素。現在，權重與我們預測 8 個正類的新 SSD300 模型兼容。

本教程的相關部分到此結束，但我們還可以做一件事並驗證子採樣權重是否確實有效。讓我們在下一節中這樣做。


## 5. 驗證我們的二次抽樣權重是否有效

在上面的示例中，我們將在 MS COCO 上訓練的 SSD300 模型的完全訓練權重從 80 個類別子採樣到我們需要的 8 個類別。

我們現在可以創建一個具有 8 個類的新 SSD300，將我們的子採樣權重加載到其中，並查看模型如何在包含這 8 個類中的一些對象的一些測試圖像上執行。我們開始做吧。

~~~python
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_patch_sampling_ops import RandomMaxCropFixedAR
from data_generator.object_detection_2d_geometric_ops import Resize
~~~

使用 TensorFlow 後端。

  

### 5.1.設置模型的參數。

與往常一樣，設置模型的參數。我們將為 SSD300 MS COCO 模型設置配置。

~~~python
img_height = 300 # 輸入圖像的高度
img_width = 300 # 輸入圖像的寬度
img_channels = 3 # 輸入圖像的顏色通道數

subtract_mean = [123, 117, 104] # 數據集中圖像的每通道均值(per-channel)

swap_channels = [2, 1, 0] # 原始SSD中的顏色通道順序是BGR，所以我們應該將其設置為`True`，但奇怪的是沒有交換結果更好

# TODO:  設置類數。
n_classes = 8 # 正類的數量(positive classes)，例如Pascal VOC 為 20，MS COCO 為 80

scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] 
# 原始 SSD300 中用於 MS COCO 數據集的錨框縮放因子。

# scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] 
# 原始 SSD300 中用於 Pascal VOC 數據集的錨框縮放因子。


aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # 原始SSD300中使用的anchor box寬高比；順序很重要

two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]# 每個預測層的兩個相鄰錨框中心點之間的空間。
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # 第一個錨框中心點距圖像上邊界和左邊界的偏移量，作為每個預測層步長的一部分。
clip_boxes = False # 是否要限制錨框完全位於圖像邊界內
variances = [0.1, 0.1, 0.2, 0.2]# 與原始實現中一樣縮放編碼目標坐標的方差
normalize_coords = True
~~~

  

### 5.2.構建模型

構建模型並將我們新創建的子採樣權重加載到其中。
~~~python
# 1: Build the Keras model

K.clear_session() # 從內存中清除以前的模型。
model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='inference',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=subtract_mean,
                divide_by_stddev=None,
                swap_channels=swap_channels,
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400,
                return_predictor_sizes=False)

print("Model built.")

# 2: Load the sub-sampled weights into the model.


# 加載我們剛剛通過子採樣創建的權重。

weights_path = weights_destination_path

model.load_weights(weights_path, by_name=True)

print("Weights file loaded:", weights_path)

# 3: 實例化 Adam 優化器和 SSD 損失函數並編譯模型。

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
~~~
  

### 5.3.加載一些圖像來測試我們的模型

我們從經過訓練的 SSD300 MS COCO 權重中對一些道路交通類別進行了子採樣，所以讓我們在一些道路交通圖像上試用我們的模型。鏈接到的 Udacity 道路交通數據集ssd7_training.ipynb notebook 適合這項任務。讓我們實例化一個數據生成器 並加載 Udacity 數據集。此處的所有內容均已預設，但如果您想了解有關數據生成器及其功能的更多信息，請查看詳細教程[這](https://github.com/pierluigiferrari/data_generator_object_detection_2d) 存儲庫。

~~~python
dataset = DataGenerator()

# TODO: Set the paths to your dataset here.
images_path = '../../datasets/Udacity_Driving/driving_dataset_consolidated_small/'
labels_path = '../../datasets/Udacity_Driving/driving_dataset_consolidated_small/labels.csv'

dataset.parse_csv(images_dir=images_path,
                  labels_filename=labels_path,
                  input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], 
# 這是包含數據集標籤的 CSV 文件中前六列的順序。如果您的標籤是 XML 格式，也許 XML 解析器會有所幫助，請查看文檔。

                  include_classes='all',
                  random_sample=False)

print("Number of images in the dataset:", dataset.get_dataset_size())
~~~

  

數據集中的圖像數量：22241

  

確保批處理生成器生成大小合適的圖像(300, 300).我們將首先隨機裁剪縱橫比為 1.0 的最大可能補丁，然後調整大小(300, 300).

~~~python
convert_to_3_channels = ConvertTo3Channels()
random_max_crop = RandomMaxCropFixedAR(patch_aspect_ratio=img_width/img_height)
resize = Resize(height=img_height, width=img_width)

generator = dataset.generate(batch_size=1,
                             shuffle=True,
                             transformations=[convert_to_3_channels,
                                              random_max_crop,
                                              resize],
                             returns={'processed_images',
                                      'processed_labels',
                                      'filenames'},
                             keep_images_without_gt=False)
~~~

~~~python
# 生成樣本

batch_images, batch_labels, batch_filenames = next(generator)

i = 0 # Which batch item to look at

print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(batch_labels[i])
~~~

圖片：../../datasets/Udacity_Driving/driving_dataset_consolidated_small/1479505696943867867.jpg

  

真值框：
[[ 1 0 148 37 173]

 [ 1 40 139 86 172]

 [1 79 143 95 158]

 [ 1 128 143 144 154]

 [ 1 149 111 256 210]]

  

### 5.4.做出預測並將其可視化

~~~python
# Make a prediction

y_pred = model.predict(batch_images)
~~~

~~~python
# 解碼原始預測。

i = 0

confidence_threshold = 0.5

y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('    class    conf  xmin    ymin    xmax    ymax')
print(y_pred_thresh[0])
~~~

  

預測框：
  
 class conf xmin  ymin   xmax  ymax
  [[ 1. 0.95 40.68 137.04 87.31 167.75]
	[ 1. 0.81 0.43 148.85 35.93 172.36]
	[ 2. 0.8 148.55 113.82 259.65 209.92]
	[ 5. 0.31 75.24 24.65 85.85 52.44]]

~~~python
# Visualize the predictions.

from matplotlib import pyplot as plt

%matplotlib inline

plt.figure(figsize=(20,12))
plt.imshow(batch_images[i])

current_axis = plt.gca()

classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist',
           'traffic_light', 'motorcycle', 'bus', 'stop_sign'] # 這樣我們就可以將類名而不是 ID 打印到圖像上


# 用藍色繪製預測框
for box in y_pred_thresh[i]:
    class_id = box[0]
    confidence = box[1]
    xmin = box[2]
    ymin = box[3]
    xmax = box[4]
    ymax = box[5]
    label = '{}: {:.2f}'.format(classes[int(class_id)], confidence)
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='blue', fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})


# 以綠色繪製地面實況框（為了更清晰省略標籤）
for box in batch_labels[i]:
    class_id = box[0]
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = '{}'.format(classes[int(class_id)])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
    #current_axis.text(box[1], box[3], label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
~~~


## 資源4
遷移學習 6 個步驟
[來源](https://www.v7labs.com/blog/transfer-learning-guide#h5)

##### 1.獲取預訓練模型
##### 2.創建基礎模型
##### 3.凍結圖層
##### 4.添加新的可訓練層
##### 5.訓練新層
##### 6.微調你的模型
