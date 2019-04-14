# comic textboxs 漫画对话框

初衷：利用目标检测识别出漫画中的对话框，可提升在手机屏幕的阅读体验

运行环境 python3 keras2.1.5

修改自
https://github.com/pierluigiferrari/ssd_keras

数据来源
```
@inproceedings{IyyerComics2016,
Author = {Mohit Iyyer and Varun Manjunatha and Anupam Guha and Yogarshi Vyas and Jordan Boyd-Graber 
and Hal {Daum\'{e} III} and Larry Davis},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
Year = "2017",
Title = {The Amazing Mysteries of the Gutter: Drawing Inferences Between Panels in Comic Book Narratives}
```

运行convert_data.py,make_csv_data.py，make_comic_h5_data.py制作训练模型所需的数据
然后
运行ssd7_training_comic.ipynb即可

MAP:0.85


Original intention: Use the target detection to identify the dialog box in the comics, which can improve the reading experience on the screen of the mobile phone.

Operating environment python3 keras2.1.5

Modified from
Https://github.com/pierluigiferrari/ssd_keras

Data Sources
```
@inproceedings{IyyerComics2016,
Author = {Mohit Iyyer and Varun Manjunatha and Anupam Guha and Yogarshi Vyas and Jordan Boyd-Graber
And Hal {Daum\'{e} III} and Larry Davis},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
Year = "2017",
Title = {The Amazing Mysteries of the Gutter: Drawing Inferences Between Panels in Comic Book Narratives}
```

Run convert_data.py, make_csv_data.py, make_comic_h5_data.py to create the data needed for the training model
then
Run ssd7_training_comic.ipynb

MAP: 0.85

效果如下：
![avatar](https://github.com/freekoy/comic-textboxs/blob/master/infer.jpg)
