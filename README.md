# Event-Based-Learning

This work proses an approach for learning semantic segmentation from only event-based information (event-based cameras).

# Requirements
* Python 2.7+
* Tensorflow 1.11
* Opencv
* Keras
* Imgaug
* Sklearn


## Citing Multi-Level Superpixels 

If you find EV-SegNet useful in your research, please consider citing:
``` 
@inproceedings{alonso2019EvSegNet,
  title={EV-SegNet: Semantic Segmentation for Event-based Cameras},
  author={Alonso, I{\~n}igo and Murillo, Ana C},
  booktitle={IEEE International Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2019}
}
```


## Dataset
Our dataset is a subset of the [DDD17: DAVIS Driving Dataset](http://sensors.ini.uzh.ch/news_page/DDD17.html). This original dataset do not provide any semantic segmentation label, we provide them as well as some modification of the event images. [See more here](https://github.com/Shathe/Event-Based-Learning/tree/master/Dataset).


[Download it here](https://drive.google.com/open?id=1Ug6iZc7WYQWCklxwcemCeyw3CPyuuxJf)


## Replicate results
```
python train_eager.py --epochs 0
```
## Train from scratch


```
python train_eager.py --epochs 500 --dataset path_to_dataset  --model_path path_to_pretrained_model  --batch_size 8
```
Wehre [path_to_dataset] is the path to the downloaded dataset (uncompressed) and [path_to_pretrained_model] is the downloaded pretrained model (uncompressed)

 
