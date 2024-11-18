# ILMF-FND
Code for paper "Contrastive Learning of Cross-modal Information Enhancement for Multimodal Fake News Detection".

# Dependency
+ python 3.5+
+ pytorch 1.0+
+ transformers 4.28.0

## Dataset
We conduct experiments on two benchmark datasets Twitter and Weibo. In experiments, we keep the same data split scheme as the benchmark. Specifically, for the Twitter dataset, we followed the work of ([Chen et al., 2022](https://github.com/cyxanna/CAFE)), and for the Weibo dataset, we followed the work of ([Wang et al., 2022](https://github.com/yaqingwang/EANN-KDD18)).


## Training
To train the ILMF-FND:
```shell script
python weibo.py 
python twitter.py 
```
 
