# Multi Plant Image Formation
Create fake outdoor multi-plant images using a generative adversarial network (GAN) that was trained to translate indoor single-plant images to appear as real 
outdoor images. The goal of this project is to provide a dataset for a secondary network to learn to locate plants in a given image. Multi-plant images are 
generated with corresponding json files that contain bounding box data for all plants.

<p align="center">
  <img src="https://github.com/alexk-1998/multi-plant-image-formation/blob/master/examples/comp.png" title="Fake multi-plant image before GAN translation" width="49%"/>
  <img src="https://github.com/alexk-1998/multi-plant-image-formation/blob/master/examples/trans.png" title="Fake multi-plant image after GAN translation" width="49%"/>
</p>

<p align="center">
  <img src="https://github.com/alexk-1998/multi-plant-image-formation/blob/master/examples/comp_bbox.png" title="Fake multi-plant bounding box image before GAN translation" width="49%"/>
  <img src="https://github.com/alexk-1998/multi-plant-image-formation/blob/master/examples/trans_bbox.png" title="Fake multi-plant bounding box image after GAN translation" width="49%"/>
</p>

The GAN was trained using the "Contrastive Unpaired Translation" project located at https://github.com/taesungp/contrastive-unpaired-translation.

# Usage

Clone the repository and run

```python3 main.py --root /directory/for/saving```

Typical usage, including additional option flags, is

```python3 main.py --root /directory/for/saving --min_scale 0.1 --max_scale 0.25 --min_plants 3 --max_plants 6 --plant_pad 100 --gpu_ids -1 --num_images 100```

# Requirements

Python >= 3.6.0

pytorch >= 1.4.0 (== 1.4.0 if running on GPU with the provided dockerfile)

OpenCV for Python

# Options

Several options are available when creating the multi-plant images. These include specifying the number and size of plants in a given image, the total number of 
images created, and more. The full list of available options can be found in ```options.py```. 

With the exception of the ```--gpu_ids``` flag, the model parameters should not be changed. Additionally, the flags ```--l_avg```, ```--a_avg```, and ```--b_avg``` 
should not be changed as these are used for colour correction of the plants prior to being passed through the GAN. The default values for these flags are 
consistent with the values used when originally training the GAN. Adjusting the colour correction method will likely produce poorly translated images.

The flag ```--no_save_all``` can be used to prevent the saving of additional images produced when generating the dataset. If the flag is enabled, only the 
translated multi-plant images and their respective json files will be saved.

In general, the plants in an image are translated by removing one plant from the multi-plant image, passing it through the network, and placing the 
plant back onto the multi-plant image. The single-plant images are not cropped at the location of the bounding box, but rather at the bounding box plus some 
tolerance specified by ```--plant_padding```. If the flag ```--replace_all``` is not enabled, only the bounding box contents are placed back onto the multi-plant 
image. However, if this flag is used, the entire translated image is placed onto the multi-plant image. In the figures below, ```--replace_all``` leads to 
additional blurriness from image translation exceeding the bounding box. This may be useful when training a network to identify plants in the multi-plant images.

<p align="center">
  <img src="https://github.com/alexk-1998/multi-plant-image-formation/blob/master/examples/trans_bbox.png" title="Translated multi-plant image without the replace_all flag" width="49%"/>
  <img src="https://github.com/alexk-1998/multi-plant-image-formation/blob/master/examples/replace_all.png" title="Translated multi-plant image with the replace_all flag" width="49%"/>
</p>
