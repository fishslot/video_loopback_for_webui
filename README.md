# Video Loopback for WebUI

[中文简介](https://github.com/fishslot/video_loopback_for_webui/blob/main/README_zh.md)
[用法介绍](https://github.com/fishslot/video_loopback_for_webui/wiki)

## Demonstration (still can be improved)

https://user-images.githubusercontent.com/122792358/218375476-a4116c74-5a9a-41e2-970a-c3cc09f796ae.mp4

The original video is from Live3D (thanks for Live3D's work!!!).
## News
I made some change on the [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) by [Mikubill](https://github.com/Mikubill), now ,try install this version https://github.com/fishslot/sd-webui-controlnet, and they can work toghter.

## Introduction

This is a Stable Diffusion WebUI extension for video img2img, which improves the stability of the video through loopback and temporal blurring methods, 
trying to reduce the flicker that is typical of img2img animations. 
This extension can be seen as a video version of [loopback and superimpose](https://github.com/DiceOwl/StableDiffusionStuff/blob/main/loopback_superimpose.py), 
with some enhanced features. It's a quite simple method, but sometimes useful.

The design is intended for character animations and usually needs to be used with Lora or DreamBooth. 
The model quality requirements are relatively high, and the stability of the model itself will greatly affect the difficulty of subsequent parameter adjustment.

The final video effect is very very sensitive to the setting of parameters, which requires careful adjustment. 
This is a very mysterious thing, if it is not adjusted well, it is better to use Batch img2img directly :)

It is hoped that open source will promote more people to test different parameter configurations, so as to find the best strategy.

With the help of this script, it allows for higher denoising (such as 0.4 or even higher), which enables us to obtain good results even in situations where the driving video is relatively blurry. However, generally speaking, denoising between 0.3~0.4 can solve many requirements.

This technology is well suited for optimizing videos generated by Live3D. Live3D entrance: https://github.com/transpchan/Live3D-v2

In addition to the parameters in the script interface, all parameters in the original webui interface will affect the generation, so don't forget to adjust this part.
<br>
This extension itself lacks some detailed testing, so there may be bugs.
The known bug at the moment is that in the img2img interface or inpainting interface, you need to make sure that there is an image in the webui interface before you start processing (just drag and drop an image).



## Installation method

Install it like any normal WebUI extension installation process.

Also, you need to install ffmpeg in advance.

## Usage

In the img2img tab or inpainting tab, find the script named "Video Loopback".

You need to make sure that there is an image in the webui interface before you start processing (just drag and drop an image).

# Features
## Loopback & superimpose & batch blend
The input_directory can be filled with the path of a video file or a folder containing images.

Each time the SD generates an image, 
the generated image is blended (superimposed) with the original image to form a new original image for the next generation. 
The intensity of the blend is specified by the `superimpose_alpha` parameter, 
which is 0 when the original image is preserved. 
`loop_n` can control how many times the entire blending process will be repeated.

Note that if `batch_size` and `batch_count` are not 1, 
the images generated in this batch will be blended into one image first, then blended with the original image. 
This strategy is referred to as batch blend later.

Batch blend can effectively improve stability and significantly reduce dependence on `fix_seed`. 
However, this strategy sometimes produces blurry images, 
which can be eliminated by reasonably configuring `denoising_schedule`, `batch_count_schedule`, and `image_post_processing_schedule`.

By the way, let's talk about `fix_seed`, which can effectively improve stability, 
but it may cause texture sticking and is more likely to produce error frames. 
And when using batch blend, `fix_seed` may actually reduce stability. 
However, `fix_seed` is still worth trying. If you want to try `fix_seed`, 
it is recommended to select `Extra` and use `subseed` to add a certain degree of randomness, 
which can reduce the side effects of `fix_seed` to a certain extent. Based on my personal experience, 
`subseed_strength` can be set a little more aggressive, for example 0.9 

## Inpainting
Switch to the inpainting page and check `use_mask`, and you can use the inpainting function. 
`Mask blur`, `Mask mode`, `Mask content`, `Inpaint area`, `Only masked padding` and so on will work as usual.

The `mask_directory` can be filled with the path of a single image or a folder. 
If it is a folder, each mask in the folder must have the same file name as the input image.
If it is empty, the alpha channel of image will be use as the mask.

## Temporal blend
In simple terms, each frame is first blended with its time-adjacent frames and then sent to the SD. 
This function can theoretically slightly enhance stability, but may produce artifacts and is difficult to find the optimal setting. 
If necessary, it can be skipped. 
By inputting a comma-separated set of decimal numbers in the `temporal_superimpose_alpha_list`, this function can be enabled. 
Each number represents a weight, and the number of weights must be odd. 
The weight in the middle represents the weight of the current frame, 
the first weight on the left represents the weight of the previous frame, 
and the first weight on the right represents the weight of the next frame, and so on. 
The weights do not need to be added up to 1, and normalization will be automatically performed internally. 
If the `temporal_superimpose_alpha_list` is empty or only has a single number, it means that this function is not enabled.

In fact, i found that `video_post_process_method` is more useful than temporal blend.

## Schedule

Using Python syntax, input the expression, and the available variables are: image_i representing the number of the current processed image, 
loop_i representing the current iteration. 

You can use contents in the math library directly.

## Deblur
In `image_post_processing_schedule`, you can use the PIL.ImageFilter module to effectively reduce the blur of the image, for example, you can set it as:

```
lambda img: img.filter(ImageFilter.EDGE_ENHANCE).filter(ImageFilter.SMOOTH) if loop_i in {5,8} else img
```

Where {5,8} specifically depends on the model being used.

## Improving Video Stability

Use `video_post_process_method` to further improve the stability of the video. Currently, only FastDVDNet is supported.

Thanks to the work of [FastDVDNet](https://github.com/m-tassano/fastdvdnet).

Why use FastDVDNet: There is no special reason, it is a random choice, maybe because their name has "fast" :) 

I will try using other models in the future. If you have any recommended models, please let me know.

## Training samples used in the demonstration

![twintails (1)](https://user-images.githubusercontent.com/122792358/212681343-c0665891-6467-4bf2-a9d7-3deb1f72d1a9.png)![twintails (2)](https://user-images.githubusercontent.com/122792358/212681349-adf69c2c-0523-438c-ac13-c9ed1f09dffd.png)![twintails (3)](https://user-images.githubusercontent.com/122792358/212681351-12a437f4-d3b6-438a-a619-555aed1a82f3.png)![twintails (4)](https://user-images.githubusercontent.com/122792358/212681355-ef454e45-b349-4080-8245-9aac3b8f8126.png)


The key point of the entire process is model training, which has high requirements for the model. A high-quality and stable model must be trained first in order for this to work, and the script serves as a helper.

![visitor](https://count.getloli.com/get/@video_loopback)
