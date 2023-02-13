# Video Loopback for WebUI

一个 Stable Diffusion WebUI 的插件，用来做视频的 img2img，通过 loopback 和时序模糊的方法提高视频的稳定性，尽量降低 img2img 动画特有的闪烁感。
这个脚本可以看作是 [loopback and superimpose](https://github.com/DiceOwl/StableDiffusionStuff/blob/main/loopback_superimpose.py) 的视频版，并做了一些增强的功能。
设计的目的是用于人物动画，一般来说需要搭配 Lora 或是 DreamBooth 一起用，且对模型质量要求比较高，模型本身的稳定性会很大程度影响后续的调参难度。
最终视频效果对参数设置非常非常敏感，需要仔细调整。这玩意挺玄学的，如果没调好还不如直接用 Batch img2img 。
希望开源可以促进更多的人测试不同的参数配置，从而找到最优策略。

在这个脚本的帮助下，允许使用更高的 denoising（比如 0.4 甚至更高），这使得我们可以在驱动视频较模糊的情况下也能获得较好的效果。不过一般来说 denoising 在 0.3~0.4 之间就能解决很多需求了。
这个技术很适合用于优化由 Live3D 生成的视频，Live3D 传送门：https://github.com/transpchan/Live3D-v2

除了脚本界面中的参数，原有webui界面中的所有参数都会影响生成，别忘了调这部分。

[用法介绍](https://github.com/fishslot/video_loopback_for_webui/wiki)

这个脚本本身还缺乏一些详细的测试，所以可能有Bug。
目前已知的Bug是，在img2img界面或inpainting界面中，需要先保证webui界面里有一张图片，才能开始处理（随便拖一张图片上去就行了）

## 效果展示(还不太成熟)
转换效果：


https://user-images.githubusercontent.com/122792358/218375476-a4116c74-5a9a-41e2-970a-c3cc09f796ae.mp4


原视频来自Live3D（感谢 Live3D 的工作!!!）

##安装方法：

按照正常的安装 WebUI extention 的流程安装即可。

还有一点就是，需要提前安装 ffmpeg 。

## 功能

# loopback & superimpose for video

`input_directory` 可以填一个视频文件的路径，也可以填一个包含图片的文件夹的路径。

每次SD生成一张图片后，将生成的图片与原图进行混合（叠加）后，作为新的原图进行下一次生成。混合的强度由 `superimpose_alpha` 参数指定，为0时，将保留原图。 `loop_n` 可以控制整个混合过程将重复多少次。

注意，如果 batch_size 和 batch_count 不为 1 ，会先将本批次生成的所有图片混合为一张图片，再与原图进行混合，后面把这种策略称为 batch blend 。

batch blend 可以有效提升稳定性，显著降低对 `fix_seed` 的依赖。不过这种策略有时会产生模糊的图像，如果合理配置 `denoising_schedule` `batch_count_schedule` 以及 `image_post_processing_schedule` 可以消除这种模糊（后续将会提到）。

这里顺便说一下 `fix_seed` ，这个可以有效提升稳定性，但是会产生纹理粘连，而且更容易产生错误帧，而且当使用 batch blend 时， `fix_seed` 可能反而会降低稳定性。但是 `fix_seed` 仍然是值得一试的。如果要尝试 `fix_seed` ，建议勾选 Extra ，使用 subseed 加入一定的随机性，一定程度上降低 `fix_seed` 的副作用。就个人经验来说，`subseed_strength` 可以设置得激进一点，比如 0.9 。

# inpainting

切换到 inpainting 页面，并勾选 `use_mask` ，就可以使用 inpainting 功能。Mask blur, Mask mode, Mask content, Inpaint area, Only masked padding 这些东西都能正常工作（大概）。

mask_directory 里面可以填单张图片的路径，也可以填一个文件夹的路径。如果填的是文件夹，文件夹下的每个mask都必须与输入图片具有相同的文件名。

# 时域混合

简单来说就是每一帧会先与时间上相邻的帧进行混合，然后再送入SD。该功能理论上可以稍微提升一点稳定性，但是可能会产生伪影，比较棘手，需要仔细寻找最优设置，实在不行干脆不用也行。在 `temporal_superimpose_alpha_list` 里输入一组逗号分隔的小数就可以启用这个功能，每个数代表一个权重，权重的数量必须是奇数，位于中间的权重表示当前帧的权重，左边紧邻的第一个权重表示前一帧的权重，右边相邻的第一个权重表示后一帧的权重，以此类推。权重不必相加等于 1 ，内部会自动归一化。如果 `temporal_superimpose_alpha_list` 为空，或是只有一个单独的数，表示不启用该功能。

# schedule

使用 python 语法，输入表达式即可，可以使用的变量有： image_i 表示当前处理的是第几张图片、loop_i 表示当前是第几次迭代。可以直接使用 math 库中的内容。

# 抗模糊
在 `image_post_processing_schedule` 可以使用 PIL.ImageFilter 模块，可以有效降低画面的模糊感，比如可以设置为：

```
lambda img:  img.filter(ImageFilter.EDGE_ENHANCE).filter(ImageFilter.SMOOTH) if loop_i in {5,8} else img
```

其中 `{5,8}` 具体填什么数字需要根据使用的模型做出改变。

# 提高视频稳定性

使用 `video_post_process_method` 来进一步提升视频的稳定性。目前只支持 FastDVDNet 。

感谢 FastDVDNet 的工作。

为什么使用 FastDVDNet：其实没什么特别的原因，可能是因为他们名字里有fast吧 :) 后续应该还会尝试用其他模型。

演示中使用的的训练样本：

![twintails (1)](https://user-images.githubusercontent.com/122792358/212681343-c0665891-6467-4bf2-a9d7-3deb1f72d1a9.png)![twintails (2)](https://user-images.githubusercontent.com/122792358/212681349-adf69c2c-0523-438c-ac13-c9ed1f09dffd.png)![twintails (3)](https://user-images.githubusercontent.com/122792358/212681351-12a437f4-d3b6-438a-a619-555aed1a82f3.png)![twintails (4)](https://user-images.githubusercontent.com/122792358/212681355-ef454e45-b349-4080-8245-9aac3b8f8126.png)


整个流程的关键点在模型训练，对模型的要求很高，需要先训一个高质量高稳定性的模型出来才行，这个脚本只是作为辅助作用。
