# video_loopback_for_webui

一个stable diffusion的脚本，用来做视频的img2img，通过loopback和时序模糊的方法提高视频的稳定性，尽量降低img2img动画特有的闪烁感。
设计的目的是用于人物动画，一般来说需要搭配 DreamBooth 一起用，且对模型质量要求比较高。
最终效果对img2img的参数设置非常非常敏感，需要仔细调整。这玩意挺玄学的，如果没调好还不如直接用 Batch img2img 。
希望开源可以促进更多的人测试不同的参数配置，从而找到最优策略。

在这个脚本的帮助下，允许使用更高的denoising（比如0.4甚至更高），这使得我们可以在驱动视频较模糊的情况下也能获得较好的效果。
这个技术很适合用于优化由CoNR生成的视频，CoNR传送门：https://github.com/transpchan/Live3D-v2

除了脚本界面中的参数，原有webui界面中的所有参数都会影响生成，别忘了调这部分。

这里应该还要再写一个更详细的文档介绍用法和各种参数来着，还应该附带一些简单的调参技巧，等我有空了弄，应该不会耽误太久。
这个脚本本身还需要缺乏一些详细的测试，所以可能有Bug。
随手加了个视频inpainting的功能，还没有经过充分测试，这个功能理论上可以些微降低人物边缘的模糊情况。勾选use_mask，并且处于inpainting界面，并设定好inpainting相关的参数，就能用了。
目前已知的Bug是，在img2img界面或inpainting界面中，需要先保证webui界面里有一张图片，才能开始处理（随便拖一张图片上去就行了）

## 效果展示(还不太成熟)
转换效果：

https://user-images.githubusercontent.com/122792358/212680768-56222b45-dd7b-428f-a29b-cda21d757703.mp4

https://user-images.githubusercontent.com/122792358/212680867-14f8f10f-f439-474d-b15f-af17536c8c94.mp4

原视频：（感谢CoNR的工作!!!）

https://user-images.githubusercontent.com/122792358/212680628-9b899d25-121a-4428-a03b-bd2ae9c595a7.mp4

Dreambooth的训练样本：

![twintails (1)](https://user-images.githubusercontent.com/122792358/212681343-c0665891-6467-4bf2-a9d7-3deb1f72d1a9.png)![twintails (2)](https://user-images.githubusercontent.com/122792358/212681349-adf69c2c-0523-438c-ac13-c9ed1f09dffd.png)![twintails (3)](https://user-images.githubusercontent.com/122792358/212681351-12a437f4-d3b6-438a-a619-555aed1a82f3.png)![twintails (4)](https://user-images.githubusercontent.com/122792358/212681355-ef454e45-b349-4080-8245-9aac3b8f8126.png)


整个流程的关键点在模型训练，对DreamBooth模型的要求很高，需要先训一个高质量高稳定性的模型出来才行，这个脚本只是作为辅助作用。
