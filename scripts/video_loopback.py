import gradio as gr
import modules.scripts as scripts
from modules import processing, shared
from modules.processing import Processed

from PIL import Image
from collections import deque
from pathlib import Path
import os, datetime, math, json


def get_now_time():
    SHA_TZ = datetime.timezone(
        datetime.timedelta(hours=8),
        name='Asia/Shanghai',
    )
    fmt = '%y%m%d_%H%M%S'
    utc_now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)
    stamp = beijing_now.strftime(fmt)
    return stamp


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def resize_img(img, target_size):
    return img.resize(target_size, Image.ANTIALIAS)


def make_video(
        input_dir, output_filename,
        frame_rate=12, input_format='%07d.png'):
    os.system(
        f"ffmpeg -r {frame_rate} "
        f"-i {Path(input_dir) / input_format} "
        f"-c:v libx264 -preset ultrafast -qp 0 "
        f"{output_filename} "
    )


class TemporalImageBlender:
    def __init__(
            self, image_path_list=None, window_size=1,
            target_size=(512, 512),
            use_mask=False,
            mask_dir='', mask_threshold=128):
        self.image_path_list = image_path_list
        self.target_size = target_size

        assert window_size % 2 == 1
        self.window_size = window_size
        self.window = deque(
            self.read_image_resize(image_path_list[i])
            for i in range(window_size // 2 + 1)
        )
        self.current_i = 0
        self.current_pos = 0

        self.use_mask = use_mask
        self.mask_dir = Path(mask_dir)
        self.mask_threshold = mask_threshold

    def read_image_resize(self, path) -> Image:
        return resize_img(Image.open(path), self.target_size)

    def move_to_next(self):
        hws = self.window_size // 2  # half window size
        self.current_i += 1
        if self.current_i >= len(self.image_path_list):
            self.current_i = len(self.image_path_list) - 1
            return
        if self.current_i + hws < len(self.image_path_list):
            self.window.append(self.read_image_resize(
                self.image_path_list[self.current_i + hws]))
        if self.current_i - hws > 0:
            self.window.popleft()
        else:
            self.current_pos += 1
        assert self.current_pos < len(self.window) <= self.window_size

    def current_image(self) -> Image:
        return self.window[self.current_pos]

    def current_mask(self):
        if not self.use_mask:
            return None
        mask = None
        if self.mask_dir:
            if self.mask_dir.is_dir():
                mask_name = self.image_path_list[self.current_i].name
                mask_path = self.mask_dir/mask_name
                if mask_path.is_file():
                    mask = Image.open(mask_path).convert('L')
                else:
                    print(f'Warning: "{mask_path}" has no mask')
            elif self.mask_dir.is_file():
                mask = Image.open(self.mask_dir).convert('L')
            else:
                raise FileNotFoundError("mask not found")
        else:
            if 'RGBA' == self.current_image().mode:
                mask = self.current_image().split()[-1]
            else:
                print(f'Warning: "{self.image_path_list[self.current_i]}" has no mask')
        # apply threshold
        if mask is not None:
            mask = mask.convert('L').point(
                lambda x: 255 if x > self.mask_threshold else 0)
        # resize mask
        if mask.size != self.target_size:
            mask = resize_img(mask, self.target_size)
        return mask

    def blend_temporal(self, alpha_list):
        assert len(alpha_list) == self.window_size, 'the length of temporal_superimpose_alpha_list must be fixed'
        hws = self.window_size // 2  # half window size
        now_fac_sum = 0.0
        output_img = self.window[0]
        for factor, img in zip(
                alpha_list[-(self.current_i + hws + 1):],
                self.window):
            now_fac_sum += factor
            img = img.convert(output_img.mode)
            output_img = Image.blend(output_img, img, factor / now_fac_sum)

        mask = self.current_mask()
        if mask:
            output_img = Image.composite(output_img, self.current_image(), mask)
            # 当mask像素取255时为img1,取0时为img2

        return output_img

    def blend_batch(self, new_imgs, superimpose_alpha):
        if not new_imgs:
            return self.current_image()

        new_img = new_imgs[0]
        for i, img in enumerate(new_imgs[1:], 1):
            new_img = Image.blend(new_img, img, 1.0 / (i + 1))

        base_img = self.current_image().convert(new_img.mode)

        output_img = Image.blend(base_img, new_img, superimpose_alpha)

        mask = self.current_mask()
        if mask:
            output_img = Image.composite(output_img, base_img, mask)

        return output_img


class Script(scripts.Script):
    def title(self):
        return "Video Loopback"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        input_dir = gr.Textbox(label='input_directory')
        output_dir = gr.Textbox(label='output_directory')
        output_frame_rate = gr.Number(label='output_frame_rate', precision=0, value=30)
        max_frames = gr.Number(label='max_frames', precision=0, value=9999)
        extract_nth_frame = gr.Number(label='extract_nth_frame', precision=0, value=1)
        loop_n = gr.Number(label='loop_n', precision=0, value=5)
        superimpose_alpha = gr.Number(label='superimpose_alpha', precision=None, value=0.3)
        fix_seed = gr.Checkbox(label='fix_seed', value=True)
        fix_subseed = gr.Checkbox(label='fix_subseed', value=False)
        temporal_superimpose_alpha_list = gr.Textbox(
            label='temporal_superimpose_alpha_list',
            value='0.03,0.95,0.02'
        )
        save_every_loop = gr.Checkbox(label='save_every_loop', value=False)

        # mask settings
        use_mask = gr.Checkbox(label='use_mask', value=False)
        with gr.Box(visible=False) as mask_settings_box:
            mask_dir = gr.Textbox(
                label='mask_directory',
                placeholder='keep this empty to use the alpha channel of image as mask'
            )
            # use_alpha_as_mask = gr.Checkbox(label='use_alpha_as_mask', value=False)
            mask_threshold = gr.Slider(minimum=0, maximum=255, step=1, label='mask_threshold', value=128)

        use_mask.change(
            fn=lambda x: gr_show(x),
            show_progress=False,
            inputs=[use_mask], outputs=[mask_settings_box]
        )

        # extra settings
        with gr.Accordion('Advanced Settings of Video Loopback', open=False):
            gr.Markdown(
                "You can use any python expression in schedule <br>"
                "Available parameters: math.*, image_i, loop_i <br>"
                "Variable image_i and loop_i counts from zero <br>"
                "If seed_schedule/subseed_schedule is not empty, fix_seed/fix_subseed is ignored <br>"
                "These examples are just to demonstrate usage and are not recommended parameters."
            )
            subseed_strength_schedule = gr.Textbox(
                label='subseed_strength_schedule',
                placeholder='Example: (sin(pi*image_i/90)+1)/2*0.07'
            )
            denoising_schedule = gr.Textbox(
                label='denoising_schedule',
                placeholder='Example: 0.4 if loop_i<3 else 0.3'
            )
            seed_schedule = gr.Textbox(
                label='seed_schedule',
                placeholder='Example: [111,222,333][image_i//5%3]'
            )
            subseed_schedule = gr.Textbox(
                label='subseed_schedule',
                placeholder='Example: 112233+image_i'
            )
            cfg_schedule = gr.Textbox(
                label='cfg_schedule',
                placeholder='Example: 7 if image_i in [0,1,5,7] else 11'
            )
            superimpose_alpha_schedule = gr.Textbox(
                label='superimpose_alpha_schedule',
                placeholder='Example: 0.3 if loop_i<3 else 0.2 if loop_i<5 else 0.1'
            )
            temporal_superimpose_schedule = gr.Textbox(
                label='temporal_superimpose_schedule',
                placeholder='Example: [0.03, 0.95, 0.02] if loop_i<3 else [0.0, 1.0, 0.0]'
            )
            prompt_schedule = gr.Textbox(
                label='prompt_schedule',
                placeholder="Example: ['1girl,smile','1girl,closed mouth'][image_i//10%2]"
            )
            negative_prompt_schedule = gr.Textbox(
                label='negative_prompt_schedule',
                placeholder="Example: f' low quality, (blurry):{loop_i/10+1}'"
            )
            batch_count_schedule = gr.Textbox(
                label='batch_count_schedule',
                placeholder="Example: 2 if loop_i<3 else 1"
            )

        return [
            input_dir,
            output_dir,
            output_frame_rate,
            max_frames,
            extract_nth_frame,
            loop_n,
            superimpose_alpha,
            fix_seed,
            fix_subseed,
            temporal_superimpose_alpha_list,
            save_every_loop,
            use_mask,
            mask_dir,
            mask_threshold,
            subseed_strength_schedule,
            denoising_schedule,
            seed_schedule,
            subseed_schedule,
            cfg_schedule,
            superimpose_alpha_schedule,
            temporal_superimpose_schedule,
            prompt_schedule,
            negative_prompt_schedule,
            batch_count_schedule
        ]

    def run(self, p,
            input_dir,
            output_dir,
            output_frame_rate,
            max_frames,
            extract_nth_frame,
            loop_n,
            superimpose_alpha,
            fix_seed,
            fix_subseed,
            temporal_superimpose_alpha_list,
            save_every_loop,
            use_mask,
            mask_dir,
            mask_threshold,
            subseed_strength_schedule,
            denoising_schedule,
            seed_schedule,
            subseed_schedule,
            cfg_schedule,
            superimpose_alpha_schedule,
            temporal_superimpose_schedule,
            prompt_schedule,
            negative_prompt_schedule,
            batch_count_schedule):
        processing.fix_seed(p)
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        processed = None

        timestamp = get_now_time()

        # save settings
        args_dict = {
            "timestamp": timestamp,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "output_frame_rate": output_frame_rate,
            "max_frames": max_frames,
            "extract_nth_frame": extract_nth_frame,
            "loop_n": loop_n,
            "superimpose_alpha": superimpose_alpha,
            "fix_seed": fix_seed,
            "fix_subseed": fix_subseed,
            "temporal_superimpose_alpha_list": temporal_superimpose_alpha_list,
            "save_every_loop": save_every_loop,
            "subseed_strength_schedule": subseed_strength_schedule,
            "denoising_schedule": denoising_schedule,
            "seed_schedule": seed_schedule,
            "subseed_schedule": subseed_schedule,
            "cfg_schedule": cfg_schedule,
            "superimpose_alpha_schedule": superimpose_alpha_schedule,
            "temporal_superimpose_schedule": temporal_superimpose_schedule,
            "prompt_schedule": prompt_schedule,
            "negative_prompt_schedule": negative_prompt_schedule,
            "batch_count_schedule": batch_count_schedule,
            # "p": p.__dict__
            "seed": p.seed,
            "subseed": p.subseed,
            "subseed_strength": p.subseed_strength,
            "cfg_scale": p.cfg_scale,
            "prompt": p.prompt,
            "negative_prompt": p.negative_prompt,
            "sampler_name": p.sampler_name,
            "width": p.width,
            "height": p.height,
            "denoising_strength": p.denoising_strength,
            "batch_size": p.batch_size,
            "n_iter": p.n_iter,
            "steps": p.steps,
            "model_name": shared.sd_model.sd_checkpoint_info.model_name,
            "model_hash": shared.sd_model.sd_model_hash
        }

        output_dir = Path(output_dir) / timestamp
        output_frames_dir = output_dir / "output_frames"
        if save_every_loop:
            output_frames_dir = output_frames_dir / 'loop_0'
        output_frames_dir.mkdir(exist_ok=True, parents=True)

        settings_file_name = f'{timestamp}.json'
        with open(output_dir/settings_file_name, 'w', encoding='utf-8') as f:
            json.dump(args_dict, f, indent=4, ensure_ascii=False)

        input_dir = Path(input_dir)
        assert input_dir.exists()
        if input_dir.is_file():
            extract_dir = output_dir / 'input_frames'
            extract_dir.mkdir()
            os.system(f"ffmpeg -i {input_dir} {extract_dir / '%07d.png'}")
            input_dir = extract_dir
        image_list = shared.listfiles(input_dir)  # sorted
        image_list = image_list[::extract_nth_frame]
        image_list = image_list[:max_frames]
        image_n = len(image_list)

        temporal_superimpose_alpha_list = \
            [float(x) for x in temporal_superimpose_alpha_list.split(',') if x
             ] or [1]
        assert len(temporal_superimpose_alpha_list) % 2 == 1

        shared.state.begin()
        shared.state.job_count = loop_n * image_n

        schedule_args = dict(**math.__dict__)

        for loop_i in range(loop_n):
            if loop_i == 1 and not save_every_loop:
                image_list = shared.listfiles(output_frames_dir)  # sorted
            elif loop_i > 0 and save_every_loop:
                image_list = shared.listfiles(output_frames_dir)
                output_frames_dir = output_frames_dir.parent/f"loop_{loop_i}"
                output_frames_dir.mkdir(exist_ok=True, parents=True)

            img_que = TemporalImageBlender(
                image_path_list=image_list,
                window_size=len(temporal_superimpose_alpha_list),
                target_size=(p.width, p.height),
                use_mask=use_mask, mask_dir=mask_dir,
                mask_threshold=mask_threshold
            )

            for image_i, image_path in enumerate(image_list):
                print(f"Loop:{loop_i + 1}/{loop_n},Image:{image_i + 1}/{image_n}")
                # shared.state.job = f"Loop:{loop_i + 1}/{loop_n},Image:{image_i + 1}/{image_n}"

                # do all schedule
                schedule_args.update({'image_i': image_i, 'loop_i': loop_i})
                if subseed_strength_schedule:
                    p.subseed_strength = eval(subseed_strength_schedule, schedule_args)
                    print(f"subseed_strength_schedule:{p.subseed_strength}")
                if denoising_schedule:
                    p.denoising_strength = eval(denoising_schedule, schedule_args)
                    print(f"denoising_schedule:{p.denoising_strength}")
                if seed_schedule:
                    p.seed = eval(seed_schedule, schedule_args)
                    print(f"seed_schedule:{p.seed}")
                if subseed_schedule:
                    p.subseed = eval(subseed_schedule, schedule_args)
                    print(f"subseed_schedule:{p.subseed}")
                if cfg_schedule:
                    p.cfg_scale = eval(cfg_schedule, schedule_args)
                    print(f"cfg_schedule:{p.cfg_scale}")
                if superimpose_alpha_schedule:
                    superimpose_alpha = eval(superimpose_alpha_schedule, schedule_args)
                    print(f"superimpose_alpha_schedule:{superimpose_alpha}")
                if temporal_superimpose_schedule:
                    new_temporal_superimpose_list = eval(temporal_superimpose_schedule, schedule_args)
                    assert len(temporal_superimpose_alpha_list) == len(new_temporal_superimpose_list)
                    temporal_superimpose_alpha_list = new_temporal_superimpose_list
                    print(f"temporal_superimpose_schedule:{temporal_superimpose_alpha_list}")
                if prompt_schedule:
                    p.prompt = eval(prompt_schedule, schedule_args)
                    print(f"prompt_schedule:{p.prompt}")
                if negative_prompt_schedule:
                    p.negative_prompt = eval(negative_prompt_schedule, schedule_args)
                    print(f"negative_prompt_schedule:{p.negative_prompt}")
                if batch_count_schedule:
                    p.n_iter = eval(batch_count_schedule, schedule_args)
                    print(f"batch_count_schedule:{p.n_iter}")

                output_filename = output_frames_dir / f"{image_i:07d}.png"

                # make base img for i2i
                base_img = img_que.blend_temporal(temporal_superimpose_alpha_list)

                print(f"seed:{p.seed}, subseed:{p.subseed}")

                p.image_mask = img_que.current_mask()
                # mask像素为0表示不变

                p.init_images = [base_img]
                processed = processing.process_images(p)
                output_img = img_que.blend_batch(
                        processed.images, superimpose_alpha)
                output_img.save(output_filename)

                img_que.move_to_next()

                if not fix_seed and not seed_schedule:
                    p.seed = processed.seed + len(processed.images)
                if not fix_subseed and not subseed_schedule:
                    p.subseed = processed.subseed + len(processed.images)

            if save_every_loop:
                output_video_name = f'{timestamp}-loop_{loop_i}.mp4'
                make_video(
                    input_dir=output_frames_dir,
                    output_filename=output_dir/output_video_name,
                    frame_rate=output_frame_rate
                )

        output_video_name = f'{timestamp}.mp4'
        make_video(
            input_dir=output_frames_dir,
            output_filename=output_dir / output_video_name,
            frame_rate=output_frame_rate
        )

        print(f"\n {timestamp} finished! now time:{get_now_time()}\n")
        shared.state.end()

        return processed
