import os, datetime, imghdr
from pathlib import Path
from typing import List, Tuple, Iterable
from PIL import Image

from modules import shared
from modules import images

resize_mode = 0  # utils.resize_mode = p.resize_mode


def resize_img(img, target_size):
    width, height = target_size
    return images.resize_image(resize_mode, img, width, height)
    # return img.resize(target_size, Image.ANTIALIAS)


def make_video(
        input_dir, output_filename,
        frame_rate=12, input_format='%07d.png'):
    os.system(
        f"ffmpeg -r {frame_rate} "
        f' -i "{Path(input_dir) / input_format}" '
        " -c:v libx264 "
        # " -c:v mpeg4 "
        " -qp 0 "
        # " -preset ultrafast "
        " -pix_fmt yuv420p "
        f' "{output_filename}" '
    )


def is_image(filename) -> bool:
    filename = Path(filename)
    return filename.is_file() and imghdr.what(filename) is not None


def get_image_paths(input_dir: Path) -> List[Path]:
    path_list: List[Path] = shared.listfiles(input_dir)  # sorted
    return [Path(p) for p in path_list if is_image(p)]


def get_prompt_for_images(
        image_path_list, prompt_suffix='.txt'
) -> List[Tuple[str, str]]:
    prompt_list = []
    for p in image_path_list:
        prompt_filename = p.with_suffix(prompt_suffix)
        if prompt_filename.is_file():
            with open(prompt_filename, 'r') as f:
                s = f.read().split(' --neg ', 1)
                prompt_list.append(
                    (s[0], s[1] if len(s) > 1 else None)
                )
        else:
            prompt_list.append((None, None))
    return prompt_list


def blend_average(img_iter: Iterable[Image.Image]) -> Image.Image:
    img_iter = iter(img_iter)
    new_img = next(img_iter)
    for i, img in enumerate(img_iter, 1):
        new_img = Image.blend(new_img, img, 1.0 / (i + 1))
    return new_img


def get_now_time() -> str:
    SHA_TZ = datetime.timezone(
        datetime.timedelta(hours=8),
        name='Asia/Shanghai',
    )
    fmt = '%y%m%d_%H%M%S'
    utc_now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)
    stamp = beijing_now.strftime(fmt)
    return stamp
