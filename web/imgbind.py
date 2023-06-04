from io import BytesIO
import os
import sys
import zipfile
import base64
import typing as t

import requests
import urllib.request

import torch
from torchvision import transforms

from PIL import Image
from selenium.webdriver.common.by import By
import pynecone as pc

from web.utils import BaseState, set_up_driver, get_search_url
from web.utils import scroll_to_end, generate_random_filename
from web.navbar import navbar

sys.path.append("ImageBind")

import data
from models import imagebind_model
from models.imagebind_model import ModalityType


class BINDState(BaseState):
    prompt = ""
    image_url = "https://images.pexels.com/photos/2607544/pexels-photo-2607544.jpeg?cs=srgb&dl=pexels-simona-kidri%C4%8D-2607544.jpg&fm=jpg"
    image_processing = False
    image_made = False
    fine_grained_prompt = ""
    download = False
    fine_grained_filter = False
    max_capacity = 10
    num_filter: t.List[int] = []
    image_list: t.List[str] = []
    _backup_image_list: t.List[str] = []
    _model: torch.nn.Module = (
        imagebind_model.imagebind_huge(pretrained=True).to("cuda").eval()
    )
    _driver: t.Any = set_up_driver()
    _device: str = "cuda"
    _filter_groups: t.List[str] = []
    _transform: t.Any = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    def process_image(self):
        self.filter = False
        self.fine_grained_filter = False
        self.image_made = False
        self.image_processing = True

    def get_image(self):
        try:
            max_capacity = int(self.max_capacity)
            if max_capacity <= 0:
                raise ValueError()
            max_capacity = min(max_capacity, 100)
        except:
            max_capacity = 10

        if len(self.prompt):
            temp = []
            self._driver.get(get_search_url(self.prompt))
            image_elements = self._driver.find_elements(By.CLASS_NAME, "rg_i")
            while len(image_elements) < max_capacity:
                image_elements = self._driver.find_elements(By.CLASS_NAME, "rg_i")
                scroll_to_end(self._driver)
            for image in image_elements[:max_capacity]:
                if image.get_attribute("src") is not None:
                    temp.append(image.get_attribute("src"))

            self.image_list = temp
            self.image_made = True
        self.image_processing = False

    def set_download(self):
        self.image_processing = True
        self.download = True
        self.image_made = False

    def download_all(self):
        filename = generate_random_filename()
        while os.path.exists(filename):
            filename = generate_random_filename()

        zip = zipfile.ZipFile("zipfiles/" + filename + ".zip", "w")
        for idx, img in enumerate(self.image_list):
            if img.startswith("http"):
                temp_img = urllib.request.urlopen(img)
                zip.writestr(f"img{idx}.jpg", temp_img.read())
            else:
                save_image = img.split(";base64,")[1]
                img_data = base64.b64decode(save_image)
                zip.writestr(f"img{idx}.jpg", img_data)
        zip.close()

        self.image_processing = False
        self.download = False
        self.image_made = True

        return pc.redirect(f"{os.environ.get('download_link')}/{filename}")

    def start_filter(self):
        self.image_processing = True
        self.image_made = False

    def start_fine_grained_filter(self, dump_param):
        self.start_filter()
        self.fine_grained_filter = False

    def filter_all(self):
        with torch.no_grad():
            text_output = []
            img_output = []
            text_list = [f"A {self.prompt}", f"Not a {self.prompt}"]
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(
                    text_list, self._device
                ),
            }
            embeddings = self._model(inputs)
            text_output = embeddings[ModalityType.TEXT].cpu()
            del embeddings

            for img in self.image_list:
                if img.startswith("http"):
                    try:
                        img = self._transform(
                            Image.open(BytesIO(requests.get(img).content)).convert(
                                "RGB"
                            )
                        ).to(self._device)
                    except:
                        continue
                else:
                    img = self._transform(
                        Image.open(
                            BytesIO(base64.b64decode(img.split(";base64,", 1)[1]))
                        ).convert("RGB")
                    ).to(self._device)

                inputs = {
                    ModalityType.VISION: img.unsqueeze(0),
                }
                embeddings = self._model(inputs)
                img_output.append(embeddings[ModalityType.VISION].cpu().squeeze(0))

            img_output = torch.stack(img_output)
            prob = torch.softmax(img_output @ text_output.T, dim=-1)
            left_images = (prob.argmax(dim=1) == 0).numpy().nonzero()[0]
            self.image_list = [self.image_list[idx] for idx in left_images]

        self._backup_image_list = self.image_list
        self.fine_grained_filter = True
        self.fine_grained_prompt = ""

    def fine_grained_filter_all(self, fine_grained_prompt):
        if not len(self.fine_grained_prompt):
            self.image_list = self._backup_image_list
        else:
            with torch.no_grad():
                text_output = []
                img_output = []
                text_list = [
                    f"{self.prompt} {fine_grained_prompt}",
                    f"{self.prompt} not {fine_grained_prompt}",
                ]
                inputs = {
                    ModalityType.TEXT: data.load_and_transform_text(
                        text_list, self._device
                    ),
                }
                embeddings = self._model(inputs)
                text_output = embeddings[ModalityType.TEXT].cpu()
                del embeddings

                removed_idx = set()
                for idx, img in enumerate(self._backup_image_list):
                    if img.startswith("http"):
                        try:
                            img = self._transform(
                                Image.open(BytesIO(requests.get(img).content)).convert(
                                    "RGB"
                                )
                            ).to(self._device)
                        except:
                            removed_idx.add(idx)
                            img = torch.randn(1, 3, 224, 224).to(self._device)
                    else:
                        img = self._transform(
                            Image.open(
                                BytesIO(base64.b64decode(img.split(";base64,", 1)[1]))
                            ).convert("RGB")
                        ).to(self._device)

                    inputs = {
                        ModalityType.VISION: img.unsqueeze(0),
                    }
                    embeddings = self._model(inputs)
                    img_output.append(embeddings[ModalityType.VISION].cpu().squeeze(0))

                img_output = torch.stack(img_output)
                prob = torch.softmax(img_output @ text_output.T, dim=-1)
                left_images = (prob.argmax(dim=1) == 0).numpy().nonzero()[0]
                self.image_list = [
                    self._backup_image_list[idx]
                    for idx in left_images
                    if idx not in removed_idx
                ]

    def end_filter(self):
        self.image_made = True
        self.image_processing = False

    def end_fine_grained_fitler(self, dump_param):
        self.end_filter()
        self.fine_grained_filter = True


def bind():
    return navbar(
        pc.center(
            pc.vstack(
                pc.vstack(
                    pc.heading("ImageBind-based Model", font_size="1.5em"),
                    pc.input(
                        placeholder="Number of images default to 10",
                        on_blur=BINDState.set_max_capacity,
                        type_="number",
                    ),
                    pc.input(
                        placeholder="Enter word for scraping...",
                        on_blur=BINDState.set_prompt,
                    ),
                    pc.button(
                        "Collect Data",
                        on_click=[BINDState.process_image, BINDState.get_image],
                        width="100%",
                        is_loading=BINDState.image_processing,
                    ),
                    pc.divider(),
                    pc.cond(
                        BINDState.fine_grained_filter,
                        pc.input(
                            value=BINDState.fine_grained_prompt,
                            placeholder="Prompt for fine-grained filtering...",
                            on_change=BINDState.set_fine_grained_prompt,
                            on_blur=[
                                BINDState.start_fine_grained_filter,
                                BINDState.fine_grained_filter_all,
                                BINDState.end_fine_grained_fitler,
                            ],
                        ),
                    ),
                    pc.cond(
                        BINDState.image_processing,
                        pc.circular_progress(is_indeterminate=True),
                        pc.cond(
                            BINDState.image_made,
                            pc.flex(
                                pc.foreach(
                                    BINDState.image_list,
                                    lambda m: pc.image(
                                        src=m, height="10em", width="10em"
                                    ),
                                ),
                                direction="row",
                                wrap="wrap",
                                overflow_y="scroll",
                                max_height="45vh",
                                justify="center",
                                max_width="1500px",
                            ),
                        ),
                    ),
                    pc.cond(
                        BINDState.image_made,
                        pc.button(
                            "Download",
                            width="100%",
                            on_click=[BINDState.set_download, BINDState.download_all],
                            color_scheme="blue",
                            is_loading=BINDState.download,
                        ),
                    ),
                    pc.cond(
                        BINDState.image_made,
                        pc.button(
                            "Filter",
                            width="100%",
                            on_click=[
                                BINDState.start_filter,
                                BINDState.filter_all,
                                BINDState.end_filter,
                            ],
                            color_scheme="gray",
                        ),
                    ),
                    bg="white",
                    padding="2em",
                    shadow="lg",
                    border_radius="lg",
                ),
            ),
            width="100%",
            overflow_y="scroll",
            min_height="calc(100vh - 50px - 2em)",
            background="radial-gradient(circle at 22% 11%,rgba(178, 132, 201,.20),hsla(0,0%,100%,0) 19%),radial-gradient(circle at 82% 25%,rgba(33,150,243,.18),hsla(0,0%,100%,0) 35%),radial-gradient(circle at 25% 61%,rgba(129, 132, 45, .28),hsla(0,0%,100%,0) 55%)",
        )
    )
