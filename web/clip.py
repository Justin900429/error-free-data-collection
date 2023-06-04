from io import BytesIO
import os
import zipfile
import base64
import typing as t

import requests
import urllib.request

import torch
import clip
from PIL import Image
from selenium.webdriver.common.by import By
import pynecone as pc


from web.utils import (
    BaseState,
    LRUCache,
    set_up_driver,
    get_search_url,
    scroll_to_end,
    generate_random_filename,
)
from web.navbar import navbar


class CLIPState(BaseState):
    prompt = ""
    image_url = "https://images.pexels.com/photos/2607544/pexels-photo-2607544.jpeg?cs=srgb&dl=pexels-simona-kidri%C4%8D-2607544.jpg&fm=jpg"
    image_processing = False
    image_made = False
    download = False
    filter = False
    max_capacity = 10
    image_list: t.List[str] = []
    _model: tuple = clip.load("ViT-B/32", device="cpu")
    _cache: LRUCache = LRUCache()
    _driver: t.Any = set_up_driver()
    _device: str = "cpu"

    def process_image(self):
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

        return pc.redirect(f"http://{os.getenv('IP_ADDRESS')}:8000/download/{filename}")

    def set_filter(self):
        self.image_processing = True
        self.image_made = False
        self.filter = True

    def filter_all(self):
        with torch.no_grad():
            if self.prompt in self._cache:
                text = self._cache.get(self.prompt).to(self._device)
            else:
                text = clip.tokenize([f"a {self.prompt}", "a blank"]).to(self._device)
                self._cache.update(self.prompt, text.cpu())

            remove_idx = set()
            temp_images = []
            for idx, img in enumerate(self.image_list):
                if img.startswith("http"):
                    try:
                        img = self._model[1](
                            Image.open(BytesIO(requests.get(img).content)).convert(
                                "RGB"
                            )
                        ).to(self._device)
                    except:
                        img = torch.randn(1, 3, 224, 224)
                        remove_idx.add(idx)
                else:
                    img = self._model[1](
                        Image.open(
                            BytesIO(base64.b64decode(img.split(";base64,", 1)[1]))
                        ).convert("RGB")
                    ).to(self._device)
                temp_images.append(img)

            if len(temp_images):
                images = torch.stack(temp_images, dim=0)
                logits_per_image, _ = self._model[0](images, text)
                left_images = (
                    logits_per_image.softmax(dim=-1).cpu()[:, 0].numpy() >= 0.9
                ).nonzero()[0]
                self.image_list = [
                    self.image_list[idx] for idx in left_images if idx not in remove_idx
                ]

        self.filter = False
        self.image_processing = False
        self.image_made = True


def index():
    return navbar(
        pc.center(
            pc.vstack(
                pc.vstack(
                    pc.heading("CLIP-based Model", font_size="1.5em"),
                    pc.input(
                        placeholder="Number of images",
                        on_blur=CLIPState.set_max_capacity,
                        type_="number",
                    ),
                    pc.input(
                        placeholder="Enter word for scraping...",
                        on_blur=CLIPState.set_prompt,
                    ),
                    pc.button(
                        "Collect Data",
                        on_click=[CLIPState.process_image, CLIPState.get_image],
                        width="100%",
                        is_loading=CLIPState.image_processing,
                    ),
                    pc.divider(),
                    pc.cond(
                        CLIPState.image_processing,
                        pc.circular_progress(is_indeterminate=True),
                        pc.cond(
                            CLIPState.image_made,
                            pc.flex(
                                pc.foreach(
                                    CLIPState.image_list,
                                    lambda m: pc.image(
                                        src=m, height="10em", width="10em"
                                    ),
                                ),
                                direction="row",
                                wrap="wrap",
                                overflow_y="scroll",
                                max_height="50vh",
                                justify="center",
                                max_width="1500px",
                            ),
                        ),
                    ),
                    pc.cond(
                        CLIPState.image_made,
                        pc.button(
                            "Download",
                            width="100%",
                            on_click=[CLIPState.set_download, CLIPState.download_all],
                            color_scheme="blue",
                            is_loading=CLIPState.download,
                        ),
                    ),
                    pc.cond(
                        CLIPState.image_made,
                        pc.button(
                            "Filter",
                            width="100%",
                            on_click=[CLIPState.set_filter, CLIPState.filter_all],
                            color_scheme="gray",
                            is_loading=CLIPState.filter,
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
