from io import BytesIO
import os
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

from web.model import MoCo
from web.utils import BaseState, set_up_driver, get_search_url, scroll_to_end
from web.utils import generate_random_filename, cluster_features
from web.navbar import navbar


class MOCOState(BaseState):
    prompt = ""
    image_url = "https://images.pexels.com/photos/2607544/pexels-photo-2607544.jpeg?cs=srgb&dl=pexels-simona-kidri%C4%8D-2607544.jpg&fm=jpg"
    image_processing = False
    image_made = False
    cluster_text = "Choose cluster"
    download = False
    filter = False
    max_capacity = 10
    eps = "1.5"
    num_filter: t.List[int] = []
    image_list: t.List[str] = []
    _backup_image_list: t.List[str] = []
    _model: torch.nn.Module = MoCo(device="cuda")
    _driver: t.Any = set_up_driver()
    _device: str = "cuda"
    _filter_groups: t.List[str] = []
    _transform: t.Any = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def process_image(self):
        self.filter = False
        self.image_made = False
        self.image_processing = True
        self.image_list.clear()

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
            self._backup_image_list = temp
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

    def set_cluster(self, target_cluster):
        self.cluster_text = f"Cluster {target_cluster}"
        self.image_list = [
            self._backup_image_list[idx]
            for idx in self._filter_groups[target_cluster - 1]
        ]

    def set_filter(self):
        self.image_processing = True
        self.image_made = False
        self.cluster_text = "Choose cluster"

    def filter_all(self):
        with torch.no_grad():
            temp_images = []
            for img in self.image_list:
                if img.startswith("http"):
                    img = self._transform(
                        Image.open(BytesIO(requests.get(img).content)).convert("RGB")
                    ).to(self._device)
                else:
                    img = self._transform(
                        Image.open(
                            BytesIO(base64.b64decode(img.split(";base64,", 1)[1]))
                        ).convert("RGB")
                    ).to(self._device)
                temp_images.append(img)

            if len(temp_images):
                self._filter_groups.clear()
                images = torch.stack(temp_images, dim=0)
                logits = self._model(images).cpu().numpy()
                try:
                    eps = float(self.eps)
                except ValueError:
                    eps = 1.5
                    self.eps = "1.5"
                self._filter_groups = cluster_features(logits, eps=eps)
                self.num_filter = list(range(1, len(self._filter_groups) + 1))

        self.filter = True
        self.image_made = True
        self.image_processing = False


def moco():
    return navbar(
        pc.center(
            pc.vstack(
                pc.vstack(
                    pc.heading("MoCo-based Model", font_size="1.5em"),
                    pc.input(
                        placeholder="Number of images default to 10",
                        on_blur=MOCOState.set_max_capacity,
                        type_="number",
                    ),
                    pc.input(
                        placeholder="Enter word for scraping...",
                        on_blur=MOCOState.set_prompt,
                    ),
                    pc.button(
                        "Collect Data",
                        on_click=[MOCOState.process_image, MOCOState.get_image],
                        width="100%",
                        is_loading=MOCOState.image_processing,
                    ),
                    pc.divider(),
                    pc.cond(
                        MOCOState.image_made,
                        pc.input(
                            value=MOCOState.eps,
                            placeholder="Enter eps for distance...",
                            on_change=MOCOState.set_eps,
                        ),
                    ),
                    pc.cond(
                        MOCOState.filter,
                        pc.menu(
                            pc.menu_button(
                                pc.button(
                                    MOCOState.cluster_text,
                                    color_scheme="green",
                                    width="100%",
                                ),
                                width="100%",
                            ),
                            pc.menu_list(
                                pc.foreach(
                                    MOCOState.num_filter,
                                    lambda m: pc.menu_item(
                                        m, on_click=MOCOState.set_cluster(m)
                                    ),
                                ),
                                overflow_y="scroll",
                                max_height="30vh",
                            ),
                        ),
                    ),
                    pc.cond(
                        MOCOState.image_processing,
                        pc.circular_progress(is_indeterminate=True),
                        pc.cond(
                            MOCOState.image_made,
                            pc.flex(
                                pc.foreach(
                                    MOCOState.image_list,
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
                        MOCOState.image_made,
                        pc.button(
                            "Download",
                            width="100%",
                            on_click=[MOCOState.set_download, MOCOState.download_all],
                            color_scheme="blue",
                            is_loading=MOCOState.download,
                        ),
                    ),
                    pc.cond(
                        MOCOState.image_made,
                        pc.button(
                            "Filter",
                            width="100%",
                            on_click=[MOCOState.set_filter, MOCOState.filter_all],
                            color_scheme="gray",
                            is_loading=(not MOCOState.filter),
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
