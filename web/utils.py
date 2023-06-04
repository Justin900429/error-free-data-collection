import os
import time
from collections import OrderedDict, defaultdict
import string
import random

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import numpy as np
from sklearn.cluster import DBSCAN
import pynecone as pc


class BaseState(pc.State):
    """The base state for the app."""


def get_search_url(search_term):
    return (
        "https://www.google.com/search?q="
        + "+".join(search_term.split(" "))
        + "&tbm=isch"
    )


def scroll_to_end(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)


def set_up_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")

    DRIVER_PATH = "./chromedriver"
    return webdriver.Chrome(executable_path=DRIVER_PATH, options=options)


def generate_random_filename(size=10):
    # Generate a random string of lowercase letters and digits
    letters_and_digits = string.ascii_lowercase + string.digits
    random_string = "".join(random.choice(letters_and_digits) for _ in range(size))
    return random_string


def cluster_features(features, eps=1.2):
    # choose_features = len(features) // 10

    # # Compute the l2 distance of features
    # dis = np.sum((features[None, :, :] - features[:, None, :]) ** 2, axis=2)
    # dis = dis + np.eye(len(features)) * 1e10
    # print(np.amin(dis, axis=1))
    # print(dis)
    # partiion_dis = np.partition(dis, choose_features, axis=1)
    # mean_dis = np.mean(partiion_dis[:, 1 : choose_features + 1])

    clustering = DBSCAN(eps=eps, min_samples=1).fit(features)
    cluster_label = defaultdict(list)
    for idx in range(len(features)):
        cluster_label[clustering.labels_[idx]].append(idx)
    return list(cluster_label.values())


async def remove_zipfiles(filename: str):
    if os.path.exists("zipfiles/" + filename + ".zip"):
        os.remove("zipfiles/" + filename + ".zip")


async def file_download(filename: str):
    return "zipfiles/" + filename + ".zip"


class LRUCache:
    def __init__(self, capacity: int = 10):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def update(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __contains__(self, key):
        return key in self.cache
