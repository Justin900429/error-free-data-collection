import os
import base64
import time
import urllib.request
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def set_up_driver():
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1920,1200")

    DRIVER_PATH = "./chromedriver"
    return webdriver.Chrome(executable_path=DRIVER_PATH)


def scroll_to_end():
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)


if __name__ == "__main__":
    driver = set_up_driver()
    NUM_OF_CLASSES = 100
    NUM_OF_IMAGES_PER_CLASS = 200
    SAVE_PATH = "./images/"
    os.makedirs(SAVE_PATH, exist_ok=True)

    with open("animal_list.txt", "r") as f:
        animal_list = f.read().splitlines()
    animal_list = animal_list[:NUM_OF_CLASSES]

    for animal in animal_list:
        if os.path.exists(f"{SAVE_PATH}/{animal}"):
            continue
        search_url = (
            "https://www.google.com/search?q="
            + "+".join(animal.split(" "))
            + "+animal&tbm=isch"
        )
        os.makedirs(f"{SAVE_PATH}/{animal}", exist_ok=True)
        driver.get(search_url)
        count_image = 0
        image_elements = driver.find_elements(By.CLASS_NAME, "rg_i")
        while len(image_elements) < NUM_OF_IMAGES_PER_CLASS:
            scroll_to_end()
            image_elements = driver.find_elements(By.CLASS_NAME, "rg_i")
        for image in image_elements:
            if image.get_attribute("src") is not None:
                save_image = image.get_attribute("src").split("data:image/jpeg;base64,")
                filename = f"{SAVE_PATH}/{animal}/{count_image}.jpg"
                if len(save_image) > 1:
                    with open(filename, "wb") as f:
                        f.write(base64.b64decode(save_image[1]))
                else:
                    urllib.request.urlretrieve(image.get_attribute("src"), filename)
                count_image += 1
    driver.quit()
