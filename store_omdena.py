import os
import numpy as np
from PIL import Image
from hub import Transform, dataset
import pandas as pd


class OmdenaGenerator(Transform):
    def __init__(self, path):
        self._path = path

    def meta(self):
        return {
            "image_lat": {"shape": (1,), "dtype": "float", "dtag" : "text"},
            "image_lon": {"shape": (1,), "dtype": "float", "dtag" : "text"},
            "cluster_lat": {"shape": (1,), "dtype": "float", "dtag" : "text"},
            "cluster_lon": {"shape": (1,), "dtype": "float", "dtag" : "text"},
            "cons_pc": {"shape": (1,), "dtype": "float", "dtag" : "text"},
            "nightlights": {"shape": (1,), "dtype": "float", "dtag" : "text"},
            "nightlights_bin": {"shape": (1,), "dtype": "int", "dtag" : "text"},
            "image": {"shape": (1,), "dtype": "object", "chunksize": 100, "dtag" : "image"},
        }

    def forward(self, image_info):
        ds = {}

        ds["image_lat"] = np.empty(1, dtype="float")
        ds["image_lat"][0] = image_info["image_lat"]

        ds["image_lon"] = np.empty(1, dtype="float")
        ds["image_lon"][0] = image_info["image_lon"]

        ds["cluster_lat"] = np.empty(1, dtype="float")
        ds["cluster_lat"][0] = image_info["cluster_lat"]

        ds["cluster_lon"] = np.empty(1, dtype="float")
        ds["cluster_lon"][0] = image_info["cluster_lon"]

        ds["cons_pc"] = np.empty(1, dtype="float")
        ds["cons_pc"][0] = image_info["cons_pc"]

        ds["nightlights"] = np.empty(1, dtype="float")
        ds["nightlights"][0] = image_info["nightlights"]

        ds["nightlights_bin"] = np.empty(1, dtype="int")
        ds["nightlights_bin"][0] = image_info["nightlights_bin"]

        ds["image"] = np.empty(1, object)
        ds["image"][0] = np.array(Image.open(get_image_path(image_info["country"], image_info["image_name"], self._path)))
        return ds


def load_dataset(path):
    df = pd.read_csv(os.path.join(path, "processed", "image_download_locs.csv"))
    country_dirs = os.listdir(os.path.join(path, "countries"))
    image_info_list = []
    for dir in country_dirs:
        images_path = os.path.join(path, "countries", dir, "images")
        image_list = os.listdir(images_path)
        for image_name in image_list:
            image_info = {}
            image_info["image_name"] = image_name
            res_df = df.loc[df["image_name"] == image_name]
            image_info["image_lat"] = res_df.iloc[0]["image_lat"]
            image_info["image_lon"] = res_df.iloc[0]["image_lon"]
            image_info["cluster_lat"] = res_df.iloc[0]["cluster_lat"]
            image_info["cluster_lon"] = res_df.iloc[0]["cluster_lon"]
            image_info["cons_pc"] = res_df.iloc[0]["cons_pc"]
            image_info["nightlights"] = res_df.iloc[0]["nightlights"]
            image_info["country"] = res_df.iloc[0]["country"]
            image_info["nightlights_bin"] = res_df.iloc[0]["nightlights_bin"]
            image_info_list.append(image_info)

    ds = dataset.generate(OmdenaGenerator(path), image_info_list)
    return ds


def get_image_path(country, image_name, dataset_path):
    folder = ""
    if country == "mw":
        folder = "malawi_2016"
    elif country == "ng":
        folder = "nigeria_2015"
    elif country == "eth":
        folder = "ethiopia_2015"
    return os.path.join(dataset_path, "countries", folder, "images", image_name)


path = "./data/"
ds = load_dataset(path)

ds.store("abhinav/aerial-omdena")
