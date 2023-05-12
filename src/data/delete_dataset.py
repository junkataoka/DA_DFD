# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import os
import torch
from glob import glob
from collections import defaultdict
import gc
gc.set_threshold(0)


def DeleteGeomFromSrc(output_filepath, dir_tar_geom, dir_src_recipe, dir_src_heatmap, dir_src_geom):
    tar_geoms = glob(os.path.join(output_filepath, f"{dir_tar_geom}/*"))
    src_recipes = glob(os.path.join(output_filepath, f"{dir_src_recipe}/*"))
    src_heatmaps = glob(os.path.join(output_filepath, f"{dir_src_heatmap}/*"))
    src_geoms = glob(os.path.join(output_filepath, f"{dir_src_geom}/*"))
    src_recipe_map = defaultdict(list)
    src_heatmap_map = defaultdict(list)

    for i in range(len(src_recipes)):
        src_recipe_map[src_recipes[i].split("/")[-1].split("-")[0]].append(src_recipes[i])
    for i in range(len(src_heatmaps)):
        src_heatmap_map[src_heatmaps[i].split("/")[-1].split("-")[0]].append(src_heatmaps[i])

    remove_geoms= set()

    
    for tar_geom in tar_geoms:
        tar_geom_arr = np.loadtxt(tar_geom, delimiter=" ").tolist()
        for src_geom in src_geoms:
            if src_geom not in remove_geoms:
                src_geom_arr = np.loadtxt(src_geom, delimiter=" ").tolist()
                if tar_geom_arr == src_geom_arr:
                    remove_geoms.add(src_geom)
    remove_recipes = list()
    remove_heatmaps = list()

    for remove_geom in remove_geoms:
        recipe_id = remove_geom.split("/")[-1].split("-")[0]
        if recipe_id in src_heatmap_map:
            remove_heatmaps += src_heatmap_map[recipe_id]
        if recipe_id in src_recipe_map:
            remove_recipes += src_recipe_map[recipe_id]

    remove_files = remove_recipes + remove_heatmaps + list(remove_geoms)
    print(len(remove_files))

    for path in remove_files:
        if os.path.exists(path):
            os.remove(path)

def DeleteRecipeFromSrc(output_filepath, dir_tar_recipe, dir_src_recipe, dir_src_heatmap, dir_src_geom):
    src_recipes = glob(os.path.join(output_filepath, f"{dir_src_recipe}/*"))
    src_heatmaps = glob(os.path.join(output_filepath, f"{dir_src_heatmap}/*"))
    src_geoms = glob(os.path.join(output_filepath, f"{dir_src_geom}/*"))
    src_recipes = glob(os.path.join(output_filepath, f"{dir_src_recipe}/*"))
    tar_recipes = glob(os.path.join(output_filepath, f"{dir_tar_recipe}/*"))
    src_heatmap_map = defaultdict(list)
    src_geom_map = defaultdict(list)

    for i in range(len(src_heatmaps)):
        src_heatmap_map[src_heatmaps[i].split("/")[-1].split("-")[0]].append(src_heatmaps[i])
    for i in range(len(src_geoms)):
        src_geom_map[src_geoms[i].split("/")[-1].split("-")[0]].append(src_geoms[i])

    remove_recipes = set()

    for tar_recipe in tar_recipes:
        tar_recipe_arr = np.loadtxt(tar_recipe, delimiter=" ").tolist()
        for src_recipe in src_recipes:
            if src_recipe not in remove_recipes:
                src_recipe_arr = np.loadtxt(src_recipe, delimiter=" ").tolist()
                if tar_recipe_arr ==  src_recipe_arr:
                    remove_recipes.add(src_recipe)

    remove_heatmaps = []
    remove_geoms = []

    for remove_recipe in list(remove_recipes):
        recipe_id = remove_recipe.split("/")[-1].split(".")[0]
        if recipe_id in src_heatmap_map:
            remove_heatmaps += src_heatmap_map[recipe_id]
        if recipe_id in src_geom_map:
            remove_geoms += src_geom_map[recipe_id]

    delete_paths = list(remove_recipes) + remove_heatmaps + remove_geoms

    for path in delete_paths:
        if os.path.exists(path):
            os.remove(path)

@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('deleting data related to target geomery from source data set')
    DeleteGeomFromSrc(output_filepath, dir_tar_geom="train-tar-GEOM", dir_src_recipe="train-src-RECIPE", dir_src_heatmap="train-src-HEATMAP", dir_src_geom="train-src-GEOM")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()