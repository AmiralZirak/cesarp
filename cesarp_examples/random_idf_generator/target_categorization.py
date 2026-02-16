# coding=utf-8
import csv
import os
import multiprocessing
import logging.config
from typing import List, Tuple, Dict
from dataclasses import dataclass

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate

import cesarp.common


@dataclass
class BuildingMetrics:
    """Data class to hold building geometric metrics"""
    building_id: int
    orientation_angle: float
    orientation_category: int
    area: float
    elongation: float


class GeometryAnalyzer:
    """Handles geometric calculations for building shapes"""

    @staticmethod
    def find_longest_edge(polygon: Polygon) -> Tuple[np.ndarray, np.ndarray]:
        coords = np.array(polygon.exterior.coords)
        max_length, longest_edge = 0, None
        for i in range(len(coords) - 1):
            start, end = coords[i], coords[i + 1]
            edge_vector = end - start
            length = np.linalg.norm(edge_vector)
            if length > max_length:
                max_length = length
                longest_edge = (start, end)
        return longest_edge

    @staticmethod
    def compute_orientation_from_edge(longest_edge: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, int]:
        start, end = longest_edge
        edge_vector = np.array(end) - np.array(start)
        perp = np.array([-edge_vector[1], edge_vector[0]])
        perp /= np.linalg.norm(perp)

        angle = np.degrees(np.arctan2(perp[1], perp[0])) % 360

        if angle < 22.5: category = 1
        elif angle < 67.5: category = 2
        elif angle < 112.5: category = 3
        elif angle < 157.5: category = 4
        elif angle < 202.5: category = 1
        elif angle < 247.5: category = 2
        elif angle < 292.5: category = 3
        elif angle < 337.5: category = 4
        else: category = 1

        return angle, category

    @staticmethod
    def calculate_metrics(x_coords: List[float], y_coords: List[float]) -> Tuple[float, int, float, float]:
        poly = Polygon(zip(x_coords, y_coords))
        longest_edge = GeometryAnalyzer.find_longest_edge(poly)
        angle, orientation_category = GeometryAnalyzer.compute_orientation_from_edge(longest_edge)
        area = poly.area
        elongation = GeometryAnalyzer.calculate_elongation(x_coords, y_coords)
        return angle, orientation_category, area, elongation

    @staticmethod
    def calculate_elongation(x_coords: List[float], y_coords: List[float]) -> float:
        poly = Polygon(zip(x_coords, y_coords))
        min_area, min_bbox = float('inf'), None
        for angle in np.arange(0, 180, 2):
            rotated = rotate(poly, angle, use_radians=False)
            bounds = rotated.bounds
            width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
            area = width * height
            if area < min_area:
                min_area, min_bbox = area, bounds
        w, h = min_bbox[2] - min_bbox[0], min_bbox[3] - min_bbox[1]
        shorter, longer = sorted([w, h])
        return shorter / longer if longer > 0 else 0


class BuildingCategorizer:
    """Main class for building categorization"""

    CONSTRUCTION_PERIODS = {
        'Pre-1919': (0, 1918),
        '1919-1945': (1919, 1945),
        '1946-1970': (1946, 1970),
        '1971-1980': (1971, 1980),
        '1981-1990': (1981, 1990),
        '1991-2000': (1991, 2000),
        '2001-2010': (2001, 2010),
        '2011-2020': (2011, 2020),
        '2021+': (2021, 9999)
    }

    @classmethod
    def categorize_construction_year(cls, year: int) -> str:
        for period, (start, end) in cls.CONSTRUCTION_PERIODS.items():
            if start <= year <= end:
                return period
        return "Unknown"

    def __init__(self, use_multiprocessing: bool = True):
        self.use_multiprocessing = use_multiprocessing

    def load_building_data(self, input_csv: str) -> Dict[int, List[Tuple]]:
        building_data = {}
        with open(input_csv, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                b_id = int(row[0])
                coords = (b_id, float(row[1]), float(row[2]), float(row[3]))
                building_data.setdefault(b_id, []).append(coords)
        return building_data

    def process_building(self, building_data: List[Tuple]) -> BuildingMetrics:
        b_id = building_data[0][0]
        x = [float(r[1]) for r in building_data]
        y = [float(r[2]) for r in building_data]
        angle, category, area, elongation = GeometryAnalyzer.calculate_metrics(x, y)
        return BuildingMetrics(b_id, angle, category, area, elongation)

    def process_buildings(self, building_data: Dict[int, List[Tuple]]) -> List[BuildingMetrics]:
        if self.use_multiprocessing:
            with multiprocessing.Pool() as pool:
                return pool.map(self.process_building, building_data.values())
        return [self.process_building(d) for d in building_data.values()]

    @staticmethod
    def digitize(data: np.ndarray, bins: int) -> np.ndarray:
        edges = np.quantile(data, np.linspace(0, 1, bins + 1))
        return np.digitize(data, edges[1:], right=True)

    def categorize_buildings(self, input_csv: str, misc_csv: str, output_csv: str):
        building_data = self.load_building_data(input_csv)
        results = self.process_buildings(building_data)

        df = pd.DataFrame({
            "building_id": [r.building_id for r in results],
            "orientation_angle": [r.orientation_angle for r in results],
            "orientation_category": [r.orientation_category for r in results],
            "area_value": [r.area for r in results],
            "area_bin": self.digitize(np.array([r.area for r in results]), 8),
            "elongation_value": [r.elongation for r in results],
            "elongation_bin": self.digitize(np.array([r.elongation for r in results]), 8),
        })

        misc = pd.read_csv(misc_csv)
        df["wwr_value"] = misc["GlazingRatio"].values
        df["wwr_bin"] = self.digitize(misc["GlazingRatio"].values, 5)

        years = misc["BuildingAge"].values
        df["building_age_value"] = years
        df["construction_period"] = [self.categorize_construction_year(y) for y in years]

        df.to_csv(output_csv, index=False)
        print(f"Saved {output_csv} with shape {df.shape}")


def __abs_path(path: str) -> str:
    return cesarp.common.abs_path(path, os.path.abspath(__file__))


def main(input_csv: str, misc_csv: str, output_csv: str):
    categorizer = BuildingCategorizer()
    categorizer.categorize_buildings(input_csv, misc_csv, output_csv)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    site_vertices_path = __abs_path("./data")
    output_dir = __abs_path("./results/example")
    os.makedirs(output_dir, exist_ok=True)

    input_csv_path = os.path.join(site_vertices_path, "Random_SiteVertices.csv")
    misc_csv_path = os.path.join(site_vertices_path, "Random_BuildingInformation.csv")
    output_csv_path = os.path.join(output_dir, "Categorized_Targets.csv")

    main(input_csv_path, misc_csv_path, output_csv_path)
