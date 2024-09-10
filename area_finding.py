import cv2
import numpy as np
from shapely.geometry import Polygon
import requests
from io import BytesIO
from math import radians, log, tan, pi, atan2, sin, cos, exp, sqrt, atan


def get_coordinates(zipcode, api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zipcode}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["status"] == "OK":
            location = data["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
        else:
            raise ValueError(f"Geocoding API error: {data['status']}")
    else:
        raise Exception("Failed to fetch coordinates")


def fetch_satellite_image(lat, lon, zoom, size, api_key):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size}x{size}&maptype=satellite&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        raise Exception("Failed to fetch image")


def process_image(image_stream):
    image = cv2.imdecode(
        np.frombuffer(image_stream.getvalue(), np.uint8), cv2.IMREAD_COLOR
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, image.shape[0]


def pixel_to_geo(pixel_x, pixel_y, center_lat, center_lon, zoom, img_size):
    world_coord = pixel_to_world(pixel_x, pixel_y, zoom, img_size)
    return world_to_geo(world_coord, center_lat, center_lon)


def pixel_to_world(pixel_x, pixel_y, zoom, img_size):
    scale = 256 * 2**zoom / img_size
    x = (pixel_x - img_size / 2) / scale
    y = (img_size / 2 - pixel_y) / scale
    return x, y


def world_to_geo(world_coord, center_lat, center_lon):
    x, y = world_coord
    lat = center_lat + (y / 6378137) * (180 / pi)
    lon = center_lon + (x / 6378137) * (180 / pi) / cos(radians(center_lat))
    return lat, lon


def measure_area_in_sqft(contours, center_lat, center_lon, zoom, img_size):
    buildings = []
    MIN_ROOFTOP_AREA = 100

    for contour in contours:
        if len(contour) < 4:
            continue

        geo_polygon = []
        for point in contour:
            lat, lon = pixel_to_geo(
                point[0][0], point[0][1], center_lat, center_lon, zoom, img_size
            )
            geo_polygon.append((lon, lat))

        polygon = Polygon(geo_polygon)
        area_in_sqm = polygon.area * (111319.9 * cos(radians(center_lat))) ** 2
        area_in_sqft = area_in_sqm * 10.764

        if area_in_sqft >= MIN_ROOFTOP_AREA:
            centroid = polygon.centroid
            buildings.append(
                {"coordinates": (centroid.y, centroid.x), "area_sqft": area_in_sqft}
            )

    return buildings


def main(zipcode, api_key):
    lat, lon = get_coordinates(zipcode, api_key)
    zoom = 0 
    img_size = 640
    image_stream = fetch_satellite_image(lat, lon, zoom, img_size, api_key)
    contours, img_size = process_image(image_stream)
    buildings = measure_area_in_sqft(contours, lat, lon, zoom, img_size)

    print(f"Zipcode: {zipcode}")
    print(f"Center coordinates: ({lat}, {lon})")
    print(f"Number of buildings detected: {len(buildings)}")
    print()

    for i, building in enumerate(buildings):
        print(f"Building {i + 1}:")
        print(f"  Coordinates: {building['coordinates']}")
        print(f"  Rooftop Area: {building['area_sqft']:.2f} sq. feet")
        print()


if __name__ == "__main__":
    GOOGLE_MAPS_API_KEY = "YOUR_KEY"
    zipcode = "600032"
    main(zipcode, GOOGLE_MAPS_API_KEY)
