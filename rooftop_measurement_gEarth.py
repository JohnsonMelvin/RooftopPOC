import ee
import cv2
import numpy as np
import requests
from shapely.geometry import Polygon
from math import radians, pi, cos
from google.oauth2 import service_account
from io import BytesIO

SERVICE_ACCOUNT_KEY_FILE = "tandem-gmap-3ebb16bd8e55.json"
SCOPES = ["https://www.googleapis.com/auth/earthengine"]

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_KEY_FILE, scopes=SCOPES
)

ee.Initialize(credentials)


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


def fetch_satellite_image(lat, lon, zoom, size):
    point = ee.Geometry.Point([lon, lat])
    
    image = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(point) \
        .filterDate('2022-01-01', '2023-01-01') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE', False) \
        .first() \
        .select(['B4', 'B3', 'B2'])

    if image is None:
        raise Exception("No suitable images found for the given criteria")

    image_info = image.getInfo()
    print("Image info:", image_info)
    
    params = {
        "region": point.buffer(1000).bounds().getInfo()["coordinates"],
        "dimensions": f"{size}x{size}",
        "format": "png",
    }
    image_url = image.getThumbURL(params)

    print("Region:", params['region'])
    print("Image URL:", image_url)
 
    response = requests.get(image_url)
    if response.status_code == 200:
        with open("raw_satellite_image.png", "wb") as f:
            f.write(response.content)
        return BytesIO(response.content)
    else:
        raise Exception(f"Failed to fetch image from GEE. Status code: {response.status_code}")


def process_image(image_stream):
    image = cv2.imdecode(
        np.frombuffer(image_stream.getvalue(), np.uint8), cv2.IMREAD_COLOR
    )
    cv2.imwrite("debug_1_original.png", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug_2_gray.png", gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite("debug_3_blurred.png", blurred)

    edges = cv2.Canny(blurred, 10, 50) 
    cv2.imwrite("debug_4_edges.png", edges)

    dilated = cv2.dilate(edges, None, iterations=2)
    cv2.imwrite("debug_5_dilated.png", dilated)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blank = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(blank, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("debug_6_all_contours.png", blank)

    return contours, image.shape[0], image


def pixel_to_geo(pixel_x, pixel_y, center_lat, center_lon, zoom, img_size):
    world_coord = pixel_to_world(pixel_x, pixel_y, zoom, img_size)
    return world_to_geo(world_coord, center_lat, center_lon)


def pixel_to_world(pixel_x, pixel_y, zoom, img_size):
    scale = 256 * 2**zoom / img_size
    x = (pixel_x - img_size / 2) * scale
    y = (img_size / 2 - pixel_y) * scale
    return x, y


def world_to_geo(world_coord, center_lat, center_lon):
    x, y = world_coord
    lat = center_lat + (y / 6378137) * (180 / pi)
    lon = center_lon + (x / 6378137) * (180 / pi) / cos(radians(center_lat))
    return lat, lon


def measure_area_in_sqft(contours, center_lat, center_lon, zoom, img_size):
    buildings = []
    MIN_ROOFTOP_AREA = 50

    print(f"Total contours found: {len(contours)}")
    for i, contour in enumerate(contours):
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

        print(f"Contour {i}: Area = {area_in_sqft:.2f} sq. feet")

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
    image_stream = fetch_satellite_image(lat, lon, zoom, img_size)
    contours, img_size, image = process_image(image_stream)

    print(f"Number of contours detected: {len(contours)}")

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

    for contour in contours:
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
    cv2.imwrite(f"processed_image_{zipcode}.png", image)
    print(f"Processed image saved as processed_image_{zipcode}.png")


if __name__ == "__main__":
    GOOGLE_MAPS_API_KEY = "YOUR_KEY"
    zipcode = "11755"
    main(zipcode, GOOGLE_MAPS_API_KEY)
