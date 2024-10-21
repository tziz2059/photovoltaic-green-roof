import pyvista as pv
import pyviewfactor as pvf
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import pvlib
import pytz
from datetime import datetime
import pandas as pd

###### INPUT DATA ###########

# Define the latitude and longitude 
latitude = 46.05  
longitude = 14.5

# Define the time zone
timezone = 'Europe/Ljubljana'

# Define the start and end dates in local time
start_date = datetime(2024, 8, 28, 8, 30, 00, tzinfo=pytz.timezone(timezone))
end_date = datetime(2024, 8, 29, 10, 32, 59, tzinfo=pytz.timezone(timezone))

# Define the geometry of the setup
# Green roof (lower rectangle)
GR_width=0.71 # green roof width (in direction South-North)
GR_length=0.71 # green roof length (in direction East-West)
GR_angle=0 # green roof angle

# Photovltaic module 1, assumed facing south
PV1_X=0 # photovoltaic module 1, offset in X based on green roof (lower left edge of both)
PV1_Y=0 # photovoltaic module 1, offset in Y based on green roof (lower left edge of both)
PV1_angle=25 # angle PV1
PV1_H=0.3 # distance from green roof (Z=0) to the lower edge of photovoltaic module
PV1_width=0.5 # width PV1
PV1_length=0.5 # length PV1

# Photovltaic module 2, assumed facing south
PV2_X=0 # photovoltaic module 1, offset in X based on green roof (lower left edge of both)
PV2_Y=0.6 # photovoltaic module 1, offset in Y based on green roof (lower left edge of both)
PV2_angle=25 # angle PV1
PV2_H=0.3 # distance from green roof (Z=0) to the UPPER edge of photovoltaic module
PV2_width=0.5 # width PV1
PV2_length=0.5 # length PV1



##################   Solar Position Calculation ##############

# Create a date range for each minute of the year
date_range = pd.date_range(start=start_date, end=end_date, freq='T')

# Convert local times to UTC
utc_times = date_range.tz_convert(pytz.UTC)

# Create a location object
location = pvlib.location.Location(latitude, longitude)

# Calculate solar positions
solar_positions = location.get_solarposition(utc_times)

# Add local time to the results DataFrame
solar_positions['local_time'] = date_range

# Convert to numpy arrays
altitudes = solar_positions['elevation'].values
azimuths = solar_positions['azimuth'].values
azimuths = azimuths
# Optionally, create a DataFrame to handle and save the data
df = pd.DataFrame({
    'Datetime': date_range,
    'Altitude': altitudes,
    'Azimuth': azimuths
})
df['Azimuth'] = 360 - df['Azimuth']
###############################################################################

# Function to create a rectangle with rotation around its left edge
def create_rectangle(lower_left, width, height, angle_deg, flip_normals=False):
    angle_rad = np.radians(angle_deg)
    corners = np.array([
        [0, height, 0],         # Upper-left corner
        [width, height, 0],     # Upper-right corner
        [width, 0, 0],          # Lower-right corner
        [0, 0, 0]               # Lower-left corner
    ])
    rotation_matrix = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    rotated_corners = (rotation_matrix @ corners.T).T
    translated_corners = rotated_corners + np.array(lower_left)
    faces = np.array([[4, 0, 1, 2, 3]])
    if flip_normals:
        faces = np.array([[4, 0, 3, 2, 1]])
    poly_data = pv.PolyData(translated_corners, faces)
    return poly_data

# Create rectangles
ground_rectangle = create_rectangle(lower_left=[0, 0, 0], width=GR_width, height=GR_length, angle_deg=GR_angle)
rect1 = create_rectangle(lower_left=[PV1_X, PV1_Y, PV1_H], width=PV1_length, height=PV1_width, angle_deg=-PV1_angle, flip_normals=True)
rect2 = create_rectangle(lower_left=[PV2_X, PV2_Y, PV2_H], width=PV2_length, height=PV1_width, angle_deg=-PV2_angle, flip_normals=True)

# Define the ground plane z-coordinate
ground_plane_z = ground_rectangle.points[:, 2].min()

# Calculate the area of the ground rectangle
ground_rectangle_area = Polygon(ground_rectangle.points[:, :2]).area

# Preliminary check for visibility
def check_visibility(source, target):
    # Calculate if target is visible from source
    if pvf.get_visibility(source, target):
        return True
    return False


# Calculate view factors
def calculate_view_factors(ground_rectangle, rect1, rect2):
    visibility1 = check_visibility(ground_rectangle, rect1)
    visibility2 = check_visibility(ground_rectangle, rect2)
    
    if visibility1:
        F1 = pvf.compute_viewfactor(rect1, ground_rectangle)
        print(f"View factor from lower rectangle to upper rectangle 1 = {F1}")
    else:
        print("Upper rectangle 1 is not visible from the lower rectangle.")
    
    if visibility2:
        F2 = pvf.compute_viewfactor(rect2, ground_rectangle)
        print(f"View factor from lower rectangle to upper rectangle 2 = {F2}")
    else:
        print("Upper rectangle 2 is not visible from the lower rectangle.")

# Calculate view factors
calculate_view_factors(ground_rectangle, rect1, rect2)


# Function to compute sun direction
def compute_sun_direction(altitude_deg, azimuth_deg):
    altitude_rad = np.radians(altitude_deg)
    azimuth_rad = np.radians(azimuth_deg)
    
    x = np.cos(altitude_rad) * np.cos(azimuth_rad)
    y = np.cos(altitude_rad) * np.sin(azimuth_rad)
    z = np.sin(altitude_rad)
    
    return np.array([x, y, z])

# Function to find intersection with the ground plane
def find_intersection(start_point, sun_direction, plane_z):
    start_z = start_point[2]
    direction_z = sun_direction[2]
    
    if direction_z == 0:
        raise ValueError("Sun direction is parallel to the ground plane.")
    
    t = (plane_z - start_z) / direction_z
    intersection_point = start_point + t * sun_direction
    return intersection_point

# Function to create rays from the rectangle to the ground plane
def create_sun_rays_to_ground(rectangle, sun_direction, ground_plane_z):
    vertices = rectangle.points
    intersections = [find_intersection(vertex, sun_direction, ground_plane_z) for vertex in vertices]
    return intersections

# Function to create a 2D polygon (shadow) from intersections
def create_shadow_polygon(intersections):
    if len(intersections) < 3:
        return None
    
    points_2d = intersections[:, :2]  # Use only x, y for 2D
    hull = ConvexHull(points_2d)
    hull_points = np.array([points_2d[vertex] for vertex in hull.vertices])
    
    # Close the polygon by repeating the first point
    hull_points = np.vstack([hull_points, hull_points[0]])

    return hull_points

# Function to create 2D polygon format
def get_polygon_2d(points):
    return [list(point) for point in points]

# Function to calculate shadow intersection area
def calculate_intersection_area(shadow_polygon, lower_rectangle):
    if shadow_polygon is None:
        return 0
    
    # Create shapely polygons
    shadow_poly = Polygon(shadow_polygon)
    lower_rect_poly = Polygon(lower_rectangle)

    # Calculate intersection
    intersection = shadow_poly.intersection(lower_rect_poly)
    
    if intersection.is_empty:
        return 0
    
    return intersection.area

# Function to calculate shading for a given minute
def calculate_shading_for_minute(datetime_index, altitude, azimuth):
    # If the sun is below the horizon, return 0 areas
    if altitude <= 0:
        return 0, 0
    
    sun_direction = compute_sun_direction(altitude, azimuth)
    
    # Calculate shadow intersections for each rectangle
    intersections_rect1 = create_sun_rays_to_ground(rect1, sun_direction, ground_plane_z)
    intersections_rect2 = create_sun_rays_to_ground(rect2, sun_direction, ground_plane_z)
    
    # Create shadow polygons
    shadow_polygon_rect1 = create_shadow_polygon(np.array(intersections_rect1))
    shadow_polygon_rect2 = create_shadow_polygon(np.array(intersections_rect2))
    
    # Calculate intersection areas
    lower_rectangle_2d = get_polygon_2d(ground_rectangle.points[:, :2])
    area_intersection_rect1 = calculate_intersection_area(shadow_polygon_rect1, lower_rectangle_2d)
    area_intersection_rect2 = calculate_intersection_area(shadow_polygon_rect2, lower_rectangle_2d)
    
    return area_intersection_rect1, area_intersection_rect2

# Prepare lists to store intersection areas and share of shading
intersection_areas_rect1 = []
intersection_areas_rect2 = []
share_of_shading = []

# Calculate shading for each minute
for index, row in df.iterrows():
    area_rect1, area_rect2 = calculate_shading_for_minute(row['Datetime'], row['Altitude'], row['Azimuth'])
    total_intersection_area = area_rect1 + area_rect2
    shading_share = total_intersection_area / ground_rectangle_area
    intersection_areas_rect1.append(area_rect1)
    intersection_areas_rect2.append(area_rect2)
    share_of_shading.append(shading_share)

# Add the intersection areas and shading share to the DataFrame
df['Intersection_Area_Rect1'] = intersection_areas_rect1
df['Intersection_Area_Rect2'] = intersection_areas_rect2
df['Shading_Share'] = share_of_shading

# Save the updated DataFrame to a CSV file
df.to_csv('solar_positions_with_shading_2024_minute.csv', index=False)

# Display some results
print(df.head())

#################################### VISUALIZATION ############################################

# Function to add intersection shaded area to plot
def add_intersection_shaded_area(p, shadow_polygon, lower_rectangle):
    if shadow_polygon is not None:
        # Convert shadow polygon to a 3D array with z=0
        shadow_polygon_3d = np.hstack([shadow_polygon, np.zeros((shadow_polygon.shape[0], 1))])
        shadow_polydata = pv.PolyData(shadow_polygon_3d)

        # Define faces for the polygon
        num_points = shadow_polygon_3d.shape[0]
        faces = np.hstack([[num_points] + list(range(num_points))])
        shadow_polydata.faces = faces
        
        # Add shadow polygon to plot
        p.add_mesh(shadow_polydata, color='grey', opacity=0.5, label='Shadow')
        
        # Calculate intersection with lower rectangle
        shadow_poly = Polygon(shadow_polygon)
        lower_rect_poly = Polygon(lower_rectangle)
        intersection = shadow_poly.intersection(lower_rect_poly)

        if not intersection.is_empty:
            if intersection.geom_type == 'Polygon':
                intersection_points = np.array(intersection.exterior.coords)
                intersection_3d = np.hstack([intersection_points, np.zeros((intersection_points.shape[0], 1))])
                intersection_polydata = pv.PolyData(intersection_3d)

                # Define faces for the intersection polygon
                num_intersection_points = intersection_3d.shape[0]
                intersection_faces = np.hstack([[num_intersection_points] + list(range(num_intersection_points))])
                intersection_polydata.faces = intersection_faces

                # Add intersection polygon to plot
                p.add_mesh(intersection_polydata, color='red', opacity=0.5, label='Intersection Area')

# Create the plotter object
p = pv.Plotter()

# Add ground rectangle and other rectangles
p.add_mesh(ground_rectangle, color='lightblue', opacity=0.5, show_edges=True, line_width=2, label='Ground Rectangle')
p.add_mesh(rect1, color='lightgreen', opacity=0.5, show_edges=True, line_width=2, label='Sloped Rectangle 1')
p.add_mesh(rect2, color='lightcoral', opacity=0.5, show_edges=True, line_width=2, label='Sloped Rectangle 2')

# Define sun direction example (use the desired minute here)
sun_direction_example = compute_sun_direction(df.iloc[0]['Altitude'], df.iloc[0]['Azimuth'])

# Calculate intersections and shadows
intersections_rect1 = create_sun_rays_to_ground(rect1, sun_direction_example, ground_plane_z)
intersections_rect2 = create_sun_rays_to_ground(rect2, sun_direction_example, ground_plane_z)

# Create shadow polygons
shadow_polygon_rect1 = create_shadow_polygon(np.array(intersections_rect1))
shadow_polygon_rect2 = create_shadow_polygon(np.array(intersections_rect2))

# Convert lower rectangle to 2D
lower_rectangle_2d = get_polygon_2d(ground_rectangle.points[:, :2])

# Add intersection shaded areas to plot
add_intersection_shaded_area(p, shadow_polygon_rect1, lower_rectangle_2d)
add_intersection_shaded_area(p, shadow_polygon_rect2, lower_rectangle_2d)

# Add rays
def create_ray_lines(rays):
    lines = []
    for start, end in rays:
        lines.append(pv.Line(start, end))
    return lines

ray_lines_rect1 = create_ray_lines(zip(rect1.points, intersections_rect1))
ray_lines_rect2 = create_ray_lines(zip(rect2.points, intersections_rect2))

for ray_line in ray_lines_rect1:
    p.add_mesh(ray_line, color='orange', line_width=1, label='Sun Rays from Rect1')

for ray_line in ray_lines_rect2:
    p.add_mesh(ray_line, color='purple', line_width=1, label='Sun Rays from Rect2')

# Add coordinate axes
p.add_axes(line_width=4, color='black', xlabel='X', ylabel='Y', zlabel='Z')

# Add a legend and show the plot
p.add_legend()
p.show()
