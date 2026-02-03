import shapefile

# Coordinates from MongoDB document "Testing payment" (0.4376 acres)
# Format: (longitude, latitude) - WGS84
polygon_coords = [
    (74.81327442321802, 34.02555224449088),  # Point 0
    (74.81373755130355, 34.02566191157703),  # Point 1
    (74.8139038482608, 34.02541293744765),   # Point 2
    (74.81332091484, 34.02524399030139),     # Point 3
    (74.81327442321802, 34.02555224449088),  # Close polygon (same as Point 0)
]

# Create a shapefile writer for polygon type
w = shapefile.Writer("polygon")
w.field('id', 'N')  # Add an ID field

# Add the polygon
w.poly([polygon_coords])
w.record(1)

# Close the writer
w.close()

# Create the .prj file for WGS84 coordinate system
prj_content = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
with open("polygon.prj", "w") as prj:
    prj.write(prj_content)

print("Shapefile created successfully with NEW 250m x 350m coordinates!")
print("Files created: polygon.shp, polygon.shx, polygon.dbf, polygon.prj")
