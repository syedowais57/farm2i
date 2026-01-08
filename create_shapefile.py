import shapefile

# Coordinates for the Polygon (LineString - lon, lat format)
polygon_coords = [
    (74.8131613062597, 34.02585101785077),
    (74.81330082003586, 34.02527131859058),
    (74.81367976961539, 34.02529006903366),
    (74.81391354945507, 34.026002582773245),
    (74.8131575356179, 34.025849455324746),
    (74.8131613062597, 34.02585101785077)  # Close the polygon
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

print("Shapefile created successfully with NEW coordinates!")
print("Files created: polygon.shp, polygon.shx, polygon.dbf, polygon.prj")
