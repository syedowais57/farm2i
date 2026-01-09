import shapefile

# New coordinates provided by the user (LineString format, approx 250m x 350m)
polygon_coords = [
    (74.86768426089122, 34.08859514192747),
    (74.8682098349648, 34.089159625234004),
    (74.87026285869177, 34.08719411890472),
    (74.86885037836731, 34.08608552600538),
    (74.86754465527858, 34.08753417516738),
    (74.86768426089057, 34.08859415072389),
    (74.86768426089122, 34.08859514192747) # Ensure fully closed
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
