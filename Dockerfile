# Use the official AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Install minimal system dependencies
RUN yum update -y && \
    yum install -y \
    mesa-libGL \
    && yum clean all

# Upgrade pip and build tools
RUN pip install --upgrade pip setuptools wheel

# Install ALL dependencies from binary wheels only
# This avoids needing any system-level build tools (GCC, GDAL, PROJ)
# Geospatial wheels bundle their own native libraries
RUN pip install --no-cache-dir --only-binary :all: \
    "numpy>=1.24.0,<2.0" \
    "pandas>=2.0.0" \
    "geopandas>=0.14.0,<1.0.0" \
    "rasterio>=1.3.0,<1.4.0" \
    "fiona>=1.9.0,<1.10.0" \
    "pyproj>=3.5.0,<3.7.0" \
    "shapely>=2.0.0" \
    "rasterstats>=0.19.0" \
    "scipy>=1.11.0,<1.14.0" \
    "matplotlib>=3.7.0,<3.10.0" \
    "requests>=2.31.0" \
    "earthengine-api>=1.0.0" \
    "fastapi>=0.109.0" \
    "uvicorn>=0.27.0" \
    "python-multipart>=0.0.6" \
    "gunicorn>=21.2.0" \
    "python-dotenv>=1.0.0" \
    "mangum>=0.17.0" \
    "awslambdaric>=2.0.0"

# Copy all application code
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to the Lambda handler
CMD [ "app.main.handler" ]
