# FarmVision Sentinel-2 Indices API

This project provide a high-performance FastAPI service that uses Google Earth Engine (GEE) to calculate, visualize, and analyze agricultural data.

---

## 📊 Vegetation & Soil Indices Reference

| Category | Index | Full Name | Formula Details | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Basic Vegetation** | **NDVI** | Normalized Difference Vegetation Index | `(B8 - B4) / (B8 + B4)` | Measuring overall plant health and biomass density. |
| | **GNDVI** | Green NDVI | `(B8 - B3) / (B8 + B3)` | Sensitive to chlorophyll content in mature plants. |
| | **EVI** | Enhanced Vegetation Index | `2.5 * ((B8 - B4) / (abs_formula))` | Monitoring biomass in high-density canopy areas. |
| | **MSAVI** | Modified Soil Adjusted Veg Index | `Soil Adjusted Logic` | Crop health in early stages (low biomass/bare soil). |
| **Advanced Stress** | **NDMI** | Normalized Difference Moisture Index | `(B8 - B11) / (B8 + B11)` | Detecting water stress and leaf hydration levels. |
| | **NDRE** | Normalized Difference Red Edge | `(B8 - B5) / (B8 + B5)` | Early detection of plant stress and vigor changes. |
| | **PSRI** | Plant Senescence Reflectance Index | `(B4 - B2) / B6` | Monitoring crop maturity, ripening, and dry-down. |
| **Nutrient Proxy** | **RECI** | Red Edge Chlorophyll Index | `(B8 / B5) - 1` | Direct proxy for total leaf chlorophyll concentration. |
| | **MCARI** | Modified Chlorophyll Absorption | `Reflectance ratio` | Chlorophyll assessment (resistant to soil background). |
| | **OC** | Organic Carbon Proxy | `3.591 * NDVI` | Estimate of soil organic matter and carbon levels. |
| | **N_Index** | Nitrogen Index | `1.25 * NDRE` | Spatial mapping of Nitrogen availability. |
| | **P_Index** | Phosphorus Index | `0.48 * OC` | Spatial mapping of Phosphorus availability. |
| | **K_Index** | Potassium Index | `0.78 * NDVI` | Spatial mapping of Potassium availability. |
| **Soil Health** | **pH_Index** | Soil pH Index | `Ratio calculation` | Proxy for soil acidity/alkalinity trends. |

---

## 🎨 Color Interpretation Master Guide

| Index Type | Lowest Values | Mid Values | Highest Values |
| :--- | :--- | :--- | :--- |
| **Biomass (NDVI/EVI)** | 🔴 Bare Soil / Stressed | 🟡 Sparse Growth | 🟢 High Vigor / Dense Canopy |
| **Nitrogen (N)** | 🔴 Low N (Deficiency) | 🟡 Average N | 🟢 High N (Healthy) |
| **Phosphorus (P)** | ❄️ Light Blue (Low P) | 🔵 Medium Blue | 🟦 Dark Blue (High P) |
| **Potassium (K)** | 🌸 Light Purple (Low K) | 🟣 Medium Purple | 🍇 Deep Purple (High K) |
| **pH (Litmus)** | 🔴 Acidic (pH < 6) | 🟢 Neutral (pH 7) | 🔵 Alkaline (pH > 8) |
| **Moisture (NDMI)** | 🟫 Dry / Dehydrated | 🟢 Normal Hydration | 🟦 High Water Content |
| **Organic Carbon** | 🟫 Low Carbon (Sandy) | 🟠 Medium Carbon | 🟢 High Carbon (Organic) |

---

## 🛠️ API & Technical Specifications

### 📡 Data Source
- **Sentinel-2 L2A**: Atmospherically corrected surface reflectance data.
- **Imagery Provider**: Google Earth Engine (GEE).
- **Cloud Masking**: Intelligent Scene Classification (SCL) based masking.

### � REST Endpoint
`POST /api/v1/calculate-indices`

**Sample Body:**
```json
{
  "coordinates": [
    {"lat": 34.02, "lng": 74.81},
    {"lat": 34.02, "lng": 74.82},
    {"lat": 34.01, "lng": 74.81}
  ],
  "start_date": "2025-06-01",
  "end_date": "2025-06-15"
}
```

### 📦 Output Formats
1. **JSON Data**: Returns numeric median values for every requested index across the time series.
2. **Visual Maps**: Generates high-resolution, clipped PNG images with normalized color scales for direct UI embedding.
