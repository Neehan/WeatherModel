NUM_YEARS = 39

WEATHER_PARAMS = {
    "Temperature at 2 Meters (C)": "T2M",
    "Temperature at 2 Meters Maximum (C)": "T2M_MAX",
    "Temperature at 2 Meters Minimum (C)": "T2M_MIN",
    "Wind Direction at 2 Meters (Degrees)": "WD2M",
    "Wind Speed at 2 Meters (m/s)": "WS2M",
    "Surface Pressure (kPa)": "PS",
    "Specific Humidity at 2 Meters (g/Kg)": "QV2M",
    "Precipitation Corrected (mm/day)": "PRECTOTCORR",
    "All Sky Surface Shortwave Downward Irradiance (MJ/m^2/day)": "ALLSKY_SFC_SW_DWN",
    "Evapotranspiration Energy Flux (MJ/m^2/day)": "EVPTRNS",
    "Profile Soil (the layer from the surface down to the bedrock) Moisture (0 to 1)": "GWETPROF",
    "Snow Depth (cm)": "SNODP",
    "Dew/Frost Point at 2 Meters (C)": "T2MDEW",
    "Cloud Amount (%)": "CLOUD_AMT",
    "Evaporation Land (kg/m^2/s * 10^6)": "EVLAND",
    "Wet Bulb Temperature at 2 Meters (C)": "T2MWET",
    "Land Snowcover Fraction (0 to 1)": "FRSNO",
    "All Sky Surface Longwave Downward Irradiance (MJ/m^2/day)": "ALLSKY_SFC_LW_DWN",
    "All Sky Surface PAR Total (MJ/m^2/day)": "ALLSKY_SFC_PAR_TOT",
    "All Sky Surface Albedo (0 to 1)": "ALLSKY_SRF_ALB",
    "Precipitable Water (cm)": "PW",
    "Surface Roughness (m)": "Z0M",
    "Surface Air Density (kg/m^3) ": "RHOA",
    "Relative Humidity at 2 Meters (%)": "RH2M",
    "Cooling Degree Days Above 18.3 C": "CDD18_3",
    "Heating Degree Days Below 18.3 C": "HDD18_3",
    "Total Column Ozone (Dobson units)": "TO3",
    "Aerosol Optical Depth 55": "AOD_55",
    "Evapotranspiration": "ET0",
    "Vapor Pressure": "VAP",
    "Vapor Pressure Deficit": "VPD",
}

WEATHER_PARAMS = list(WEATHER_PARAMS.values())
