"""Public research data source connectors for structural engineering.

Provides metadata and download helpers for common open databases.
Actual data download requires network access and may need user authentication.
"""

import os
import json

SOURCES = {
    "tpu": {
        "name": "TPU Aerodynamic Database",
        "organization": "Tokyo Polytechnic University",
        "url": "https://db.wind.arch.t-kougei.ac.jp/",
        "description": "Wind tunnel measurement data for low-rise and high-rise building models. "
                       "Includes mean and fluctuating pressure coefficients, time series data.",
        "data_types": ["wind_pressure_coefficients", "time_series", "mean_values", "peak_values"],
        "building_types": ["flat_roof", "gable_roof", "hip_roof", "high_rise"],
        "access": "free",
        "format": "CSV, online query",
        "fields": ["wind_engineering", "structural_engineering"],
    },
    "peer_nga": {
        "name": "PEER NGA-West2 Ground Motion Database",
        "organization": "Pacific Earthquake Engineering Research Center",
        "url": "https://ngawest2.berkeley.edu/",
        "description": "Comprehensive database of recorded ground motions from shallow crustal "
                       "earthquakes in active tectonic regions worldwide.",
        "data_types": ["acceleration_time_series", "response_spectra", "ground_motion_parameters"],
        "access": "free (registration required)",
        "format": "AT2, CSV",
        "fields": ["earthquake_engineering", "seismic_design"],
    },
    "cesmd": {
        "name": "Center for Engineering Strong Motion Data",
        "organization": "USGS / CGS",
        "url": "https://www.strongmotioncenter.org/",
        "description": "Instrumented building and bridge response records from earthquakes. "
                       "Includes structural response at multiple floor levels.",
        "data_types": ["structural_response", "floor_acceleration", "building_records"],
        "access": "free",
        "format": "V1/V2 COSMOS format",
        "fields": ["earthquake_engineering", "structural_health_monitoring"],
    },
    "nist_aero": {
        "name": "NIST Aerodynamic Database",
        "organization": "National Institute of Standards and Technology",
        "url": "https://www.nist.gov/el/materials-and-structural-systems-division-73100/nist-aerodynamic-database",
        "description": "Wind pressure data for low-rise buildings from UWO boundary layer wind tunnel.",
        "data_types": ["wind_pressure_coefficients", "time_series"],
        "access": "free",
        "format": "Binary, MATLAB",
        "fields": ["wind_engineering"],
    },
    "designsafe": {
        "name": "DesignSafe-CI (NHERI)",
        "organization": "Natural Hazards Engineering Research Infrastructure",
        "url": "https://www.designsafe-ci.org/",
        "description": "Shared data repository for natural hazards research. Includes wind, "
                       "earthquake, storm surge, and tsunami datasets.",
        "data_types": ["multi_hazard", "experimental", "simulation", "field_data"],
        "access": "free (registration required)",
        "format": "Various",
        "fields": ["wind_engineering", "earthquake_engineering", "coastal_engineering"],
    },
    "knet": {
        "name": "K-NET / KiK-net",
        "organization": "National Research Institute for Earth Science and Disaster Resilience (NIED)",
        "url": "https://www.kyoshin.bosai.go.jp/",
        "description": "Japanese strong-motion seismograph network. Over 1,700 stations "
                       "recording acceleration data from earthquakes across Japan.",
        "data_types": ["acceleration_time_series", "ground_motion_parameters"],
        "access": "free (registration required)",
        "format": "ASCII, WIN format",
        "fields": ["earthquake_engineering", "seismology"],
    },
    "cosmos": {
        "name": "COSMOS Virtual Data Center",
        "organization": "Consortium of Organizations for Strong-Motion Observation Systems",
        "url": "https://www.strongmotioncenter.org/vdc/",
        "description": "Global collection of processed strong-motion records from international networks.",
        "data_types": ["acceleration_time_series", "response_spectra"],
        "access": "free",
        "format": "COSMOS V1/V2",
        "fields": ["earthquake_engineering"],
    },
}


def list_sources(field: str = None) -> list:
    """List available data sources, optionally filtered by field.

    Parameters
    ----------
    field : str, optional
        Filter by field (e.g., "wind_engineering", "earthquake_engineering").

    Returns
    -------
    list[dict]
        Source metadata dicts.
    """
    if field is None:
        return list(SOURCES.values())
    return [s for s in SOURCES.values() if field in s.get("fields", [])]


def get_source(key: str) -> dict:
    """Get metadata for a specific data source."""
    if key not in SOURCES:
        available = ", ".join(SOURCES.keys())
        raise KeyError(f"Unknown source '{key}'. Available: {available}")
    return SOURCES[key]


def suggest_sources(topic: str) -> list:
    """Suggest relevant data sources for a research topic.

    Parameters
    ----------
    topic : str
        Research topic description.

    Returns
    -------
    list[dict]
        Matching sources with relevance scores.
    """
    topic_lower = topic.lower()
    results = []

    keywords_map = {
        "tpu": ["wind", "pressure", "aerodynamic", "low-rise", "roof", "wind tunnel", "tpu"],
        "peer_nga": ["earthquake", "seismic", "ground motion", "spectral", "acceleration", "shake"],
        "cesmd": ["instrumented", "building response", "floor acceleration", "structural health"],
        "nist_aero": ["wind", "pressure", "nist", "low-rise", "aerodynamic"],
        "designsafe": ["hazard", "wind", "earthquake", "tsunami", "storm", "experimental"],
        "knet": ["japan", "earthquake", "seismic", "kik-net", "strong motion"],
        "cosmos": ["strong motion", "earthquake", "global", "acceleration"],
    }

    for key, keywords in keywords_map.items():
        score = sum(1 for kw in keywords if kw in topic_lower)
        if score > 0:
            source = dict(SOURCES[key])
            source["relevance_score"] = score
            source["key"] = key
            results.append(source)

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results
