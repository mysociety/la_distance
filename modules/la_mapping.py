"""
Helpers to construct maps in altair of local authorities
"""

import json
from functools import lru_cache
from pathlib import Path

import altair as alt
import geopandas
from notebook_helper import *

from . import la

pd.options.mode.chained_assignment = None

top_level = Path(__file__).parent.parent
lower_tier = top_level / "resources" / "LocalAuthorities-lowertier.gpkg"


def gdf_to_data(gdf: geopandas.GeoDataFrame) -> alt.Data:
    """
    convert geopandas dataframe into the alt.data
    format expected by pandas
    For encoding purposes this moves all 'columns' into keys under
    properties.
    e.g. 'x' can be accessed as 'properties.x:Q'. 
    """
    choro_json = json.loads(gdf.to_json())
    data = alt.Data(values=choro_json['features'])
    return data


def gdf_chart(gdf: geopandas.GeoDataFrame) -> Chart:
    """
    start off an altair chart using a geodataframe
    """
    return Chart(gdf_to_data(gdf))


def get_label_and_offset(gdf, label, x, y, switch=False) -> pd.DataFrame:
    """
    split a specific label over multiple coordinates
    gets around lack of vega support for multi-line labems
    """
    if "&" in label:
        top, bottom = [x.strip() for x in label.split("&", 1)]
        bottom = "& " + bottom
    else:
        top, bottom = label.split(" ", 1)
    if switch:
        top, bottom = bottom, top
    item = gdf.loc[gdf["Group-labe"].str.contains(label)]
    item["Group-labe"] = top
    item["geometry"] = item["geometry"].translate(x, y)
    gdf.loc[gdf["Group-labe"].str.contains(label), "Group-labe"] = bottom
    return pd.concat([gdf, item])


@lru_cache
def lower_tier_base_layer() -> Chart:
    """
    construct the base layers of a uk lower_tier_map
    """

    # get background and group layers
    background_layer = geopandas.read_file(
        lower_tier, layer="7 Background").to_crs(epsg=4326)
    groups_layer = geopandas.read_file(
        lower_tier, layer="2 Groups").to_crs(epsg=4326)

    base = Chart(gdf_to_data(background_layer)).mark_geoshape(
        stroke='black',
        fill='lightgrey',
        strokeWidth=0
    ).encode(
    )

    groups = Chart(gdf_to_data(groups_layer)).mark_geoshape(
        stroke='black',
        fill=None,
        strokeWidth=1
    ).encode()

    base_uk = base + groups
    return base_uk


@lru_cache
def lower_tier_label_layer() -> Chart:
    """
    load and manipulate label layer
    """
    # altair doesn't like multi-line labels,
    # this moves two manually by creating new labels
    labels_layer = geopandas.read_file(
        lower_tier, layer="1 Group labels").to_crs(epsg=4326)

    labels_layer["Group-labe"] = labels_layer["Group-labe"].str.replace(
        "Leics &\\Rut.", "Leics & Rut.", regex=False)
    labels_layer = get_label_and_offset(
        labels_layer, "Leics & Rut.", -0.00001, +0.00001)
    labels_layer = get_label_and_offset(
        labels_layer, "Northern Ireland", -0.00001, +0.00001)
    labels_layer = get_label_and_offset(
        labels_layer, "East Yorks & Humber", 0, +0.00001)
    labels_layer = get_label_and_offset(
        labels_layer, "Lanarks. & Falkirk", 0, -0.00001, switch=True)
    labels_layer = get_label_and_offset(
        labels_layer, "West of England", 0, +0.00001)
    labels_layer = get_label_and_offset(
        labels_layer, "North Wales", 0, -0.000007, switch=True)
    # split layers into left and right to feed into altair seperately
    labels_layer['centroid_lon'] = labels_layer['geometry'].x
    labels_layer['centroid_lat'] = labels_layer['geometry'].y

    left_labels_mask = labels_layer["LabelPosit"] == "Left"
    right_labels_mask = labels_layer["LabelPosit"] == "Right"
    labels_layer_left = labels_layer.loc[left_labels_mask]
    labels_layer_right = labels_layer.loc[right_labels_mask]

    text_props = {
        "baseline": "middle",
        "font": "Source Sans Pro"
    }

    labels_left = Chart(gdf_to_data(labels_layer_left)
                            ).mark_text(align="left", **text_props)
    labels_right = Chart(gdf_to_data(labels_layer_right)).mark_text(
        align="right", **text_props)

    labels_left = labels_left.encode(
        longitude='properties.centroid_lon:Q',
        latitude='properties.centroid_lat:Q',
        text='properties.Group-labe:O')

    labels_right = labels_right.encode(
        longitude='properties.centroid_lon:Q',
        latitude='properties.centroid_lat:Q',
        text='properties.Group-labe:O')

    text_offset = -1
    labels_left_white = labels_left.mark_text(
        align="left", color="white", dx=text_offset, dy=text_offset, **text_props)
    labels_right_white = labels_right.mark_text(
        align="right", color="white", dx=text_offset, dy=text_offset, **text_props)

    labels = labels_left_white + labels_right_white
    labels += labels_left + labels_right
    return labels


def lower_tier_layer() -> geopandas.GeoDataFrame:
    """
    Get a geopandas dataframe preped with local authority codes to merge 
    with other data frames
    """
    gdp = geopandas.read_file(
        lower_tier, layer="6 LTLA-2021").to_crs(epsg=4326)
    gdp["local-authority-code"] = gdp["Lacode"].la.gss_to_code()
    gdp = gdp.set_index("local-authority-code")
    return gdp


def lower_tier_sandwich(tier_layer: Chart) -> Chart:
    """
    given the encoded tier layer, sandwich between base and labels
    """
    base_layer = lower_tier_base_layer()
    label_layer = lower_tier_label_layer()
    return base_layer + tier_layer + label_layer
