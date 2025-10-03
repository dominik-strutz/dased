import unittest
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
import os
import tempfile
import warnings

from dased.layout.layout import DASLayout, DASLayoutGeographic


class TestDASLayout(unittest.TestCase):
    """Test cases for the DASLayout class."""

    def setUp(self):
        """Set up test fixtures."""
        # Define simple straight line knots for testing
        self.knots_line = np.array([[0, 0], [100, 0]])
        # Define a square layout for testing
        self.knots_square = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
        # Define spacing
        self.spacing = 10.0
        # Suppress expected warnings during tests
        warnings.filterwarnings("ignore", category=UserWarning)
    
    def test_init_basic(self):
        """Test basic initialization of DASLayout."""
        layout = DASLayout(self.knots_line, spacing=self.spacing)
        self.assertEqual(layout.n_channels, 9)  # Expect 9 channels at 10m spacing across 100m
        self.assertIsNone(layout.signal_decay)
        self.assertEqual(layout.data_types, ["P", "S"])
        self.assertAlmostEqual(layout.cable_length, 100.0, places=1)
    
    def test_init_with_field_properties(self):
        """Test initialization with field properties."""
        # Create a constant elevation
        field_props = {"elevation": 50.0}
        layout = DASLayout(self.knots_line, spacing=self.spacing, field_properties=field_props)
        
        # Check that elevation was set properly
        self.assertEqual(layout.field_properties["elevation"], 50.0)
        
        # Check that all channel Z coordinates use this elevation
        self.assertTrue(np.allclose(layout.channel_locations[:, 2], 50.0))
    
    def test_init_with_xarray_elevation(self):
        """Test initialization with xarray DataArray for elevation."""
        # Create a simple elevation grid
        x = np.linspace(-50, 150, 20)
        y = np.linspace(-50, 150, 20)
        xx, yy = np.meshgrid(x, y)
        # Create a sloped elevation: z = 0.1*x + 0.2*y
        elev = 0.1 * xx + 0.2 * yy
        elevation_da = xr.DataArray(
            elev, 
            coords=[("y", y), ("x", x)],
            name="elevation"
        )
        
        field_props = {"elevation": elevation_da}
        layout = DASLayout(self.knots_line, spacing=self.spacing, field_properties=field_props)
        
        # Test that elevation increases along x-axis (since our line is along x)
        self.assertTrue(layout.channel_locations[-1, 2] > layout.channel_locations[0, 2])
        
        # Check the elevation calculation for the first channel (at x=5, y=0)
        expected_elevation = 0.1 * 5 + 0.2 * 0  # z = 0.1*x + 0.2*y
        self.assertAlmostEqual(layout.channel_locations[0, 2], expected_elevation, places=1)
    
    def test_init_with_callable_elevation(self):
        """Test initialization with callable for elevation."""
        # Define a simple elevation function: z = 0.2*x - 0.1*y
        def elevation_func(x, y):
            return 0.2 * np.asarray(x) - 0.1 * np.asarray(y)
        
        field_props = {"elevation": elevation_func}
        layout = DASLayout(self.knots_line, spacing=self.spacing, field_properties=field_props)
        
        # Check the elevation for the first channel (at x=5, y=0)
        expected_elevation = 0.2 * 5 - 0.1 * 0
        self.assertAlmostEqual(layout.channel_locations[0, 2], expected_elevation, places=1)
    
    def test_signal_decay(self):
        """Test signal decay calculation."""
        decay_rate = 2.0  # 2.0 dB/km
        layout = DASLayout(
            self.knots_line, spacing=self.spacing, signal_decay=decay_rate
        )
        
        # Get the GeoDataFrame that includes attenuation data
        gdf = layout.get_gdf()
        
        # First channel should have no attenuation (distance=0)
        self.assertAlmostEqual(gdf.iloc[0]["attenuation"], 1.0)
        
        # Last channel should have some attenuation
        # For a 100m cable with 2.0 dB/km decay, the attenuation at the end should be:
        # 10^(-(2.0/20) * 0.1) ≈ 0.9772
        expected_attenuation = 10 ** (-(decay_rate / 20.0) * (100.0 / 1000.0))
        self.assertAlmostEqual(gdf.iloc[-1]["attenuation"], expected_attenuation, places=4)
    
    def test_wrap_around(self):
        """Test wrap_around functionality."""
        # Without wrap-around (default)
        layout1 = DASLayout(self.knots_square, spacing=self.spacing)
        
        # With wrap-around
        layout2 = DASLayout(self.knots_square, spacing=self.spacing, wrap_around=True)
        
        # With wrap-around, the cable should form a complete loop
        # So the cable length should be longer
        self.assertTrue(layout2.cable_length > layout1.cable_length)
        
        # For a square with wrap-around, we expect a perimeter of 400 units
        self.assertAlmostEqual(layout2.cable_length, 400.0, places=1)
    
    def test_get_shapely(self):
        """Test conversion to Shapely LineString."""
        layout = DASLayout(self.knots_line, spacing=self.spacing)
        line = layout.get_shapely()
        
        self.assertIsInstance(line, LineString)
        self.assertEqual(len(line.coords), layout.n_channels)
        
        # The length of the LineString should be approximately the cable length
        self.assertAlmostEqual(line.length, layout.cable_length, places=1)
    
    def test_get_gdf(self):
        """Test conversion to GeoDataFrame."""
        layout = DASLayout(self.knots_line, spacing=self.spacing)
        gdf = layout.get_gdf()
        
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        # Two data types (P and S) for each channel
        self.assertEqual(len(gdf), layout.n_channels * len(layout.data_types))
        
        # Check that all columns are present
        expected_columns = [
            "channel_id", "data_type", "u_x", "u_y", "u_z", "distance", 
            "attenuation", "geometry", "z"
        ]
        for col in expected_columns:
            self.assertIn(col, gdf.columns)
            
        # Check that points are correct geometry type
        self.assertIsInstance(gdf.iloc[0]["geometry"], Point)
    
    def test_plot(self):
        """Test plotting functionality."""
        layout = DASLayout(self.knots_square, spacing=self.spacing)
        
        # Test basic plotting
        fig, ax = layout.plot(show_knots=True)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Test with line style
        fig, ax = layout.plot(plot_style="line", color="red")
        plt.close(fig)
    
    def test_aperture(self):
        """Test aperture calculation."""
        layout = DASLayout(self.knots_line, spacing=self.spacing)
        
        # For a straight line of length 100, the aperture is the full length
        expected_aperture = 100.0
        self.assertAlmostEqual(layout.aperture(), expected_aperture, places=1)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test with negative spacing
        with self.assertRaises(ValueError):
            DASLayout(self.knots_line, spacing=-1.0)
            
        # Test with empty knots
        with self.assertRaises(ValueError):
            DASLayout(np.array([]), spacing=self.spacing)
            
        # Test with improperly shaped knots
        with self.assertRaises(ValueError):
            DASLayout(np.array([1, 2, 3]), spacing=self.spacing)


class TestDASLayoutGeographic(unittest.TestCase):
    """Test cases for the DASLayoutGeographic class."""

    def setUp(self):
        """Set up test fixtures."""
        # Define simple straight line in geographic coordinates (lon/lat)
        # These are arbitrary coordinates in the Western US
        self.geo_knots_line = np.array([[-122.0, 37.0], [-121.9, 37.0]])
        self.spacing = 100.0  # 100 meters spacing
        # Suppress expected warnings during tests
        warnings.filterwarnings("ignore", category=UserWarning)
    
    def test_init_basic(self):
        """Test basic initialization of DASLayoutGeographic."""
        layout = DASLayoutGeographic(
            self.geo_knots_line, spacing=self.spacing
        )
        
        # Check that UTM coordinates were calculated correctly
        self.assertEqual(layout.utm_zone, 10)  # Western US should be UTM zone 10
        self.assertEqual(layout.utm_hemisphere, 'N')  # Northern hemisphere
        
        # Check that we have a reasonable number of channels
        self.assertGreater(layout.n_channels, 5)
        
        # Check that cable_length is reasonable for the geographic distance
        # ~8-9km for 0.1° longitude at 37°N
        self.assertGreater(layout.cable_length, 7000)
        self.assertLess(layout.cable_length, 10000)
    
    def test_get_gdf_with_crs(self):
        """Test GeoDataFrame conversion with CRS handling."""
        layout = DASLayoutGeographic(
            self.geo_knots_line, spacing=self.spacing
        )
        
        # Get GDF in default geographic coordinates
        gdf_geo = layout.get_gdf()
        self.assertEqual(gdf_geo.crs.to_epsg(), 4326)  # Default is EPSG:4326 (WGS84)
        
        # Get GDF in UTM coordinates
        gdf_utm = layout.get_gdf(output_crs=layout.crs_utm)
        self.assertEqual(gdf_utm.crs.to_epsg(), layout.crs_utm.to_epsg())
        
        # Get GDF in custom CRS (Web Mercator)
        gdf_web = layout.get_gdf(output_crs="EPSG:3857")
        self.assertEqual(gdf_web.crs.to_epsg(), 3857)
    
    def test_geographic_elevation(self):
        """Test handling of geographic elevation data."""
        # Create a test grid in geographic coordinates
        lon = np.linspace(-122.1, -121.8, 20)
        lat = np.linspace(36.9, 37.1, 20)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Simple elevation model: elevation increases to the North and East
        elev_data = 100 + 1000 * (lon_grid + 122.0) + 2000 * (lat_grid - 37.0)
        
        # Create an xarray
        elev_da = xr.DataArray(
            elev_data,
            coords=[("latitude", lat), ("longitude", lon)],
            name="elevation"
        )
        
        # Create layout with geographic elevation data
        field_props = {"elevation": elev_da}
        layout = DASLayoutGeographic(
            self.geo_knots_line, 
            spacing=self.spacing,
            field_properties=field_props
        )
        
        # Elevation should increase from West to East in our example
        self.assertLess(layout.channel_locations[0, 2], layout.channel_locations[-1, 2])
    
    def test_geographic_callable(self):
        """Test handling of callable property with geographic inputs."""
        # Function that takes lon/lat and returns a value
        def elevation_func(lon, lat):
            return 100 + 1000 * (np.asarray(lon) + 122.0) + 2000 * (np.asarray(lat) - 37.0)
        
        field_props = {"elevation": elevation_func}
        layout = DASLayoutGeographic(
            self.geo_knots_line, 
            spacing=self.spacing,
            field_properties=field_props
        )
        
        # Elevation should increase from West to East in our example
        self.assertLess(layout.channel_locations[0, 2], layout.channel_locations[-1, 2])
    
    def test_invalid_inputs(self):
        """Test handling of invalid geographic inputs."""
        # Test with empty knots
        with self.assertRaises(ValueError):
            DASLayoutGeographic(np.array([]), spacing=self.spacing)
            
        # Test with improperly shaped knots
        with self.assertRaises(ValueError):
            DASLayoutGeographic(np.array([1, 2, 3]), spacing=self.spacing)


if __name__ == '__main__':
    unittest.main()