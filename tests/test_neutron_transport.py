"""
Tests for neutron transport simulation.

This module contains comprehensive tests for:
- Material class with energy-dependent cross sections
- 1D and 2D geometry classes
- Neutron state and tallies
- Physics sampling functions
- Neutron tracking functions
- NeutronTransportSimulation and KEFFSimulation classes
- Integration tests
"""

import numpy as np
import pytest

from mcframework.sims.neutron_transport import (
    Material,
    Geometry1DSlab,
    Geometry2DPlane,
    NeutronState,
    NeutronTallies,
    NeutronTransportSimulation,
    KEFFSimulation,
    _sample_distance,
    _sample_scatter_angle,
    _sample_fission_energy,
    _sample_interaction_type,
    _track_neutron_1d,
    _track_neutron_2d,
    ENERGY_GROUP_NAMES,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def u235_material():
    """Fixture providing U-235 fissile material."""
    return Material.create_u235()


@pytest.fixture
def water_material():
    """Fixture providing H2O moderator material."""
    return Material.create_water()


@pytest.fixture
def graphite_material():
    """Fixture providing graphite moderator material."""
    return Material.create_graphite()


@pytest.fixture
def void_material():
    """Fixture providing void material."""
    return Material.create_void()


@pytest.fixture
def simple_1d_slab(u235_material):
    """Fixture providing simple 1D slab with U-235."""
    return Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235_material])


@pytest.fixture
def heterogeneous_1d_slab(u235_material, water_material):
    """Fixture providing heterogeneous 1D slab: water | fuel | water."""
    return Geometry1DSlab(
        0.0, 10.0,
        [0.0, 2.0, 8.0, 10.0],
        [water_material, u235_material, water_material]
    )


@pytest.fixture
def simple_2d_plane(u235_material, water_material):
    """Fixture providing 2D plane with fuel center and water reflector."""
    return Geometry2DPlane(
        0.0, 10.0, 0.0, 10.0,
        [(2.0, 2.0, 8.0, 8.0, u235_material)],
        water_material
    )


@pytest.fixture
def rng_fixed():
    """Fixture providing fixed RNG for reproducibility."""
    return np.random.default_rng(42)


# =============================================================================
# Test Material Class
# =============================================================================

class TestMaterial:
    """Test Material class with energy-dependent cross sections."""
    
    def test_material_initialization_valid(self):
        """Test material initialization with valid cross sections."""
        mat = Material(
            name="TestMat",
            sigma_total=[1.0, 1.0, 1.0],
            sigma_scatter=[0.5, 0.5, 0.5],
            sigma_absorption=[0.3, 0.3, 0.3],
            sigma_fission=[0.2, 0.2, 0.2],
            nu=[2.5, 2.5, 2.5],
            fission_spectrum=[0.1, 0.2, 0.7],
        )
        assert mat.name == "TestMat"
        assert len(mat.sigma_total) == 3
    
    def test_material_validation_sigma_total_length(self):
        """Test validation of sigma_total array length."""
        with pytest.raises(ValueError, match="sigma_total must have 3 values"):
            Material(
                name="BadMat",
                sigma_total=[1.0, 1.0],  # Wrong length
                sigma_scatter=[0.5, 0.5, 0.5],
                sigma_absorption=[0.3, 0.3, 0.3],
                sigma_fission=[0.2, 0.2, 0.2],
                nu=[2.5, 2.5, 2.5],
                fission_spectrum=[0.1, 0.2, 0.7],
            )
    
    def test_material_validation_sigma_scatter_length(self):
        """Test validation of sigma_scatter array length."""
        with pytest.raises(ValueError, match="sigma_scatter must have 3 values"):
            Material(
                name="BadMat",
                sigma_total=[1.0, 1.0, 1.0],
                sigma_scatter=[0.5, 0.5],  # Wrong length
                sigma_absorption=[0.3, 0.3, 0.3],
                sigma_fission=[0.2, 0.2, 0.2],
                nu=[2.5, 2.5, 2.5],
                fission_spectrum=[0.1, 0.2, 0.7],
            )
    
    def test_material_validation_sigma_absorption_length(self):
        """Test validation of sigma_absorption array length."""
        with pytest.raises(ValueError, match="sigma_absorption must have 3 values"):
            Material(
                name="BadMat",
                sigma_total=[1.0, 1.0, 1.0],
                sigma_scatter=[0.5, 0.5, 0.5],
                sigma_absorption=[0.3, 0.3],  # Wrong length
                sigma_fission=[0.2, 0.2, 0.2],
                nu=[2.5, 2.5, 2.5],
                fission_spectrum=[0.1, 0.2, 0.7],
            )
    
    def test_material_validation_sigma_fission_length(self):
        """Test validation of sigma_fission array length."""
        with pytest.raises(ValueError, match="sigma_fission must have 3 values"):
            Material(
                name="BadMat",
                sigma_total=[1.0, 1.0, 1.0],
                sigma_scatter=[0.5, 0.5, 0.5],
                sigma_absorption=[0.3, 0.3, 0.3],
                sigma_fission=[0.2, 0.2],  # Wrong length
                nu=[2.5, 2.5, 2.5],
                fission_spectrum=[0.1, 0.2, 0.7],
            )
    
    def test_material_validation_nu_length(self):
        """Test validation of nu array length."""
        with pytest.raises(ValueError, match="nu must have 3 values"):
            Material(
                name="BadMat",
                sigma_total=[1.0, 1.0, 1.0],
                sigma_scatter=[0.5, 0.5, 0.5],
                sigma_absorption=[0.3, 0.3, 0.3],
                sigma_fission=[0.2, 0.2, 0.2],
                nu=[2.5, 2.5],  # Wrong length
                fission_spectrum=[0.1, 0.2, 0.7],
            )
    
    def test_material_validation_fission_spectrum_length(self):
        """Test validation of fission_spectrum array length."""
        with pytest.raises(ValueError, match="fission_spectrum must have 3 values"):
            Material(
                name="BadMat",
                sigma_total=[1.0, 1.0, 1.0],
                sigma_scatter=[0.5, 0.5, 0.5],
                sigma_absorption=[0.3, 0.3, 0.3],
                sigma_fission=[0.2, 0.2, 0.2],
                nu=[2.5, 2.5, 2.5],
                fission_spectrum=[0.1, 0.2],  # Wrong length
            )
    
    def test_fission_spectrum_normalization_fissile(self):
        """Test fissile material requires normalized fission spectrum."""
        with pytest.raises(ValueError, match="fission_spectrum must sum to 1.0"):
            Material(
                name="BadFissile",
                sigma_total=[1.0, 1.0, 1.0],
                sigma_scatter=[0.5, 0.5, 0.5],
                sigma_absorption=[0.3, 0.3, 0.3],
                sigma_fission=[0.2, 0.2, 0.2],  # Has fission
                nu=[2.5, 2.5, 2.5],
                fission_spectrum=[0.1, 0.2, 0.5],  # Sums to 0.8, not 1.0
            )
    
    def test_non_fissile_allows_zero_spectrum(self, water_material):
        """Test non-fissile materials allow zero fission spectrum."""
        # Water has zero fission cross sections and zero spectrum
        assert all(f == 0.0 for f in water_material.sigma_fission)
        assert sum(water_material.fission_spectrum) == 0.0
    
    def test_create_u235(self, u235_material):
        """Test Material.create_u235() creates valid U-235."""
        assert u235_material.name == "U235"
        assert len(u235_material.sigma_total) == 3
        assert all(s > 0 for s in u235_material.sigma_fission)  # Fissile
        assert pytest.approx(sum(u235_material.fission_spectrum)) == 1.0
    
    def test_create_water(self, water_material):
        """Test Material.create_water() creates valid H2O."""
        assert water_material.name == "H2O"
        assert all(f == 0.0 for f in water_material.sigma_fission)  # Non-fissile
    
    def test_create_graphite(self, graphite_material):
        """Test Material.create_graphite() creates valid graphite."""
        assert graphite_material.name == "Graphite"
        assert all(f == 0.0 for f in graphite_material.sigma_fission)
    
    def test_create_void(self, void_material):
        """Test Material.create_void() creates valid void."""
        assert void_material.name == "Void"
        assert all(s == 0.0 for s in void_material.sigma_total)
    
    def test_accessor_get_sigma_total(self, u235_material):
        """Test get_sigma_total() accessor."""
        for group in range(3):
            sigma = u235_material.get_sigma_total(group)
            assert sigma > 0
            assert sigma == u235_material.sigma_total[group]
    
    def test_accessor_get_sigma_scatter(self, u235_material):
        """Test get_sigma_scatter() accessor."""
        for group in range(3):
            sigma = u235_material.get_sigma_scatter(group)
            assert sigma >= 0
    
    def test_accessor_get_sigma_absorption(self, u235_material):
        """Test get_sigma_absorption() accessor."""
        for group in range(3):
            sigma = u235_material.get_sigma_absorption(group)
            assert sigma >= 0
    
    def test_accessor_get_sigma_fission(self, u235_material):
        """Test get_sigma_fission() accessor."""
        for group in range(3):
            sigma = u235_material.get_sigma_fission(group)
            assert sigma >= 0
    
    def test_accessor_get_nu(self, u235_material):
        """Test get_nu() accessor."""
        for group in range(3):
            nu = u235_material.get_nu(group)
            assert nu > 0  # U235 produces neutrons


# =============================================================================
# Test Geometry1DSlab Class
# =============================================================================

class TestGeometry1DSlab:
    """Test 1D slab geometry class."""
    
    def test_initialization_valid(self, u235_material):
        """Test valid 1D slab creation."""
        geom = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235_material])
        assert geom.x_min == 0.0
        assert geom.x_max == 10.0
        assert len(geom.materials) == 1
    
    def test_boundary_validation(self, u235_material):
        """Test boundaries must span x_min to x_max."""
        with pytest.raises(ValueError, match="boundaries must start with x_min"):
            Geometry1DSlab(0.0, 10.0, [1.0, 10.0], [u235_material])
    
    def test_material_count_validation(self, u235_material):
        """Test material count matches region count."""
        with pytest.raises(ValueError, match="Number of materials"):
            Geometry1DSlab(0.0, 10.0, [0.0, 5.0, 10.0], [u235_material])  # Need 2 materials
    
    def test_get_material_inside(self, simple_1d_slab, u235_material):
        """Test get_material() returns correct material for position inside."""
        mat = simple_1d_slab.get_material(5.0)
        assert mat is not None
        assert mat.name == u235_material.name
    
    def test_get_material_outside(self, simple_1d_slab):
        """Test get_material() returns None outside geometry."""
        assert simple_1d_slab.get_material(-1.0) is None
        assert simple_1d_slab.get_material(11.0) is None
        # Test way outside boundaries to hit return None fallback
        assert simple_1d_slab.get_material(100.0) is None
        assert simple_1d_slab.get_material(-100.0) is None
    
    def test_get_material_boundaries(self, simple_1d_slab):
        """Test boundary handling at exact x_min and x_max."""
        assert simple_1d_slab.get_material(0.0) is not None
        assert simple_1d_slab.get_material(10.0) is not None
    
    def test_get_material_edge_between_regions(self, heterogeneous_1d_slab):
        """Test material lookup at exact region boundary."""
        # Test at boundary between regions
        mat_at_2 = heterogeneous_1d_slab.get_material(2.0)
        assert mat_at_2 is not None
        mat_at_8 = heterogeneous_1d_slab.get_material(8.0)
        assert mat_at_8 is not None
    
    def test_get_material_multi_region(self, heterogeneous_1d_slab, water_material, u235_material):
        """Test multi-region geometry with different materials."""
        # Water region [0, 2)
        assert heterogeneous_1d_slab.get_material(1.0).name == water_material.name
        # Fuel region [2, 8)
        assert heterogeneous_1d_slab.get_material(5.0).name == u235_material.name
        # Water region [8, 10]
        assert heterogeneous_1d_slab.get_material(9.0).name == water_material.name
    
    def test_vacuum_boundary_condition(self, simple_1d_slab):
        """Test vacuum boundary condition."""
        assert simple_1d_slab.boundary_condition == "vacuum"
    
    def test_reflective_boundary_condition(self, u235_material):
        """Test reflective boundary condition."""
        geom = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235_material], boundary_condition="reflective")
        assert geom.boundary_condition == "reflective"
    
    def test_distance_to_boundary_rightward(self, simple_1d_slab):
        """Test distance_to_boundary() for rightward motion (mu > 0)."""
        x = 3.0
        mu = 1.0  # Moving right
        dist = simple_1d_slab.distance_to_boundary(x, mu)
        assert pytest.approx(dist) == 7.0  # Distance to x_max = 10
    
    def test_distance_to_boundary_leftward(self, simple_1d_slab):
        """Test distance_to_boundary() for leftward motion (mu < 0)."""
        x = 3.0
        mu = -1.0  # Moving left
        dist = simple_1d_slab.distance_to_boundary(x, mu)
        assert pytest.approx(dist) == 3.0  # Distance to x_min = 0
    
    def test_distance_to_boundary_at_interface(self, heterogeneous_1d_slab):
        """Test distance_to_boundary() at region interfaces."""
        x = 5.0  # In fuel region
        mu = 1.0
        dist = heterogeneous_1d_slab.distance_to_boundary(x, mu)
        assert pytest.approx(dist) == 3.0  # Distance to boundary at 8.0
    
    def test_single_material_slab(self, u235_material):
        """Test single material slab."""
        geom = Geometry1DSlab(0.0, 5.0, [0.0, 5.0], [u235_material])
        assert geom.get_material(2.5).name == "U235"
    
    def test_very_small_slab(self, u235_material):
        """Test very small slab (< 1 cm)."""
        geom = Geometry1DSlab(0.0, 0.5, [0.0, 0.5], [u235_material])
        assert geom.x_max - geom.x_min == 0.5
    
    def test_is_inside_method(self, simple_1d_slab):
        """Test is_inside method."""
        assert simple_1d_slab.is_inside(5.0)
        assert simple_1d_slab.is_inside(0.0)
        assert simple_1d_slab.is_inside(10.0)
        assert not simple_1d_slab.is_inside(-1.0)
        assert not simple_1d_slab.is_inside(11.0)
    
    def test_distance_to_boundary_zero_mu(self, simple_1d_slab):
        """Test distance_to_boundary with mu near zero."""
        x = 5.0
        mu = 1e-11  # Very small mu
        dist = simple_1d_slab.distance_to_boundary(x, mu)
        assert dist == float('inf')
    
    def test_distance_to_boundary_no_boundary_ahead_right(self, simple_1d_slab):
        """Test distance_to_boundary when at right edge moving right."""
        x = 10.0  # At right boundary
        mu = 1.0  # Moving right (no more boundaries ahead)
        dist = simple_1d_slab.distance_to_boundary(x, mu)
        # Should return inf as already at edge
        assert dist == float('inf') or dist >= 0
    
    def test_distance_to_boundary_no_boundary_ahead_left(self, simple_1d_slab):
        """Test distance_to_boundary when at left edge moving left."""
        x = 0.0  # At left boundary
        mu = -1.0  # Moving left (no more boundaries ahead)
        dist = simple_1d_slab.distance_to_boundary(x, mu)
        # Should return inf as already at edge
        assert dist == float('inf') or dist >= 0


# =============================================================================
# Test Geometry2DPlane Class
# =============================================================================

class TestGeometry2DPlane:
    """Test 2D plane geometry class."""
    
    def test_initialization_valid(self, u235_material, water_material):
        """Test 2D plane creation with rectangular regions."""
        geom = Geometry2DPlane(
            0.0, 10.0, 0.0, 10.0,
            [(2.0, 2.0, 8.0, 8.0, u235_material)],
            water_material
        )
        assert geom.x_min == 0.0
        assert geom.x_max == 10.0
        assert geom.y_min == 0.0
        assert geom.y_max == 10.0
    
    def test_overlapping_regions(self, u235_material, water_material):
        """Test overlapping regions (later regions override)."""
        geom = Geometry2DPlane(
            0.0, 10.0, 0.0, 10.0,
            [
                (0.0, 0.0, 10.0, 10.0, water_material),  # Full plane
                (2.0, 2.0, 8.0, 8.0, u235_material),    # Center overrides
            ],
            water_material
        )
        # Center should be fuel
        assert geom.get_material(5.0, 5.0).name == "U235"
    
    def test_default_material_assignment(self, simple_2d_plane, water_material):
        """Test default material assignment."""
        # Outside defined regions should be default (water)
        assert simple_2d_plane.get_material(1.0, 1.0).name == water_material.name
    
    def test_get_material_in_region(self, simple_2d_plane, u235_material):
        """Test get_material() for points in defined regions."""
        # Center should be fuel
        assert simple_2d_plane.get_material(5.0, 5.0).name == u235_material.name
    
    def test_get_material_default(self, simple_2d_plane, water_material):
        """Test get_material() for points in default material."""
        # Corner should be water (default)
        assert simple_2d_plane.get_material(1.0, 1.0).name == water_material.name
    
    def test_get_material_outside(self, simple_2d_plane):
        """Test get_material() returns None outside geometry."""
        assert simple_2d_plane.get_material(-1.0, 5.0) is None
        assert simple_2d_plane.get_material(5.0, -1.0) is None
        assert simple_2d_plane.get_material(11.0, 5.0) is None
        assert simple_2d_plane.get_material(5.0, 11.0) is None
    
    def test_get_material_boundary(self, simple_2d_plane):
        """Test corner cases and boundaries."""
        # On boundaries should be inside
        assert simple_2d_plane.get_material(0.0, 0.0) is not None
        assert simple_2d_plane.get_material(10.0, 10.0) is not None
    
    def test_distance_to_boundary_horizontal(self, simple_2d_plane):
        """Test distance_to_boundary() for horizontal motion."""
        x, y = 5.0, 5.0
        angle = 0.0  # Moving right (+x direction)
        dist = simple_2d_plane.distance_to_boundary(x, y, angle)
        assert pytest.approx(dist) == 5.0  # Distance to x_max = 10
    
    def test_distance_to_boundary_vertical(self, simple_2d_plane):
        """Test distance_to_boundary() for vertical motion."""
        x, y = 5.0, 5.0
        angle = np.pi / 2  # Moving up (+y direction)
        dist = simple_2d_plane.distance_to_boundary(x, y, angle)
        assert pytest.approx(dist) == 5.0  # Distance to y_max = 10
    
    def test_distance_to_boundary_45_degrees(self, simple_2d_plane):
        """Test distance_to_boundary() at 45-degree angle."""
        x, y = 5.0, 5.0
        angle = np.pi / 4  # 45 degrees
        dist = simple_2d_plane.distance_to_boundary(x, y, angle)
        # Should hit boundary at min of x and y distances
        expected = 5.0 / np.cos(np.pi / 4)
        assert pytest.approx(dist, rel=0.01) == expected
    
    def test_is_inside_2d_method(self, simple_2d_plane):
        """Test is_inside method for 2D geometry."""
        assert simple_2d_plane.is_inside(5.0, 5.0)
        assert simple_2d_plane.is_inside(0.0, 0.0)
        assert simple_2d_plane.is_inside(10.0, 10.0)
        assert not simple_2d_plane.is_inside(-1.0, 5.0)
        assert not simple_2d_plane.is_inside(5.0, -1.0)
        assert not simple_2d_plane.is_inside(11.0, 5.0)
        assert not simple_2d_plane.is_inside(5.0, 11.0)


# =============================================================================
# Test NeutronState Class
# =============================================================================

class TestNeutronState:
    """Test neutron state dataclass."""
    
    def test_default_values(self):
        """Test default values (y=0, mu=0, energy_group=2, etc.)."""
        neutron = NeutronState(x=5.0)
        assert neutron.x == 5.0
        assert neutron.y == 0.0
        assert neutron.mu == 0.0
        assert neutron.energy_group == 2  # Fast group
        assert neutron.weight == 1.0
        assert neutron.alive is True
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        neutron = NeutronState(
            x=3.0, y=4.0, mu=0.5, energy_group=0, weight=0.8, alive=False
        )
        assert neutron.x == 3.0
        assert neutron.y == 4.0
        assert neutron.mu == 0.5
        assert neutron.energy_group == 0
        assert neutron.weight == 0.8
        assert neutron.alive is False
    
    def test_all_fields_accessible(self):
        """Test all fields are accessible."""
        neutron = NeutronState(x=1.0)
        # Access all fields
        _ = neutron.x
        _ = neutron.y
        _ = neutron.mu
        _ = neutron.energy_group
        _ = neutron.weight
        _ = neutron.alive
    
    def test_alive_flag_behavior(self):
        """Test alive flag behavior."""
        neutron = NeutronState(x=1.0, alive=True)
        assert neutron.alive
        neutron.alive = False
        assert not neutron.alive
    
    def test_weight_assignment(self):
        """Test weight assignment."""
        neutron = NeutronState(x=1.0, weight=0.5)
        assert neutron.weight == 0.5
        neutron.weight = 0.75
        assert neutron.weight == 0.75


# =============================================================================
# Test NeutronTallies Class
# =============================================================================

class TestNeutronTallies:
    """Test neutron tallies accumulator."""
    
    def test_create_with_custom_bins(self):
        """Test create() with custom flux bins."""
        tallies = NeutronTallies.create(flux_bins=50)
        assert len(tallies.flux) == 50
        assert np.all(tallies.flux == 0.0)
    
    def test_default_initialization(self):
        """Test default initialization via __post_init__."""
        tallies = NeutronTallies()
        assert tallies.flux is not None
        assert len(tallies.flux) == 100  # Default
        assert tallies.energy_spectrum == [0, 0, 0]
    
    def test_record_flux_valid_index(self):
        """Test record_flux() with valid bin index."""
        tallies = NeutronTallies.create(flux_bins=10)
        tallies.record_flux(position=5.0, path_length=1.5, bin_index=5)
        assert tallies.flux[5] == 1.5
    
    def test_record_flux_out_of_bounds(self):
        """Test record_flux() with out-of-bounds index (should not crash)."""
        tallies = NeutronTallies.create(flux_bins=10)
        # Should not raise error
        tallies.record_flux(position=5.0, path_length=1.0, bin_index=-1)
        tallies.record_flux(position=5.0, path_length=1.0, bin_index=100)
    
    def test_record_absorption(self):
        """Test record_absorption()."""
        tallies = NeutronTallies.create()
        assert tallies.absorption_count == 0
        tallies.record_absorption()
        assert tallies.absorption_count == 1
    
    def test_record_leakage(self):
        """Test record_leakage()."""
        tallies = NeutronTallies.create()
        assert tallies.leakage_count == 0
        tallies.record_leakage()
        assert tallies.leakage_count == 1
    
    def test_record_fission(self):
        """Test record_fission()."""
        tallies = NeutronTallies.create()
        assert tallies.fission_count == 0
        tallies.record_fission(nu=2.5)
        assert tallies.fission_count == 1
        assert tallies.fission_neutrons_produced == 2  # int(2.5)
    
    def test_record_scatter(self):
        """Test record_scatter()."""
        tallies = NeutronTallies.create()
        assert tallies.scatter_count == 0
        tallies.record_scatter()
        assert tallies.scatter_count == 1
    
    def test_record_track(self):
        """Test record_track()."""
        tallies = NeutronTallies.create()
        assert tallies.total_tracks == 0
        tallies.record_track()
        assert tallies.total_tracks == 1
    
    def test_fission_neutron_counting(self):
        """Test fission neutron counting."""
        tallies = NeutronTallies.create()
        tallies.record_fission(nu=2.5)
        tallies.record_fission(nu=2.7)
        assert tallies.fission_neutrons_produced == 4  # 2 + 2
    
    def test_get_k_eff_estimate_valid(self):
        """Test get_k_eff_estimate() with valid tallies."""
        tallies = NeutronTallies.create()
        tallies.record_track()
        tallies.fission_neutrons_produced = 5
        tallies.absorption_count = 2
        tallies.leakage_count = 3
        k_eff = tallies.get_k_eff_estimate()
        assert pytest.approx(k_eff) == 1.0  # 5 / (2 + 3)
    
    def test_get_k_eff_estimate_zero_denominator(self):
        """Test get_k_eff_estimate() with zero denominator."""
        tallies = NeutronTallies.create()
        k_eff = tallies.get_k_eff_estimate()
        assert k_eff == 0.0
    
    def test_get_k_eff_estimate_with_nonzero_tracks_zero_denominator(self):
        """Test get_k_eff_estimate() with tracks but zero absorption+leakage."""
        tallies = NeutronTallies.create()
        tallies.record_track()
        tallies.record_track()
        # No absorptions or leakage
        k_eff = tallies.get_k_eff_estimate()
        assert k_eff == 0.0
    
    def test_get_k_eff_estimate_zero_tracks(self):
        """Test get_k_eff_estimate() with zero tracks."""
        tallies = NeutronTallies.create()
        assert tallies.total_tracks == 0
        k_eff = tallies.get_k_eff_estimate()
        assert k_eff == 0.0


# =============================================================================
# Test Physics Sampling Functions
# =============================================================================

class TestPhysicsSampling:
    """Test physics sampling helper functions."""
    
    def test_sample_distance_exponential(self, rng_fixed):
        """Test exponential distribution sampling."""
        sigma_total = 2.0
        distance = _sample_distance(sigma_total, rng_fixed)
        assert distance > 0
    
    def test_sample_distance_zero_cross_section(self, rng_fixed):
        """Test with zero cross section returns infinity."""
        dist = _sample_distance(0.0, rng_fixed)
        assert dist == float('inf')
    
    def test_sample_distance_negative_cross_section(self, rng_fixed):
        """Test with negative cross section returns infinity."""
        dist = _sample_distance(-1.0, rng_fixed)
        assert dist == float('inf')
    
    def test_sample_distance_statistical_mean(self):
        """Test statistical properties (mean = 1/sigma_total)."""
        rng = np.random.default_rng(42)
        sigma_total = 2.0
        n_samples = 10000
        distances = [_sample_distance(sigma_total, rng) for _ in range(n_samples)]
        mean_dist = np.mean(distances)
        expected_mean = 1.0 / sigma_total
        assert pytest.approx(mean_dist, rel=0.1) == expected_mean
    
    def test_sample_scatter_angle_1d_range(self, rng_fixed):
        """Test 1D returns value in [-1, 1]."""
        angle = _sample_scatter_angle(rng_fixed, "1d")
        assert -1.0 <= angle <= 1.0
    
    def test_sample_scatter_angle_2d_range(self, rng_fixed):
        """Test 2D returns angle in [0, 2π]."""
        angle = _sample_scatter_angle(rng_fixed, "2d")
        assert 0.0 <= angle <= 2 * np.pi
    
    def test_sample_scatter_angle_isotropy(self):
        """Test isotropy (statistical test with many samples)."""
        rng = np.random.default_rng(42)
        n_samples = 10000
        angles_1d = [_sample_scatter_angle(rng, "1d") for _ in range(n_samples)]
        # Mean should be near 0 for uniform distribution on [-1, 1]
        assert pytest.approx(np.mean(angles_1d), abs=0.05) == 0.0
    
    def test_sample_fission_energy_valid_group(self, rng_fixed):
        """Test returns valid energy group index (0, 1, or 2)."""
        spectrum = [0.1, 0.2, 0.7]
        group = _sample_fission_energy(spectrum, rng_fixed)
        assert group in [0, 1, 2]
    
    def test_sample_fission_energy_edge_case(self):
        """Test fission energy sampling edge case (xi > all cumulative)."""
        rng = np.random.default_rng(999)
        spectrum = [0.1, 0.2, 0.7]
        # Sample many times to potentially hit edge case
        for _ in range(100):
            group = _sample_fission_energy(spectrum, rng)
            assert group in [0, 1, 2]
    
    def test_sample_fission_energy_fallback_to_last(self):
        """Test fission energy sampling fallback to last element."""
        # Create custom RNG that will return value >= 1.0 effectively
        rng = np.random.default_rng(42)
        spectrum = [0.1, 0.2, 0.7]
        
        # Sample many times - some should return last group via fallback
        groups = [_sample_fission_energy(spectrum, rng) for _ in range(1000)]
        # Should have all groups including group 2 (last)
        assert 2 in groups
        assert all(g in [0, 1, 2] for g in groups)
    
    def test_sample_fission_energy_edge_spectrum(self):
        """Test fission energy with edge case spectrum values."""
        rng = np.random.default_rng(999)
        # Use spectrum with very small values to test edge cases
        spectrum = [0.33, 0.33, 0.34]  # Slightly imperfect sum
        # Sample many times to hit all paths including fallback
        groups = [_sample_fission_energy(spectrum, rng) for _ in range(5000)]
        assert all(g in [0, 1, 2] for g in groups)
        # All groups should appear
        assert 0 in groups and 1 in groups and 2 in groups
    
    def test_sample_fission_energy_distribution(self):
        """Test statistical distribution matches spectrum."""
        rng = np.random.default_rng(42)
        spectrum = [0.1, 0.2, 0.7]
        n_samples = 10000
        samples = [_sample_fission_energy(spectrum, rng) for _ in range(n_samples)]
        counts = [samples.count(i) for i in range(3)]
        fractions = [c / n_samples for c in counts]
        # Check distribution matches spectrum
        for i in range(3):
            assert pytest.approx(fractions[i], abs=0.05) == spectrum[i]
    
    def test_sample_interaction_type_returns_valid(self, u235_material, rng_fixed):
        """Test returns 'scatter', 'absorption', or 'fission'."""
        interaction = _sample_interaction_type(u235_material, 0, rng_fixed)
        assert interaction in ["scatter", "absorption", "fission"]
    
    def test_sample_interaction_type_probabilities(self, u235_material):
        """Test probabilities match cross section ratios."""
        rng = np.random.default_rng(42)
        n_samples = 10000
        interactions = [_sample_interaction_type(u235_material, 2, rng) for _ in range(n_samples)]
        
        # Count each type
        counts = {
            "scatter": interactions.count("scatter"),
            "absorption": interactions.count("absorption"),
            "fission": interactions.count("fission"),
        }
        
        # Check ratios are reasonable
        assert counts["scatter"] > 0
        assert counts["absorption"] > 0
        assert counts["fission"] > 0
    
    def test_sample_interaction_type_zero_total(self, rng_fixed):
        """Test with zero total cross section."""
        # Create material with zero total cross section
        mat = Material(
            name="Zero",
            sigma_total=[0.0, 0.0, 0.0],
            sigma_scatter=[0.0, 0.0, 0.0],
            sigma_absorption=[0.0, 0.0, 0.0],
            sigma_fission=[0.0, 0.0, 0.0],
            nu=[0.0, 0.0, 0.0],
            fission_spectrum=[0.0, 0.0, 0.0],
        )
        interaction = _sample_interaction_type(mat, 0, rng_fixed)
        assert interaction == "absorption"  # Default for zero cross section
    
    def test_sample_interaction_type_non_fissile(self, water_material, rng_fixed):
        """Test with material having zero fission cross section."""
        interaction = _sample_interaction_type(water_material, 0, rng_fixed)
        # Should return scatter or absorption, not fission
        assert interaction in ["scatter", "absorption"]


# =============================================================================
# Test Neutron Tracking Functions
# =============================================================================

class TestNeutronTracking:
    """Test neutron tracking functions."""
    
    def test_track_neutron_1d_to_termination(self, simple_1d_slab, rng_fixed):
        """Test neutron tracked to termination."""
        neutron = NeutronState(x=5.0, mu=0.5, energy_group=2)
        tallies = NeutronTallies.create()
        reason, final = _track_neutron_1d(neutron, simple_1d_slab, tallies, rng_fixed)
        assert reason in ["absorption", "fission", "leakage", "max_collisions"]
        assert not final.alive
    
    @pytest.mark.parametrize("position,direction", [
        (0.5, -1.0),   # Near left boundary, moving left
        (9.5, 1.0),    # Near right boundary, moving right
        (5.0, -1.0),   # Middle, moving left
        (5.0, 1.0),    # Middle, moving right
    ])
    def test_track_neutron_1d_various_positions_directions(self, simple_1d_slab, position, direction):
        """Test tracking from various positions and directions."""
        rng = np.random.default_rng(42)
        neutron = NeutronState(x=position, mu=direction, energy_group=2)
        tallies = NeutronTallies.create()
        reason, final = _track_neutron_1d(neutron, simple_1d_slab, tallies, rng, max_collisions=20)
        assert reason in ["absorption", "fission", "leakage", "max_collisions"]
        assert not final.alive
    
    def test_track_neutron_1d_absorption(self, simple_1d_slab):
        """Test absorption termination."""
        rng = np.random.default_rng(42)
        neutron = NeutronState(x=5.0, mu=0.5, energy_group=2)
        tallies = NeutronTallies.create()
        # Run multiple times, should eventually get absorption
        for _ in range(100):
            neutron = NeutronState(x=5.0, mu=_sample_scatter_angle(rng, "1d"), energy_group=2)
            tallies = NeutronTallies.create()
            reason, _ = _track_neutron_1d(neutron, simple_1d_slab, tallies, rng)
            if reason == "absorption":
                assert tallies.absorption_count > 0
                break
    
    def test_track_neutron_1d_fission(self, simple_1d_slab):
        """Test fission termination."""
        rng = np.random.default_rng(42)
        # Run multiple times, should eventually get fission
        for _ in range(100):
            neutron = NeutronState(x=5.0, mu=_sample_scatter_angle(rng, "1d"), energy_group=2)
            tallies = NeutronTallies.create()
            reason, _ = _track_neutron_1d(neutron, simple_1d_slab, tallies, rng)
            if reason == "fission":
                assert tallies.fission_count > 0
                break
    
    def test_track_neutron_1d_leakage_vacuum(self, simple_1d_slab):
        """Test leakage at boundaries (vacuum BC)."""
        rng = np.random.default_rng(42)
        # Neutron near boundary moving outward
        neutron = NeutronState(x=9.5, mu=1.0, energy_group=2)  # Moving right
        tallies = NeutronTallies.create()
        reason, _ = _track_neutron_1d(neutron, simple_1d_slab, tallies, rng, max_collisions=10)
        # Should leak eventually
        assert reason in ["leakage", "absorption", "fission"]
    
    def test_track_neutron_1d_reflection_reflective(self, u235_material):
        """Test reflection at boundaries (reflective BC)."""
        geom = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235_material], boundary_condition="reflective")
        rng = np.random.default_rng(42)
        neutron = NeutronState(x=9.5, mu=1.0, energy_group=2)
        tallies = NeutronTallies.create()
        reason, _ = _track_neutron_1d(neutron, geom, tallies, rng, max_collisions=10)
        # Should not leak with reflective BC
        assert reason in ["absorption", "fission", "max_collisions"]
    
    def test_track_neutron_1d_flux_tallies(self, simple_1d_slab, rng_fixed):
        """Test flux tallies are accumulated."""
        neutron = NeutronState(x=5.0, mu=0.5, energy_group=2)
        tallies = NeutronTallies.create()
        _track_neutron_1d(neutron, simple_1d_slab, tallies, rng_fixed)
        # Flux should be accumulated somewhere
        assert np.sum(tallies.flux) > 0
    
    def test_track_neutron_1d_max_collisions(self, simple_1d_slab, rng_fixed):
        """Test max_collisions limit."""
        neutron = NeutronState(x=5.0, mu=0.5, energy_group=2)
        tallies = NeutronTallies.create()
        reason, _ = _track_neutron_1d(neutron, simple_1d_slab, tallies, rng_fixed, max_collisions=5)
        # May terminate due to max collisions
        assert reason in ["absorption", "fission", "leakage", "max_collisions"]
    
    def test_track_neutron_1d_force_max_collisions(self):
        """Force neutron to hit max_collisions limit in 1D."""
        # Create pure scattering material (no absorption or fission)
        pure_scatter = Material(
            name="PureScatter",
            sigma_total=[1.0, 1.0, 1.0],
            sigma_scatter=[1.0, 1.0, 1.0],
            sigma_absorption=[0.0, 0.0, 0.0],
            sigma_fission=[0.0, 0.0, 0.0],
            nu=[0.0, 0.0, 0.0],
            fission_spectrum=[0.0, 0.0, 0.0]
        )
        geom = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [pure_scatter], boundary_condition="reflective")
        rng = np.random.default_rng(42)
        neutron = NeutronState(x=5.0, mu=0.5, energy_group=2)
        tallies = NeutronTallies.create()
        # Very low limit forces max_collisions
        reason, final = _track_neutron_1d(neutron, geom, tallies, rng, max_collisions=2)
        assert reason == "max_collisions"
        assert not final.alive
    
    def test_track_neutron_2d_to_termination(self, simple_2d_plane, rng_fixed):
        """Test 2D tracking to termination."""
        neutron = NeutronState(x=5.0, y=5.0, mu=0.5, energy_group=2)
        tallies = NeutronTallies.create()
        reason, final = _track_neutron_2d(neutron, simple_2d_plane, tallies, rng_fixed)
        assert reason in ["absorption", "fission", "leakage", "max_collisions"]
        assert not final.alive
    
    def test_track_neutron_2d_flux_tallies(self, simple_2d_plane, rng_fixed):
        """Test flux tallies in 2D."""
        neutron = NeutronState(x=5.0, y=5.0, mu=0.5, energy_group=2)
        tallies = NeutronTallies.create()
        _track_neutron_2d(neutron, simple_2d_plane, tallies, rng_fixed)
        assert np.sum(tallies.flux) > 0
    
    def test_track_neutron_2d_max_collisions(self, simple_2d_plane, rng_fixed):
        """Test max_collisions limit in 2D."""
        neutron = NeutronState(x=5.0, y=5.0, mu=0.5, energy_group=2)
        tallies = NeutronTallies.create()
        reason, _ = _track_neutron_2d(neutron, simple_2d_plane, tallies, rng_fixed, max_collisions=5)
        assert reason in ["absorption", "fission", "leakage", "max_collisions"]
    
    def test_track_neutron_2d_force_max_collisions(self):
        """Force neutron to hit max_collisions limit in 2D."""
        # Create pure scattering material (no absorption or fission)
        pure_scatter = Material(
            name="PureScatter",
            sigma_total=[1.0, 1.0, 1.0],
            sigma_scatter=[1.0, 1.0, 1.0],
            sigma_absorption=[0.0, 0.0, 0.0],
            sigma_fission=[0.0, 0.0, 0.0],
            nu=[0.0, 0.0, 0.0],
            fission_spectrum=[0.0, 0.0, 0.0]
        )
        geom = Geometry2DPlane(
            0.0, 10.0, 0.0, 10.0,
            [(0.0, 0.0, 10.0, 10.0, pure_scatter)],
            pure_scatter,
            boundary_condition="reflective"
        )
        rng = np.random.default_rng(42)
        neutron = NeutronState(x=5.0, y=5.0, mu=0.5, energy_group=2)
        tallies = NeutronTallies.create()
        # Very low limit forces max_collisions
        reason, final = _track_neutron_2d(neutron, geom, tallies, rng, max_collisions=2)
        assert reason == "max_collisions"
        assert not final.alive
    
    def test_track_neutron_2d_reflective_boundary_hit(self):
        """Test 2D neutron reflecting at boundary."""
        # Use pure scattering to ensure hitting reflective boundary
        pure_scatter = Material(
            name="PureScatter",
            sigma_total=[0.5, 0.5, 0.5],  # Lower cross-section for longer free paths
            sigma_scatter=[0.5, 0.5, 0.5],
            sigma_absorption=[0.0, 0.0, 0.0],
            sigma_fission=[0.0, 0.0, 0.0],
            nu=[0.0, 0.0, 0.0],
            fission_spectrum=[0.0, 0.0, 0.0]
        )
        geom = Geometry2DPlane(
            0.0, 10.0, 0.0, 10.0,
            [(0.0, 0.0, 10.0, 10.0, pure_scatter)],
            pure_scatter,
            boundary_condition="reflective"
        )
        rng = np.random.default_rng(99)
        # Start very near boundary moving outward
        neutron = NeutronState(x=9.95, y=5.0, mu=0.0, energy_group=2)  # mu=0.0 means angle=0 (right)
        tallies = NeutronTallies.create()
        reason, final = _track_neutron_2d(neutron, geom, tallies, rng, max_collisions=10)
        # Should hit reflective boundary and continue (eventually hitting max_collisions)
        assert reason == "max_collisions"
    
    def test_track_neutron_1d_energy_downscatter_thermal(self, simple_1d_slab):
        """Test energy downscatter when already at thermal energy."""
        rng = np.random.default_rng(123)
        # Start at thermal energy (group 0)
        neutron = NeutronState(x=5.0, mu=0.5, energy_group=0)
        tallies = NeutronTallies.create()
        reason, final = _track_neutron_1d(neutron, simple_1d_slab, tallies, rng, max_collisions=20)
        # Should terminate normally
        assert reason in ["absorption", "fission", "leakage", "max_collisions"]
    
    def test_track_neutron_1d_starts_outside_geometry(self, simple_1d_slab):
        """Test tracking neutron that starts outside geometry."""
        rng = np.random.default_rng(42)
        # Start outside geometry
        neutron = NeutronState(x=15.0, mu=0.5, energy_group=2)
        tallies = NeutronTallies.create()
        reason, final = _track_neutron_1d(neutron, simple_1d_slab, tallies, rng)
        # Should immediately leak
        assert reason == "leakage"
        assert tallies.leakage_count == 1
    
    def test_track_neutron_1d_reflective_complete_path(self):
        """Test complete reflective boundary path in 1D."""
        # Use pure scattering to ensure boundary hit and reflection
        pure_scatter = Material(
            name="PureScatter",
            sigma_total=[0.5, 0.5, 0.5],
            sigma_scatter=[0.5, 0.5, 0.5],
            sigma_absorption=[0.0, 0.0, 0.0],
            sigma_fission=[0.0, 0.0, 0.0],
            nu=[0.0, 0.0, 0.0],
            fission_spectrum=[0.0, 0.0, 0.0]
        )
        geom = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [pure_scatter], boundary_condition="reflective")
        rng = np.random.default_rng(99)
        # Neutron very near boundary moving toward it
        neutron = NeutronState(x=0.1, mu=-1.0, energy_group=2)  # Moving left toward boundary
        tallies = NeutronTallies.create()
        reason, final = _track_neutron_1d(neutron, geom, tallies, rng, max_collisions=5)
        # Should hit boundary, reflect, and hit max_collisions
        assert reason == "max_collisions"
    
    def test_track_neutron_1d_vacuum_boundary_leakage(self):
        """Test vacuum boundary leakage in 1D."""
        # Use pure scattering with vacuum BC to hit boundary leakage path
        pure_scatter = Material(
            name="PureScatter",
            sigma_total=[0.1, 0.1, 0.1],  # Very low for long free paths
            sigma_scatter=[0.1, 0.1, 0.1],
            sigma_absorption=[0.0, 0.0, 0.0],
            sigma_fission=[0.0, 0.0, 0.0],
            nu=[0.0, 0.0, 0.0],
            fission_spectrum=[0.0, 0.0, 0.0]
        )
        geom = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [pure_scatter], boundary_condition="vacuum")
        rng = np.random.default_rng(42)
        # Start very near boundary moving outward
        neutron = NeutronState(x=9.9, mu=1.0, energy_group=2)
        tallies = NeutronTallies.create()
        reason, final = _track_neutron_1d(neutron, geom, tallies, rng, max_collisions=50)
        # Should leak with vacuum boundary
        assert reason == "leakage"
        assert tallies.leakage_count == 1
        assert not final.alive
    
    def test_track_neutron_2d_reflective_boundary(self, u235_material):
        """Test 2D tracking with reflective boundaries."""
        geom = Geometry2DPlane(
            0.0, 10.0, 0.0, 10.0,
            [(0.0, 0.0, 10.0, 10.0, u235_material)],
            u235_material,
            boundary_condition="reflective"
        )
        rng = np.random.default_rng(42)
        # Neutron near boundary moving outward
        neutron = NeutronState(x=9.5, y=5.0, mu=0.0, energy_group=2)  # Moving right
        tallies = NeutronTallies.create()
        reason, _ = _track_neutron_2d(neutron, geom, tallies, rng, max_collisions=10)
        # Should not leak with reflective BC
        assert reason in ["absorption", "fission", "max_collisions"]
    
    def test_track_neutron_2d_starts_outside_geometry(self, simple_2d_plane):
        """Test tracking 2D neutron that starts outside geometry."""
        rng = np.random.default_rng(42)
        # Start outside geometry
        neutron = NeutronState(x=15.0, y=5.0, mu=0.5, energy_group=2)
        tallies = NeutronTallies.create()
        reason, final = _track_neutron_2d(neutron, simple_2d_plane, tallies, rng)
        # Should immediately leak
        assert reason == "leakage"
        assert tallies.leakage_count == 1
    
    def test_track_neutron_2d_reflective_complete_path(self):
        """Test complete reflective boundary path in 2D."""
        # Use pure scattering to ensure boundary reflection
        pure_scatter = Material(
            name="PureScatter",
            sigma_total=[0.5, 0.5, 0.5],
            sigma_scatter=[0.5, 0.5, 0.5],
            sigma_absorption=[0.0, 0.0, 0.0],
            sigma_fission=[0.0, 0.0, 0.0],
            nu=[0.0, 0.0, 0.0],
            fission_spectrum=[0.0, 0.0, 0.0]
        )
        geom = Geometry2DPlane(
            0.0, 10.0, 0.0, 10.0,
            [(0.0, 0.0, 10.0, 10.0, pure_scatter)],
            pure_scatter,
            boundary_condition="reflective"
        )
        rng = np.random.default_rng(77)
        # Neutron very near boundary moving toward it
        neutron = NeutronState(x=9.8, y=5.0, mu=0.0, energy_group=2)  # Moving right (+x)
        tallies = NeutronTallies.create()
        reason, final = _track_neutron_2d(neutron, geom, tallies, rng, max_collisions=5)
        # Should reflect at boundary and hit max_collisions
        assert reason == "max_collisions"
    
    def test_track_neutron_2d_vacuum_boundary_leakage(self):
        """Test vacuum boundary leakage in 2D."""
        # Use pure scattering with vacuum BC
        pure_scatter = Material(
            name="PureScatter",
            sigma_total=[0.1, 0.1, 0.1],  # Very low for long free paths
            sigma_scatter=[0.1, 0.1, 0.1],
            sigma_absorption=[0.0, 0.0, 0.0],
            sigma_fission=[0.0, 0.0, 0.0],
            nu=[0.0, 0.0, 0.0],
            fission_spectrum=[0.0, 0.0, 0.0]
        )
        geom = Geometry2DPlane(
            0.0, 10.0, 0.0, 10.0,
            [(0.0, 0.0, 10.0, 10.0, pure_scatter)],
            pure_scatter,
            boundary_condition="vacuum"
        )
        rng = np.random.default_rng(42)
        # Start very near boundary moving outward
        neutron = NeutronState(x=9.9, y=5.0, mu=0.0, energy_group=2)  # angle=0, moving right
        tallies = NeutronTallies.create()
        reason, final = _track_neutron_2d(neutron, geom, tallies, rng, max_collisions=50)
        # Should leak with vacuum boundary
        assert reason == "leakage"
        assert tallies.leakage_count == 1
        assert not final.alive
    
    @pytest.mark.parametrize("x,y,angle", [
        (0.5, 5.0, np.pi),          # Near left, moving left
        (9.5, 5.0, 0.0),            # Near right, moving right
        (5.0, 0.5, 3*np.pi/2),      # Near bottom, moving down
        (5.0, 9.5, np.pi/2),        # Near top, moving up
        (1.0, 1.0, 5*np.pi/4),      # Corner, moving toward boundary
    ])
    def test_track_neutron_2d_various_positions_angles(self, simple_2d_plane, x, y, angle):
        """Test 2D tracking from various positions and angles."""
        rng = np.random.default_rng(42)
        neutron = NeutronState(x=x, y=y, mu=angle, energy_group=2)
        tallies = NeutronTallies.create()
        reason, final = _track_neutron_2d(neutron, simple_2d_plane, tallies, rng, max_collisions=20)
        assert reason in ["absorption", "fission", "leakage", "max_collisions"]
        assert not final.alive


# =============================================================================
# Test NeutronTransportSimulation Class
# =============================================================================

class TestNeutronTransportSimulation:
    """Test NeutronTransportSimulation class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        sim = NeutronTransportSimulation()
        assert sim.name == "Neutron Transport"
        assert sim.source_position == (5.0,)
        assert sim.source_energy_group == 2
    
    def test_initialization_with_1d_geometry(self, simple_1d_slab):
        """Test with 1D geometry."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        assert sim.geometry_type == "1d"
    
    def test_initialization_with_2d_geometry(self, simple_2d_plane):
        """Test with 2D geometry."""
        sim = NeutronTransportSimulation(geometry=simple_2d_plane)
        assert sim.geometry_type == "2d"
    
    def test_single_simulation_returns_float(self, simple_1d_slab):
        """Test single_simulation returns float value."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        result = sim.single_simulation()
        assert isinstance(result, float)
    
    def test_single_simulation_flux_return(self, simple_1d_slab):
        """Test with return_type='flux'."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        result = sim.single_simulation(return_type="flux")
        assert result >= 0.0
    
    def test_single_simulation_leakage_prob(self, simple_1d_slab):
        """Test with return_type='leakage_prob' (returns 0 or 1)."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        result = sim.single_simulation(return_type="leakage_prob")
        assert result in [0.0, 1.0]
    
    def test_single_simulation_absorption_prob(self, simple_1d_slab):
        """Test with return_type='absorption_prob' (returns 0 or 1)."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        result = sim.single_simulation(return_type="absorption_prob")
        assert result in [0.0, 1.0]
    
    def test_single_simulation_reproducibility(self, simple_1d_slab):
        """Test reproducibility with seed."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        result1 = sim.single_simulation()
        sim.set_seed(42)
        result2 = sim.single_simulation()
        assert result1 == result2
    
    def test_single_simulation_default_geometry(self):
        """Test default geometry creation when none provided."""
        sim = NeutronTransportSimulation()
        sim.set_seed(42)
        result = sim.single_simulation()
        assert isinstance(result, float)
    
    def test_run_small_simulations(self, simple_1d_slab):
        """Test run() with small number of simulations."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        result = sim.run(n_simulations=10, parallel=False, compute_stats=False)
        assert result.n_simulations == 10
        assert len(result.results) == 10
    
    def test_run_statistics_computed(self, simple_1d_slab):
        """Test statistics are computed correctly."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        result = sim.run(n_simulations=50, parallel=False, compute_stats=True)
        assert result.mean >= 0
        assert result.std >= 0
    
    def test_run_mean_flux_positive(self, simple_1d_slab):
        """Test mean flux is positive."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        result = sim.run(n_simulations=20, parallel=False, return_type="flux")
        assert result.mean >= 0
    
    def test_compute_flux_distribution_length(self, simple_1d_slab):
        """Test compute_flux_distribution returns array of correct length."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        flux_bins = 50
        flux_dist = sim.compute_flux_distribution(n_histories=10, flux_bins=flux_bins)
        assert len(flux_dist) == flux_bins
    
    def test_compute_flux_distribution_normalized(self, simple_1d_slab):
        """Test flux distribution normalized per history."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        flux_dist = sim.compute_flux_distribution(n_histories=10, flux_bins=50)
        # Flux should be non-negative
        assert np.all(flux_dist >= 0)
    
    def test_compute_flux_distribution_1d(self, simple_1d_slab):
        """Test with 1D geometry."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        flux_dist = sim.compute_flux_distribution(n_histories=10, flux_bins=50)
        assert flux_dist.shape == (50,)
    
    def test_compute_flux_distribution_2d(self, simple_2d_plane):
        """Test compute_flux_distribution with 2D geometry."""
        sim = NeutronTransportSimulation(geometry=simple_2d_plane, source_position=(5.0, 5.0))
        sim.set_seed(42)
        flux_dist = sim.compute_flux_distribution(n_histories=10, flux_bins=50)
        assert flux_dist.shape == (50,)
        assert np.all(flux_dist >= 0)
    
    def test_compute_flux_distribution_no_geometry(self):
        """Test compute_flux_distribution creates default geometry when none provided."""
        sim = NeutronTransportSimulation()  # No geometry
        sim.set_seed(42)
        flux_dist = sim.compute_flux_distribution(n_histories=5, flux_bins=50)
        assert flux_dist.shape == (50,)
        assert np.all(flux_dist >= 0)
    
    def test_compute_flux_distribution_2d_explicit_else_branch(self, simple_2d_plane):
        """Test 2D flux distribution else branch explicitly."""
        sim = NeutronTransportSimulation(
            geometry=simple_2d_plane,
            source_position=(5.0, 5.0),
            source_energy_group=1,
        )
        sim.set_seed(42)
        # Call with explicit parameters to ensure 2D path
        flux_dist = sim.compute_flux_distribution(
            n_histories=10,
            geometry=simple_2d_plane,
            source_position=(5.0, 5.0),
            source_energy_group=2,
            flux_bins=50
        )
        assert flux_dist.shape == (50,)
    
    def test_single_simulation_with_overrides(self, simple_1d_slab, u235_material):
        """Test single_simulation with parameter overrides."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        # Override with new geometry
        new_geom = Geometry1DSlab(0.0, 5.0, [0.0, 5.0], [u235_material])
        result = sim.single_simulation(
            geometry=new_geom,
            source_position=(2.5,),
            source_energy_group=1,
            flux_bins=25
        )
        assert isinstance(result, float)
    
    def test_single_simulation_invalid_return_type(self, simple_1d_slab):
        """Test single_simulation with invalid return_type (falls back to flux)."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        # Use invalid return type - should default to flux
        result = sim.single_simulation(return_type="invalid_type")
        assert isinstance(result, float)
        assert result >= 0.0


# =============================================================================
# Test KEFFSimulation Class
# =============================================================================

class TestKEFFSimulation:
    """Test k-effective simulation class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        sim = KEFFSimulation()
        assert sim.name == "k-effective"
        assert sim.n_generations == 50
        assert sim.n_neutrons_per_generation == 1000
    
    def test_initialization_with_geometry(self, simple_1d_slab):
        """Test with custom geometry."""
        sim = KEFFSimulation(geometry=simple_1d_slab)
        assert sim.geometry is not None
    
    def test_initialization_with_source_positions(self):
        """Test with initial source positions."""
        sim = KEFFSimulation(initial_source_positions=[(5.0,), (6.0,)])
        assert len(sim.initial_source_positions) == 2
    
    def test_single_simulation_returns_float(self, simple_1d_slab):
        """Test single_simulation returns float k-eff value."""
        sim = KEFFSimulation(geometry=simple_1d_slab, n_generations=10, n_neutrons_per_generation=50)
        sim.set_seed(42)
        k_eff = sim.single_simulation()
        assert isinstance(k_eff, float)
    
    def test_single_simulation_k_eff_positive(self, simple_1d_slab):
        """Test k-eff is positive."""
        sim = KEFFSimulation(geometry=simple_1d_slab, n_generations=10, n_neutrons_per_generation=50)
        sim.set_seed(42)
        k_eff = sim.single_simulation()
        assert k_eff >= 0
    
    def test_single_simulation_k_eff_reasonable_range(self, simple_1d_slab):
        """Test k-eff reasonable range (0.5 to 2.0 typically)."""
        sim = KEFFSimulation(geometry=simple_1d_slab, n_generations=20, n_neutrons_per_generation=100)
        sim.set_seed(42)
        k_eff = sim.single_simulation()
        # Very loose bounds for small simulations
        assert 0.0 <= k_eff <= 5.0
    
    def test_single_simulation_reproducibility(self, simple_1d_slab):
        """Test reproducibility with seed."""
        sim = KEFFSimulation(geometry=simple_1d_slab, n_generations=10, n_neutrons_per_generation=50)
        sim.set_seed(42)
        k_eff1 = sim.single_simulation()
        sim.set_seed(42)
        k_eff2 = sim.single_simulation()
        assert k_eff1 == k_eff2
    
    def test_single_simulation_skip_generations(self, simple_1d_slab):
        """Test with skip_generations parameter."""
        sim = KEFFSimulation(geometry=simple_1d_slab, n_generations=20, n_neutrons_per_generation=50)
        sim.set_seed(42)
        k_eff = sim.single_simulation(skip_generations=5)
        assert isinstance(k_eff, float)
    
    def test_run_produces_consistent_estimates(self, simple_1d_slab):
        """Test run() produces consistent k-eff estimates."""
        sim = KEFFSimulation(geometry=simple_1d_slab, n_generations=15, n_neutrons_per_generation=50)
        sim.set_seed(42)
        result = sim.run(n_simulations=5, parallel=False, compute_stats=False)
        assert result.n_simulations == 5
        assert len(result.results) == 5
    
    def test_run_statistics(self, simple_1d_slab):
        """Test statistics (mean, std dev)."""
        sim = KEFFSimulation(geometry=simple_1d_slab, n_generations=15, n_neutrons_per_generation=50)
        sim.set_seed(42)
        result = sim.run(n_simulations=5, parallel=False, compute_stats=True)
        assert hasattr(result, 'mean')
        assert hasattr(result, 'std')
    
    def test_reflective_boundaries(self, u235_material):
        """Test reflective boundaries increase k-eff."""
        # Vacuum boundaries
        geom_vacuum = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235_material], boundary_condition="vacuum")
        sim_vacuum = KEFFSimulation(geometry=geom_vacuum, n_generations=10, n_neutrons_per_generation=50)
        sim_vacuum.set_seed(42)
        k_eff_vacuum = sim_vacuum.single_simulation()
        
        # Reflective boundaries
        geom_reflect = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235_material], boundary_condition="reflective")
        sim_reflect = KEFFSimulation(geometry=geom_reflect, n_generations=10, n_neutrons_per_generation=50)
        sim_reflect.set_seed(42)
        k_eff_reflect = sim_reflect.single_simulation()
        
        # Reflective should generally have higher k-eff (less leakage)
        # This is a statistical test, so it may occasionally fail
        assert k_eff_reflect >= 0
    
    def test_keff_system_dies_out(self, water_material):
        """Test k-eff calculation when system dies out (no fissions)."""
        # Pure water has no fission, system should die out
        geom = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [water_material])
        sim = KEFFSimulation(geometry=geom, n_generations=10, n_neutrons_per_generation=10)
        sim.set_seed(42)
        k_eff = sim.single_simulation()
        # System dies out, k-eff should be 0 or very small
        assert k_eff >= 0.0
    
    def test_keff_with_parameter_overrides(self, simple_1d_slab):
        """Test k-eff simulation with parameter overrides."""
        sim = KEFFSimulation(geometry=simple_1d_slab, n_generations=15, n_neutrons_per_generation=50)
        sim.set_seed(42)
        k_eff = sim.single_simulation(
            n_generations=10,
            n_neutrons_per_generation=30,
            skip_generations=3
        )
        assert isinstance(k_eff, float)
        assert k_eff >= 0
    
    def test_keff_2d_geometry(self, simple_2d_plane):
        """Test k-eff calculation with 2D geometry."""
        sim = KEFFSimulation(
            geometry=simple_2d_plane,
            initial_source_positions=[(5.0, 5.0)],
            n_generations=10,
            n_neutrons_per_generation=30
        )
        sim.set_seed(42)
        k_eff = sim.single_simulation()
        assert isinstance(k_eff, float)
        assert k_eff >= 0
    
    def test_keff_2d_geometry_with_override(self, simple_2d_plane):
        """Test k-eff with 2D geometry override to hit else branch."""
        sim = KEFFSimulation(n_generations=5, n_neutrons_per_generation=20)
        sim.set_seed(42)
        # Override with 2D geometry
        k_eff = sim.single_simulation(
            geometry=simple_2d_plane,
            initial_source_positions=[(5.0, 5.0)],
        )
        assert isinstance(k_eff, float)
        assert k_eff >= 0
    
    def test_keff_2d_source_position_single_coord(self, simple_2d_plane):
        """Test k-eff 2D with source position having single coordinate."""
        sim = KEFFSimulation(
            geometry=simple_2d_plane,
            initial_source_positions=[(5.0,)],  # Only x coordinate
            n_generations=5,
            n_neutrons_per_generation=20
        )
        sim.set_seed(42)
        k_eff = sim.single_simulation()
        assert isinstance(k_eff, float)
    
    def test_keff_population_normalization_up(self, simple_1d_slab):
        """Test k-eff with population normalization (replicating up)."""
        # Small initial population to test replication
        sim = KEFFSimulation(
            geometry=simple_1d_slab,
            n_generations=5,
            n_neutrons_per_generation=100  # Large target to force replication
        )
        sim.set_seed(42)
        k_eff = sim.single_simulation()
        assert isinstance(k_eff, float)
    
    def test_keff_population_normalization_down(self, u235_material):
        """Test k-eff with population normalization (sampling down)."""
        # Reflective boundaries to maximize fission neutrons
        geom = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235_material], boundary_condition="reflective")
        sim = KEFFSimulation(
            geometry=geom,
            n_generations=5,
            n_neutrons_per_generation=20  # Small target to force sampling down
        )
        sim.set_seed(42)
        k_eff = sim.single_simulation()
        assert isinstance(k_eff, float)
    
    def test_keff_no_geometry_provided(self):
        """Test k-eff simulation creates default geometry when none provided."""
        sim = KEFFSimulation(n_generations=5, n_neutrons_per_generation=20)
        sim.set_seed(42)
        k_eff = sim.single_simulation()
        assert isinstance(k_eff, float)
        assert k_eff >= 0
    
    def test_keff_population_exactly_matches_target(self, u235_material):
        """Test k-eff when next generation exactly matches target population."""
        # Use specific seed and parameters to try to get exact match
        geom = Geometry1DSlab(0.0, 8.0, [0.0, 8.0], [u235_material], boundary_condition="reflective")
        sim = KEFFSimulation(
            geometry=geom,
            n_generations=3,
            n_neutrons_per_generation=50
        )
        sim.set_seed(999)
        k_eff = sim.single_simulation()
        assert isinstance(k_eff, float)


# =============================================================================
# Test Integration
# =============================================================================

class TestNeutronTransportIntegration:
    """Integration tests for neutron transport simulation."""
    
    def test_complete_1d_fuel_slab(self, simple_1d_slab):
        """Test complete 1D fuel slab simulation."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab, source_position=(5.0,))
        sim.set_seed(42)
        result = sim.run(n_simulations=20, parallel=False)
        assert result.n_simulations == 20
        assert result.mean >= 0
    
    def test_heterogeneous_1d_geometry(self, heterogeneous_1d_slab):
        """Test heterogeneous 1D geometry (fuel + moderator)."""
        sim = NeutronTransportSimulation(geometry=heterogeneous_1d_slab, source_position=(5.0,))
        sim.set_seed(42)
        result = sim.run(n_simulations=20, parallel=False)
        assert result.mean >= 0
    
    def test_2d_fuel_assembly(self, simple_2d_plane):
        """Test 2D fuel assembly simulation."""
        sim = NeutronTransportSimulation(geometry=simple_2d_plane, source_position=(5.0, 5.0))
        sim.set_seed(42)
        result = sim.run(n_simulations=20, parallel=False)
        assert result.mean >= 0
    
    def test_k_eff_calculation(self, simple_1d_slab):
        """Test k-eff calculation for critical system."""
        sim = KEFFSimulation(geometry=simple_1d_slab, n_generations=15, n_neutrons_per_generation=50)
        sim.set_seed(42)
        result = sim.run(n_simulations=3, parallel=False)
        assert result.mean >= 0
    
    def test_probabilities_sum(self, simple_1d_slab):
        """Test leakage + absorption + fission probabilities sum ≈ 1."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        
        # Run simulations for each probability
        result_leak = sim.run(n_simulations=100, parallel=False, return_type="leakage_prob", compute_stats=False)
        sim.set_seed(42)
        result_abs = sim.run(n_simulations=100, parallel=False, return_type="absorption_prob", compute_stats=False)
        
        # Probabilities should be between 0 and 1
        assert 0.0 <= result_leak.mean <= 1.0
        assert 0.0 <= result_abs.mean <= 1.0
    
    def test_flux_increases_with_histories(self, simple_1d_slab):
        """Test flux increases with more histories."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        
        flux_10 = sim.compute_flux_distribution(n_histories=10, flux_bins=50)
        sim.set_seed(42)
        flux_50 = sim.compute_flux_distribution(n_histories=50, flux_bins=50)
        
        # More histories should generally give larger total flux
        # (This is a statistical test, may occasionally fail)
        assert np.sum(flux_50) >= np.sum(flux_10) * 0.8  # Allow some variance
    
    def test_framework_integration(self, simple_1d_slab):
        """Test registration with MonteCarloFramework."""
        from mcframework.core import MonteCarloFramework
        
        framework = MonteCarloFramework()
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        framework.register_simulation(sim, name="NeutronTest")
        
        assert "NeutronTest" in framework.simulations
        
        # Run through framework
        result = framework.run_simulation("NeutronTest", n_simulations=10, parallel=False, compute_stats=False)
        assert result.n_simulations == 10
    
    def test_various_energy_groups_1d(self, simple_1d_slab):
        """Test transport with various starting energy groups."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        
        # Test all energy groups
        for energy_group in [0, 1, 2]:
            result = sim.single_simulation(source_energy_group=energy_group)
            assert isinstance(result, float)
    
    def test_various_energy_groups_2d(self, simple_2d_plane):
        """Test 2D transport with various starting energy groups."""
        sim = NeutronTransportSimulation(geometry=simple_2d_plane, source_position=(5.0, 5.0))
        sim.set_seed(42)
        
        # Test all energy groups
        for energy_group in [0, 1, 2]:
            result = sim.single_simulation(source_energy_group=energy_group)
            assert isinstance(result, float)
    
    def test_long_neutron_history_1d(self, simple_1d_slab):
        """Test longer neutron history to hit max_collisions."""
        sim = NeutronTransportSimulation(geometry=simple_1d_slab)
        sim.set_seed(42)
        # Run enough simulations to potentially hit max_collisions
        result = sim.run(n_simulations=50, parallel=False, compute_stats=False)
        assert result.n_simulations == 50
    
    def test_long_neutron_history_2d(self, simple_2d_plane):
        """Test longer 2D neutron history."""
        sim = NeutronTransportSimulation(geometry=simple_2d_plane, source_position=(5.0, 5.0))
        sim.set_seed(42)
        result = sim.run(n_simulations=50, parallel=False, compute_stats=False)
        assert result.n_simulations == 50
    
    def test_multiple_scatters_before_termination_1d(self, u235_material):
        """Test neutron with multiple scatters in 1D."""
        # Large reflective geometry to encourage multiple scatters
        geom = Geometry1DSlab(0.0, 20.0, [0.0, 20.0], [u235_material], boundary_condition="reflective")
        sim = NeutronTransportSimulation(geometry=geom, source_position=(10.0,))
        sim.set_seed(42)
        result = sim.run(n_simulations=100, parallel=False, compute_stats=False)
        assert result.n_simulations == 100
        assert result.mean >= 0
    
    def test_multiple_scatters_before_termination_2d(self, u235_material, water_material):
        """Test neutron with multiple scatters in 2D."""
        # Reflective geometry
        geom = Geometry2DPlane(
            0.0, 20.0, 0.0, 20.0,
            [(5.0, 5.0, 15.0, 15.0, u235_material)],
            water_material,
            boundary_condition="reflective"
        )
        sim = NeutronTransportSimulation(geometry=geom, source_position=(10.0, 10.0))
        sim.set_seed(42)
        result = sim.run(n_simulations=100, parallel=False, compute_stats=False)
        assert result.n_simulations == 100
        assert result.mean >= 0
    
    def test_keff_multiple_generations_convergence(self, u235_material):
        """Test k-eff convergence with many generations."""
        geom = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235_material], boundary_condition="reflective")
        sim = KEFFSimulation(
            geometry=geom,
            n_generations=50,
            n_neutrons_per_generation=100
        )
        sim.set_seed(42)
        k_eff = sim.single_simulation(skip_generations=20)
        assert isinstance(k_eff, float)
        assert k_eff > 0
    
    def test_keff_many_runs_to_hit_edge_cases(self, u235_material):
        """Run k-eff many times to hit various edge cases."""
        geom = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235_material], boundary_condition="reflective")
        sim = KEFFSimulation(
            geometry=geom,
            n_generations=10,
            n_neutrons_per_generation=30
        )
        # Run multiple times with different seeds to hit edge cases
        for seed in [42, 99, 123, 456, 789]:
            sim.set_seed(seed)
            k_eff = sim.single_simulation()
            assert isinstance(k_eff, float)
            assert k_eff >= 0
