"""Neutron transport simulation via Markov chain Monte Carlo.

This module implements neutron transport in 1D and 2D geometries with
energy-dependent cross sections. Supports criticality (k-effective),
flux distribution, and leakage/absorption statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.random import Generator

from ..core import MonteCarloSimulation

__all__ = [
    "Material",
    "Geometry1DSlab",
    "Geometry2DPlane",
    "NeutronState",
    "NeutronTallies",
    "NeutronTransportSimulation",
    "KEFFSimulation",
    "_sample_distance",
    "_sample_scatter_angle",
    "_sample_fission_energy",
    "_sample_interaction_type",
]


# ============================================================================
# Energy Groups (simplified 3-group structure)
# ============================================================================
ENERGY_GROUPS = {
    "thermal": 0.025,  # 0.025 eV
    "epithermal": 1.0,  # 1 eV
    "fast": 2.0e6,  # 2 MeV
}

ENERGY_GROUP_NAMES = ["thermal", "epithermal", "fast"]
ENERGY_GROUP_VALUES = [0.025, 1.0, 2.0e6]


@dataclass
class Material:
    r"""
    Nuclear material with energy-dependent cross sections.

    Cross sections are stored per energy group (thermal, epithermal, fast).
    All cross sections are in units of cm^-1 (macroscopic).

    Parameters
    ----------
    name : str
        Material identifier (e.g., "U235", "H2O", "Graphite").
    sigma_total : list[float]
        Total cross section for each energy group [thermal, epithermal, fast].
    sigma_scatter : list[float]
        Scattering cross section for each energy group.
    sigma_absorption : list[float]
        Absorption (capture) cross section for each energy group.
    sigma_fission : list[float]
        Fission cross section for each energy group.
    nu : list[float]
        Average number of neutrons produced per fission for each energy group.
    fission_spectrum : list[float]
        Probability distribution for secondary neutron energy groups.
        Should sum to 1.0.

    Notes
    -----
    The total cross section should satisfy:

    .. math::
       \sigma_{\text{total}} = \sigma_{\text{scatter}} + \sigma_{\text{absorption}} + \sigma_{\text{fission}}

    Examples
    --------
    >>> u235 = Material.create_u235()
    >>> h2o = Material.create_water()
    """

    name: str
    sigma_total: list[float]
    sigma_scatter: list[float]
    sigma_absorption: list[float]
    sigma_fission: list[float]
    nu: list[float]
    fission_spectrum: list[float]

    def __post_init__(self):
        """Validate cross section data."""
        n = len(ENERGY_GROUP_NAMES)
        if len(self.sigma_total) != n:
            raise ValueError(f"sigma_total must have {n} values")
        if len(self.sigma_scatter) != n:
            raise ValueError(f"sigma_scatter must have {n} values")
        if len(self.sigma_absorption) != n:
            raise ValueError(f"sigma_absorption must have {n} values")
        if len(self.sigma_fission) != n:
            raise ValueError(f"sigma_fission must have {n} values")
        if len(self.nu) != n:
            raise ValueError(f"nu must have {n} values")
        if len(self.fission_spectrum) != n:
            raise ValueError(f"fission_spectrum must have {n} values")

        # Check that fission spectrum is normalized (only for fissile materials)
        is_fissile = any(sigma_f > 0 for sigma_f in self.sigma_fission)
        if is_fissile and not np.isclose(sum(self.fission_spectrum), 1.0):
            raise ValueError("fission_spectrum must sum to 1.0 for fissile materials")

    def get_sigma_total(self, energy_group: int) -> float:
        """Get total cross section for energy group."""
        return self.sigma_total[energy_group]

    def get_sigma_scatter(self, energy_group: int) -> float:
        """Get scattering cross section for energy group."""
        return self.sigma_scatter[energy_group]

    def get_sigma_absorption(self, energy_group: int) -> float:
        """Get absorption cross section for energy group."""
        return self.sigma_absorption[energy_group]

    def get_sigma_fission(self, energy_group: int) -> float:
        """Get fission cross section for energy group."""
        return self.sigma_fission[energy_group]

    def get_nu(self, energy_group: int) -> float:
        """Get average neutrons per fission for energy group."""
        return self.nu[energy_group]

    @staticmethod
    def create_u235() -> Material:
        """
        Create U-235 fuel material with realistic cross sections.

        Returns
        -------
        Material
            U-235 fissile material.
        """
        return Material(
            name="U235",
            sigma_total=[8.0, 4.0, 5.0],  # cm^-1
            sigma_scatter=[2.0, 1.5, 2.5],  # cm^-1
            sigma_absorption=[3.0, 1.0, 0.5],  # cm^-1
            sigma_fission=[3.0, 1.5, 2.0],  # cm^-1
            nu=[2.5, 2.5, 2.5],  # neutrons per fission
            fission_spectrum=[0.1, 0.2, 0.7],  # mostly fast neutrons
        )

    @staticmethod
    def create_water() -> Material:
        """
        Create H2O moderator material.

        Returns
        -------
        Material
            Water moderator (high scattering, low absorption).
        """
        return Material(
            name="H2O",
            sigma_total=[3.0, 2.0, 1.5],
            sigma_scatter=[2.8, 1.9, 1.4],
            sigma_absorption=[0.2, 0.1, 0.1],
            sigma_fission=[0.0, 0.0, 0.0],
            nu=[0.0, 0.0, 0.0],
            fission_spectrum=[0.0, 0.0, 0.0],  # Not used for non-fissile
        )

    @staticmethod
    def create_graphite() -> Material:
        """
        Create graphite moderator material.

        Returns
        -------
        Material
            Graphite moderator (moderate scattering, very low absorption).
        """
        return Material(
            name="Graphite",
            sigma_total=[0.5, 0.4, 0.35],
            sigma_scatter=[0.48, 0.38, 0.33],
            sigma_absorption=[0.02, 0.02, 0.02],
            sigma_fission=[0.0, 0.0, 0.0],
            nu=[0.0, 0.0, 0.0],
            fission_spectrum=[0.0, 0.0, 0.0],
        )

    @staticmethod
    def create_void() -> Material:
        """
        Create void (vacuum) material.

        Returns
        -------
        Material
            Void with zero cross sections.
        """
        return Material(
            name="Void",
            sigma_total=[0.0, 0.0, 0.0],
            sigma_scatter=[0.0, 0.0, 0.0],
            sigma_absorption=[0.0, 0.0, 0.0],
            sigma_fission=[0.0, 0.0, 0.0],
            nu=[0.0, 0.0, 0.0],
            fission_spectrum=[0.0, 0.0, 0.0],
        )


@dataclass
class NeutronState:
    """
    State vector for a neutron during transport.

    Parameters
    ----------
    x : float
        x-position (cm).
    y : float
        y-position (cm), used only for 2D geometries.
    mu : float
        Direction cosine for 1D (-1 to 1), or angle (radians) for 2D.
    energy_group : int
        Energy group index (0=thermal, 1=epithermal, 2=fast).
    weight : float
        Statistical weight for variance reduction (1.0 for analog Monte Carlo).
    alive : bool
        Whether the neutron is still being tracked.
    """

    x: float
    y: float = 0.0
    mu: float = 0.0
    energy_group: int = 2  # Start in fast group
    weight: float = 1.0
    alive: bool = True


# ============================================================================
# Geometry Classes
# ============================================================================


class Geometry1DSlab:
    r"""
    One-dimensional slab geometry with multiple material regions.

    The slab extends from x_min to x_max with regions defined by boundaries.

    Parameters
    ----------
    x_min : float
        Left boundary of the slab (cm).
    x_max : float
        Right boundary of the slab (cm).
    boundaries : list[float]
        Sorted list of region boundaries including x_min and x_max.
    materials : list[Material]
        Material for each region. Length should be len(boundaries) - 1.
    boundary_condition : {"vacuum", "reflective"}
        Boundary condition at slab edges.

    Examples
    --------
    >>> u235 = Material.create_u235()
    >>> h2o = Material.create_water()
    >>> # Create a 10 cm slab: 2cm water | 6cm U235 | 2cm water
    >>> geom = Geometry1DSlab(0.0, 10.0, [0.0, 2.0, 8.0, 10.0], [h2o, u235, h2o])
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        boundaries: list[float],
        materials: list[Material],
        boundary_condition: Literal["vacuum", "reflective"] = "vacuum",
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.boundaries = sorted(boundaries)
        self.materials = materials
        self.boundary_condition = boundary_condition

        if len(materials) != len(boundaries) - 1:
            raise ValueError("Number of materials must equal number of regions (len(boundaries) - 1)")
        if self.boundaries[0] != x_min or self.boundaries[-1] != x_max:
            raise ValueError("boundaries must start with x_min and end with x_max")

    def get_material(self, x: float) -> Optional[Material]:
        """
        Get the material at position x.

        Parameters
        ----------
        x : float
            Position in the slab.

        Returns
        -------
        Material or None
            Material at position x, or None if outside geometry.
        """
        if x < self.x_min or x > self.x_max:
            return None

        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] <= x < self.boundaries[i + 1]:
                return self.materials[i]

        # Handle right boundary exactly
        if x == self.x_max:
            return self.materials[-1]

        return None  # pragma: no cover

    def is_inside(self, x: float) -> bool:
        """Check if position x is inside the geometry."""
        return self.x_min <= x <= self.x_max

    def distance_to_boundary(self, x: float, mu: float) -> float:
        """
        Calculate distance to next boundary in direction mu.

        Parameters
        ----------
        x : float
            Current position.
        mu : float
            Direction cosine (-1 to 1).

        Returns
        -------
        float
            Distance to next boundary (positive value).
        """
        if abs(mu) < 1e-10:
            return float("inf")

        if mu > 0:
            # Moving right - find next boundary above x
            for b in self.boundaries:
                if b > x + 1e-10:
                    return (b - x) / mu
            return float("inf")
        else:
            # Moving left - find next boundary below x
            for b in reversed(self.boundaries):
                if b < x - 1e-10:
                    return (x - b) / abs(mu)
            return float("inf")


class Geometry2DPlane:
    r"""
    Two-dimensional rectangular geometry with material regions.

    The plane extends from (x_min, y_min) to (x_max, y_max).
    Regions are defined as rectangles with associated materials.

    Parameters
    ----------
    x_min : float
        Left boundary (cm).
    x_max : float
        Right boundary (cm).
    y_min : float
        Bottom boundary (cm).
    y_max : float
        Top boundary (cm).
    regions : list[tuple[float, float, float, float, Material]]
        List of regions, each defined as (x1, y1, x2, y2, material).
    default_material : Material
        Material to use for regions not covered by regions list.
    boundary_condition : {"vacuum", "reflective"}
        Boundary condition at edges.

    Examples
    --------
    >>> u235 = Material.create_u235()
    >>> h2o = Material.create_water()
    >>> # 10x10 cm plane with 6x6 cm fuel in center, water surrounding
    >>> geom = Geometry2DPlane(0, 10, 0, 10, [(2, 2, 8, 8, u235)], h2o)
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        regions: list[Tuple[float, float, float, float, Material]],
        default_material: Material,
        boundary_condition: Literal["vacuum", "reflective"] = "vacuum",
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.regions = regions
        self.default_material = default_material
        self.boundary_condition = boundary_condition

    def get_material(self, x: float, y: float) -> Optional[Material]:
        """
        Get material at position (x, y).

        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.

        Returns
        -------
        Material or None
            Material at (x, y), or None if outside geometry.
        """
        if not self.is_inside(x, y):
            return None

        # Check regions in reverse order (later regions override earlier ones)
        for x1, y1, x2, y2, material in reversed(self.regions):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return material

        return self.default_material

    def is_inside(self, x: float, y: float) -> bool:
        """Check if position (x, y) is inside the geometry."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def distance_to_boundary(self, x: float, y: float, angle: float) -> float:
        """
        Calculate distance to geometry boundary in direction angle.

        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.
        angle : float
            Direction angle in radians.

        Returns
        -------
        float
            Distance to boundary.
        """
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        distances = []

        # Distance to each boundary
        if abs(cos_theta) > 1e-10:
            if cos_theta > 0:
                distances.append((self.x_max - x) / cos_theta)
            else:
                distances.append((self.x_min - x) / cos_theta)

        if abs(sin_theta) > 1e-10:
            if sin_theta > 0:
                distances.append((self.y_max - y) / sin_theta)
            else:
                distances.append((self.y_min - y) / sin_theta)

        # Return minimum positive distance
        positive_dists = [d for d in distances if d > 1e-10]
        return min(positive_dists) if positive_dists else float("inf")


# ============================================================================
# Tally System
# ============================================================================


@dataclass
class NeutronTallies:
    """
    Accumulator for neutron transport statistics.

    Parameters
    ----------
    flux_bins : int
        Number of spatial bins for flux tallies.
    geometry_type : {"1d", "2d"}
        Type of geometry being simulated.
    """

    flux: np.ndarray = None  # Flux per spatial bin
    absorption_count: int = 0
    leakage_count: int = 0
    fission_count: int = 0
    scatter_count: int = 0
    total_tracks: int = 0
    fission_neutrons_produced: int = 0
    energy_spectrum: list[int] = None  # Count per energy group

    def __post_init__(self):
        """Initialize arrays."""
        if self.flux is None:
            self.flux = np.zeros(100)  # Default 100 bins
        if self.energy_spectrum is None:
            self.energy_spectrum = [0, 0, 0]  # Three energy groups

    @staticmethod
    def create(flux_bins: int = 100) -> NeutronTallies:
        """Create a new tally object with specified bins."""
        return NeutronTallies(
            flux=np.zeros(flux_bins),
            energy_spectrum=[0, 0, 0],
        )

    def record_flux(self, position: float, path_length: float, bin_index: int):
        """Record flux contribution in a spatial bin."""
        if 0 <= bin_index < len(self.flux):
            self.flux[bin_index] += path_length

    def record_absorption(self):
        """Record an absorption event."""
        self.absorption_count += 1

    def record_leakage(self):
        """Record a leakage event."""
        self.leakage_count += 1

    def record_fission(self, nu: float):
        """Record a fission event and neutrons produced."""
        self.fission_count += 1
        self.fission_neutrons_produced += int(nu)

    def record_scatter(self):
        """Record a scattering event."""
        self.scatter_count += 1

    def record_track(self):
        """Increment total track count."""
        self.total_tracks += 1

    def get_k_eff_estimate(self) -> float:
        """
        Estimate k-effective from tallies.

        Returns
        -------
        float
            k_eff = (neutrons produced) / (neutrons absorbed + leaked)
        """
        if self.total_tracks == 0:
            return 0.0
        denominator = self.absorption_count + self.leakage_count
        if denominator == 0:
            return 0.0
        return float(self.fission_neutrons_produced) / float(denominator)


# ============================================================================
# Physics Sampling Functions
# ============================================================================


def _sample_distance(sigma_total: float, rng: Generator) -> float:
    r"""
    Sample free-flight distance from exponential distribution.

    The distance to collision follows:

    .. math::
       P(s) = \sigma_{\text{total}} \exp(-\sigma_{\text{total}} s)

    Parameters
    ----------
    sigma_total : float
        Total macroscopic cross section (cm^-1).
    rng : Generator
        Random number generator.

    Returns
    -------
    float
        Distance to next collision (cm).
    """
    if sigma_total <= 0:
        return float("inf")
    return -np.log(rng.random()) / sigma_total


def _sample_scatter_angle(rng: Generator, dimension: Literal["1d", "2d"]) -> float:
    """
    Sample isotropic scattering angle.

    Parameters
    ----------
    rng : Generator
        Random number generator.
    dimension : {"1d", "2d"}
        Geometry dimension.

    Returns
    -------
    float
        Direction cosine (1D) or angle in radians (2D).
    """
    if dimension == "1d":
        # Isotropic in 1D: uniform on [-1, 1]
        return 2.0 * rng.random() - 1.0
    else:
        # Isotropic in 2D: uniform angle on [0, 2π]
        return 2.0 * np.pi * rng.random()


def _sample_fission_energy(fission_spectrum: list[float], rng: Generator) -> int:
    """
    Sample energy group for fission neutron.

    Parameters
    ----------
    fission_spectrum : list[float]
        Probability distribution over energy groups.
    rng : Generator
        Random number generator.

    Returns
    -------
    int
        Energy group index.
    """
    cumulative = np.cumsum(fission_spectrum)
    xi = rng.random()
    for i, cum_prob in enumerate(cumulative):
        if xi <= cum_prob:
            return i
    return len(fission_spectrum) - 1  # pragma: no cover


def _sample_interaction_type(
    material: Material,
    energy_group: int,
    rng: Generator,
) -> Literal["scatter", "absorption", "fission"]:
    """
    Sample the type of neutron interaction.

    Parameters
    ----------
    material : Material
        Material in which interaction occurs.
    energy_group : int
        Current energy group.
    rng : Generator
        Random number generator.

    Returns
    -------
    str
        Interaction type: "scatter", "absorption", or "fission".
    """
    sigma_s = material.get_sigma_scatter(energy_group)
    sigma_a = material.get_sigma_absorption(energy_group)
    sigma_f = material.get_sigma_fission(energy_group) # noqa: F841 # used for debugging
    sigma_t = material.get_sigma_total(energy_group)

    if sigma_t <= 0:
        return "absorption"

    xi = rng.random()

    # Cumulative probabilities
    p_scatter = sigma_s / sigma_t
    p_absorption = (sigma_s + sigma_a) / sigma_t

    if xi < p_scatter:
        return "scatter"
    elif xi < p_absorption:
        return "absorption"
    else:
        return "fission"


# ============================================================================
# Neutron Tracking Functions
# ============================================================================


def _track_neutron_1d(
    neutron: NeutronState,
    geometry: Geometry1DSlab,
    tallies: NeutronTallies,
    rng: Generator,
    max_collisions: int = 1000,
) -> Tuple[str, NeutronState]:
    """
    Track a single neutron through 1D slab geometry until termination.

    Parameters
    ----------
    neutron : NeutronState
        Initial neutron state.
    geometry : Geometry1DSlab
        Geometry specification.
    tallies : NeutronTallies
        Tally accumulator.
    rng : Generator
        Random number generator.
    max_collisions : int
        Maximum collisions before forced termination.

    Returns
    -------
    tuple[str, NeutronState]
        (termination_reason, final_state) where reason is
        "absorption", "fission", "leakage", or "max_collisions".
    """
    tallies.record_track()
    n_collisions = 0

    while neutron.alive and n_collisions < max_collisions:
        # Get current material
        material = geometry.get_material(neutron.x)

        if material is None:
            # Leaked out of geometry
            tallies.record_leakage()
            neutron.alive = False
            return "leakage", neutron

        # Sample distance to collision
        sigma_t = material.get_sigma_total(neutron.energy_group)
        distance = _sample_distance(sigma_t, rng)

        # Distance to boundary
        dist_boundary = geometry.distance_to_boundary(neutron.x, neutron.mu)

        # Move neutron
        if distance < dist_boundary:
            # Collision occurs before boundary
            neutron.x += neutron.mu * distance

            # Tally flux contribution (track length estimator)
            bin_idx = int(
                (neutron.x - geometry.x_min) / (geometry.x_max - geometry.x_min) * len(tallies.flux)
            )
            tallies.record_flux(neutron.x, distance, bin_idx)

            # Sample interaction type
            interaction = _sample_interaction_type(material, neutron.energy_group, rng)

            if interaction == "scatter":
                tallies.record_scatter()
                # Isotropic scatter
                neutron.mu = _sample_scatter_angle(rng, "1d")
                # Energy downscatter (simplified: move to lower energy group)
                if neutron.energy_group > 0 and rng.random() < 0.3:
                    neutron.energy_group -= 1

            elif interaction == "absorption":
                tallies.record_absorption()
                neutron.alive = False
                return "absorption", neutron

            elif interaction == "fission":
                nu = material.get_nu(neutron.energy_group)
                tallies.record_fission(nu)
                neutron.alive = False
                return "fission", neutron

            n_collisions += 1

        else:
            # Reaches boundary before collision
            neutron.x += neutron.mu * dist_boundary

            # Tally flux
            bin_idx = int(
                (neutron.x - geometry.x_min) / (geometry.x_max - geometry.x_min) * len(tallies.flux)
            )
            tallies.record_flux(neutron.x, dist_boundary, bin_idx)

            # Check boundary condition
            if geometry.boundary_condition == "vacuum":
                tallies.record_leakage()
                neutron.alive = False
                return "leakage", neutron
            elif geometry.boundary_condition == "reflective":
                # Reflect direction
                neutron.mu = -neutron.mu
                # Small nudge to avoid boundary
                neutron.x += neutron.mu * 1e-8

    # Max collisions reached
    neutron.alive = False
    return "max_collisions", neutron


def _track_neutron_2d(
    neutron: NeutronState,
    geometry: Geometry2DPlane,
    tallies: NeutronTallies,
    rng: Generator,
    max_collisions: int = 1000,
) -> Tuple[str, NeutronState]:
    """
    Track a single neutron through 2D geometry until termination.

    Parameters
    ----------
    neutron : NeutronState
        Initial neutron state (mu interpreted as angle in radians).
    geometry : Geometry2DPlane
        Geometry specification.
    tallies : NeutronTallies
        Tally accumulator.
    rng : Generator
        Random number generator.
    max_collisions : int
        Maximum collisions before forced termination.

    Returns
    -------
    tuple[str, NeutronState]
        (termination_reason, final_state).
    """
    tallies.record_track()
    n_collisions = 0

    while neutron.alive and n_collisions < max_collisions:
        # Get current material
        material = geometry.get_material(neutron.x, neutron.y)

        if material is None:
            tallies.record_leakage()
            neutron.alive = False
            return "leakage", neutron

        # Sample distance to collision
        sigma_t = material.get_sigma_total(neutron.energy_group)
        distance = _sample_distance(sigma_t, rng)

        # Distance to boundary
        dist_boundary = geometry.distance_to_boundary(neutron.x, neutron.y, neutron.mu)

        # Move neutron
        if distance < dist_boundary:
            # Collision occurs
            neutron.x += distance * np.cos(neutron.mu)
            neutron.y += distance * np.sin(neutron.mu)

            # Tally flux
            x_frac = (neutron.x - geometry.x_min) / (geometry.x_max - geometry.x_min)
            bin_idx = int(x_frac * len(tallies.flux))
            tallies.record_flux(neutron.x, distance, bin_idx)

            # Sample interaction
            interaction = _sample_interaction_type(material, neutron.energy_group, rng)

            if interaction == "scatter":
                tallies.record_scatter()
                neutron.mu = _sample_scatter_angle(rng, "2d")
                # Energy downscatter
                if neutron.energy_group > 0 and rng.random() < 0.3:
                    neutron.energy_group -= 1

            elif interaction == "absorption":
                tallies.record_absorption()
                neutron.alive = False
                return "absorption", neutron

            elif interaction == "fission":
                nu = material.get_nu(neutron.energy_group)
                tallies.record_fission(nu)
                neutron.alive = False
                return "fission", neutron

            n_collisions += 1

        else:
            # Reaches boundary
            neutron.x += dist_boundary * np.cos(neutron.mu)
            neutron.y += dist_boundary * np.sin(neutron.mu)

            # Tally flux
            x_frac = (neutron.x - geometry.x_min) / (geometry.x_max - geometry.x_min)
            bin_idx = int(x_frac * len(tallies.flux))
            tallies.record_flux(neutron.x, dist_boundary, bin_idx)

            if geometry.boundary_condition == "vacuum":
                tallies.record_leakage()
                neutron.alive = False
                return "leakage", neutron
            elif geometry.boundary_condition == "reflective":
                # Reflect (simplified: reverse direction)
                neutron.mu = neutron.mu + np.pi
                if neutron.mu > 2 * np.pi:
                    neutron.mu -= 2 * np.pi
                # Small nudge
                neutron.x += 1e-8 * np.cos(neutron.mu)
                neutron.y += 1e-8 * np.sin(neutron.mu)

    neutron.alive = False
    return "max_collisions", neutron


# ============================================================================
# Main Simulation Classes
# ============================================================================


class NeutronTransportSimulation(MonteCarloSimulation):
    r"""
    Monte Carlo neutron transport simulation for 1D and 2D geometries.

    Simulates individual neutron histories through specified geometry with
    energy-dependent cross sections. Computes flux distributions, absorption
    rates, and leakage probabilities.

    Parameters
    ----------
    name : str, optional
        Simulation name. Defaults to "Neutron Transport".
    geometry : Geometry1DSlab or Geometry2DPlane
        Geometry specification.
    source_position : tuple[float, ...]
        Source position: (x,) for 1D or (x, y) for 2D.
    source_energy_group : int
        Initial energy group for source neutrons (0=thermal, 1=epithermal, 2=fast).
    flux_bins : int
        Number of spatial bins for flux tallies.

    Examples
    --------
    >>> from mcframework.sims.neutron_transport import *
    >>> u235 = Material.create_u235()
    >>> geom = Geometry1DSlab(0, 10, [0, 10], [u235])
    >>> sim = NeutronTransportSimulation(geometry=geom, source_position=(5.0,))
    >>> result = sim.run(1000, parallel=False)  # doctest: +SKIP

    Notes
    -----
    The simulation tracks individual neutrons from source to termination
    (absorption, fission, or leakage). Statistics are accumulated using
    track-length estimators for flux and event counters for reactions.
    """

    def __init__(
        self,
        name: str = "Neutron Transport",
        geometry: Optional[Geometry1DSlab | Geometry2DPlane] = None,
        source_position: Optional[tuple] = None,
        source_energy_group: int = 2,
        flux_bins: int = 100,
    ):
        super().__init__(name)
        self.geometry = geometry
        self.source_position = source_position or (5.0,)
        self.source_energy_group = source_energy_group
        self.flux_bins = flux_bins

        # Determine geometry type
        if isinstance(geometry, Geometry1DSlab):
            self.geometry_type = "1d"
        elif isinstance(geometry, Geometry2DPlane):
            self.geometry_type = "2d"
        else:
            self.geometry_type = "1d"

    def single_simulation(
        self,
        *,
        geometry: Optional[Geometry1DSlab | Geometry2DPlane] = None,
        source_position: Optional[tuple] = None,
        source_energy_group: Optional[int] = None,
        flux_bins: Optional[int] = None,
        return_type: Literal["flux", "leakage_prob", "absorption_prob"] = "flux",
        _rng: Optional[Generator] = None,
        **kwargs,
    ) -> float:
        r"""
        Simulate a single neutron history.

        Parameters
        ----------
        geometry : Geometry1DSlab or Geometry2DPlane, optional
            Geometry to use (overrides instance geometry).
        source_position : tuple, optional
            Source position (overrides instance position).
        source_energy_group : int, optional
            Initial energy group (overrides instance value).
        flux_bins : int, optional
            Number of flux bins (overrides instance value).
        return_type : {"flux", "leakage_prob", "absorption_prob"}
            Quantity to return from simulation.
        _rng : Generator, optional
            Random number generator.
        **kwargs
            Additional arguments (ignored).

        Returns
        -------
        float
            Depends on return_type:
            - "flux": total flux (sum over all bins)
            - "leakage_prob": 1.0 if leaked, 0.0 otherwise
            - "absorption_prob": 1.0 if absorbed, 0.0 otherwise
        """
        rng = self._rng(_rng, self.rng)

        # Use instance values if not overridden
        geom = geometry or self.geometry
        src_pos = source_position or self.source_position
        src_energy = source_energy_group if source_energy_group is not None else self.source_energy_group
        n_bins = flux_bins or self.flux_bins

        # Create default geometry if none provided
        if geom is None:
            u235 = Material.create_u235()
            geom = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235])

        # Initialize neutron
        if isinstance(geom, Geometry1DSlab):
            neutron = NeutronState(
                x=src_pos[0],
                y=0.0,
                mu=_sample_scatter_angle(rng, "1d"),
                energy_group=src_energy,
            )
            geom_type = "1d"
        else:
            neutron = NeutronState(
                x=src_pos[0],
                y=src_pos[1] if len(src_pos) > 1 else 0.0,
                mu=_sample_scatter_angle(rng, "2d"),
                energy_group=src_energy,
            )
            geom_type = "2d"

        # Create tallies
        tallies = NeutronTallies.create(n_bins)

        # Track neutron
        if geom_type == "1d":
            reason, final_state = _track_neutron_1d(neutron, geom, tallies, rng)
        else:
            reason, final_state = _track_neutron_2d(neutron, geom, tallies, rng)

        # Return requested quantity
        if return_type == "flux":
            return float(np.sum(tallies.flux))
        elif return_type == "leakage_prob":
            return 1.0 if reason == "leakage" else 0.0
        elif return_type == "absorption_prob":
            return 1.0 if reason == "absorption" else 0.0
        else:
            return float(np.sum(tallies.flux))

    def compute_flux_distribution(
        self,
        n_histories: int,
        geometry: Optional[Geometry1DSlab | Geometry2DPlane] = None,
        source_position: Optional[tuple] = None,
        source_energy_group: Optional[int] = None,
        flux_bins: int = 100,
        parallel: bool = False,
    ) -> np.ndarray:
        """
        Compute spatial flux distribution over multiple histories.

        Parameters
        ----------
        n_histories : int
            Number of neutron histories to simulate.
        geometry : Geometry1DSlab or Geometry2DPlane, optional
            Geometry specification.
        source_position : tuple, optional
            Source position.
        source_energy_group : int, optional
            Initial energy group.
        flux_bins : int
            Number of spatial bins.
        parallel : bool
            Whether to use parallel execution.

        Returns
        -------
        np.ndarray
            Flux distribution normalized per history.
        """
        rng = self.rng
        geom = geometry or self.geometry
        src_pos = source_position or self.source_position
        src_energy = source_energy_group if source_energy_group is not None else self.source_energy_group

        if geom is None:
            u235 = Material.create_u235()
            geom = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235])

        flux_total = np.zeros(flux_bins)

        for _ in range(n_histories):
            # Initialize neutron
            if isinstance(geom, Geometry1DSlab):
                neutron = NeutronState(
                    x=src_pos[0],
                    mu=_sample_scatter_angle(rng, "1d"),
                    energy_group=src_energy,
                )
                tallies = NeutronTallies.create(flux_bins)
                _track_neutron_1d(neutron, geom, tallies, rng)
            else:
                neutron = NeutronState(
                    x=src_pos[0],
                    y=src_pos[1] if len(src_pos) > 1 else 0.0,
                    mu=_sample_scatter_angle(rng, "2d"),
                    energy_group=src_energy,
                )
                tallies = NeutronTallies.create(flux_bins)
                _track_neutron_2d(neutron, geom, tallies, rng)

            flux_total += tallies.flux

        return flux_total / n_histories


class KEFFSimulation(MonteCarloSimulation):
    r"""
    k-effective (criticality) calculation using power iteration.

    Computes the effective neutron multiplication factor k_eff using
    successive fission generations. Each generation tracks neutrons from
    fission sources to their next fission or termination.

    Parameters
    ----------
    name : str, optional
        Simulation name. Defaults to "k-effective".
    geometry : Geometry1DSlab or Geometry2DPlane
        Geometry specification.
    initial_source_positions : list[tuple]
        Initial fission source positions.
    n_generations : int
        Number of fission generations per simulation.
    n_neutrons_per_generation : int
        Target number of neutrons per generation.

    Examples
    --------
    >>> from mcframework.sims.neutron_transport import *
    >>> u235 = Material.create_u235()
    >>> geom = Geometry1DSlab(0, 10, [0, 10], [u235])
    >>> sim = KEFFSimulation(geometry=geom)
    >>> result = sim.run(100, parallel=False)  # doctest: +SKIP

    Notes
    -----
    The k-effective eigenvalue represents the ratio of neutrons in successive
    generations:

    .. math::
       k_{\text{eff}} = \frac{n_{g+1}}{n_g}

    where :math:`n_g` is the number of neutrons in generation :math:`g`.
    A value of k_eff = 1 indicates a critical system.
    """

    def __init__(
        self,
        name: str = "k-effective",
        geometry: Optional[Geometry1DSlab | Geometry2DPlane] = None,
        initial_source_positions: Optional[list[tuple]] = None,
        n_generations: int = 50,
        n_neutrons_per_generation: int = 1000,
    ):
        super().__init__(name)
        self.geometry = geometry
        self.initial_source_positions = initial_source_positions or [(5.0,)]
        self.n_generations = n_generations
        self.n_neutrons_per_generation = n_neutrons_per_generation

        # Determine geometry type
        if isinstance(geometry, Geometry1DSlab):
            self.geometry_type = "1d"
        elif isinstance(geometry, Geometry2DPlane):
            self.geometry_type = "2d"
        else:
            self.geometry_type = "1d"

    def single_simulation(
        self,
        *,
        geometry: Optional[Geometry1DSlab | Geometry2DPlane] = None,
        initial_source_positions: Optional[list[tuple]] = None,
        n_generations: Optional[int] = None,
        n_neutrons_per_generation: Optional[int] = None,
        skip_generations: int = 10,
        _rng: Optional[Generator] = None,
        **kwargs,
    ) -> float:
        r"""
        Run power iteration to estimate k-effective.

        Parameters
        ----------
        geometry : Geometry1DSlab or Geometry2DPlane, optional
            Geometry to use (overrides instance geometry).
        initial_source_positions : list[tuple], optional
            Initial source positions (overrides instance value).
        n_generations : int, optional
            Number of generations (overrides instance value).
        n_neutrons_per_generation : int, optional
            Neutrons per generation (overrides instance value).
        skip_generations : int, default 10
            Number of initial generations to skip for convergence.
        _rng : Generator, optional
            Random number generator.
        **kwargs
            Additional arguments (ignored).

        Returns
        -------
        float
            Estimated k-effective value.
        """
        rng = self._rng(_rng, self.rng)

        # Use instance values if not overridden
        geom = geometry or self.geometry
        src_positions = initial_source_positions or self.initial_source_positions
        n_gen = n_generations or self.n_generations
        n_per_gen = n_neutrons_per_generation or self.n_neutrons_per_generation

        # Create default geometry if none provided
        if geom is None:
            u235 = Material.create_u235()
            geom = Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235])

        # Determine geometry type
        if isinstance(geom, Geometry1DSlab):
            geom_type = "1d"
        else:
            geom_type = "2d"

        # Initialize fission bank with source positions
        fission_bank = []
        for pos in src_positions:
            if geom_type == "1d":
                fission_bank.append((pos[0], 0.0, 2))  # (x, y, energy_group)
            else:
                x = pos[0]
                y = pos[1] if len(pos) > 1 else 0.0
                fission_bank.append((x, y, 2))

        # Replicate to reach target population
        while len(fission_bank) < n_per_gen:
            fission_bank.extend(fission_bank[: n_per_gen - len(fission_bank)])
        fission_bank = fission_bank[:n_per_gen]

        k_eff_values = []

        # Power iteration
        for gen in range(n_gen):
            next_fission_bank = []
            tallies = NeutronTallies.create()

            # Track all neutrons in this generation
            for x, y, energy_group in fission_bank:
                if geom_type == "1d":
                    neutron = NeutronState(
                        x=x,
                        mu=_sample_scatter_angle(rng, "1d"),
                        energy_group=energy_group,
                    )
                    reason, final = _track_neutron_1d(neutron, geom, tallies, rng)
                else:
                    neutron = NeutronState(
                        x=x,
                        y=y,
                        mu=_sample_scatter_angle(rng, "2d"),
                        energy_group=energy_group,
                    )
                    reason, final = _track_neutron_2d(neutron, geom, tallies, rng)

                # If fission occurred, create secondary neutrons
                if reason == "fission":
                    material = (
                        geom.get_material(final.x, final.y)
                        if geom_type == "2d"
                        else geom.get_material(final.x)
                    )
                    if material:
                        nu = material.get_nu(final.energy_group)
                        n_secondaries = int(nu) + (1 if rng.random() < (nu - int(nu)) else 0)

                        for _ in range(n_secondaries):
                            # Sample energy from fission spectrum
                            new_energy = _sample_fission_energy(material.fission_spectrum, rng)
                            next_fission_bank.append((final.x, final.y, new_energy))

            # Compute k-eff for this generation
            k_gen = len(next_fission_bank) / len(fission_bank) if len(fission_bank) > 0 else 0.0

            if gen >= skip_generations:
                k_eff_values.append(k_gen)

            # Prepare for next generation
            if len(next_fission_bank) == 0:
                # System died out
                break

            # Normalize population to target
            if len(next_fission_bank) > n_per_gen:
                # Randomly sample down
                indices = rng.choice(len(next_fission_bank), n_per_gen, replace=False)
                fission_bank = [next_fission_bank[i] for i in indices]
            elif len(next_fission_bank) < n_per_gen:
                # Replicate up
                while len(next_fission_bank) < n_per_gen:
                    next_fission_bank.append(next_fission_bank[rng.integers(0, len(next_fission_bank))])
                fission_bank = next_fission_bank[:n_per_gen]
            else:
                fission_bank = next_fission_bank  # pragma: no cover

        # Return average k-eff (excluding initial skip generations)
        if len(k_eff_values) > 0:
            return float(np.mean(k_eff_values))
        else:
            return 0.0
