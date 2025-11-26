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
    "plot_flux_map_2d",
    "_sample_distance",
    "_sample_scatter_angle",
    "_sample_fission_energy",
    "_sample_interaction_type",
]


# ============================================================================
# Energy Groups (simplified 3-group structure)
# ============================================================================
ENERGY_GROUPS = {
    "thermal": 0.025,      # 0.025 eV
    "epithermal": 1.0,     # 1 eV  
    "fast": 2.0e6,         # 2 MeV
}

ENERGY_GROUP_NAMES = ["thermal", "epithermal", "fast"]
ENERGY_GROUP_VALUES = [0.025, 1.0, 2.0e6]



# ============================================================================
# Helpers
# ============================================================================

def _flux_bin_index(
    x: float,
    x_min: float,
    x_max: float,
    n_bins: int,
) -> int:
    """Compute clamped spatial flux bin index for a position x.
    
    Parameters
    ----------
    x : float
        Position.
    x_min : float
        Minimum position.
    x_max : float
        Maximum position.
    n_bins : int
        Number of bins.

    Returns
    -------
    int
        Clamped spatial flux bin index for a position x.
    """
    if n_bins <= 1 or x_max <= x_min:
        return 0
    frac = (x - x_min) / (x_max - x_min)
    idx = int(frac * n_bins)
    return max(0, min(idx, n_bins - 1))

def _bin_segments_2d(segments, geom, nx, ny, flux_map):
    """
    Add track-length contributions from segments into flux bins.

    Parameters
    ----------
    segments : list of (x0,y0,x1,y1)
    flux_map : (ny,nx) array to accumulate into
    """

    x_min, x_max = geom.x_min, geom.x_max
    y_min, y_max = geom.y_min, geom.y_max
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    for x0, y0, x1, y1 in segments:
        # Total segment length
        seg_len = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

        # Midpoint bin (approximate contribution)
        xm = 0.5*(x0 + x1)
        ym = 0.5*(y0 + y1)

        ix = int((xm - x_min) / dx)
        iy = int((ym - y_min) / dy)

        if 0 <= ix < nx and 0 <= iy < ny:
            flux_map[iy, ix] += seg_len

def plot_flux_map_2d(flux_map: np.ndarray, geometry: Geometry2DPlane, filename: str = "flux_2d.png") -> None:
        """
        Plot a 2D flux map.

        Parameters
        ----------
        flux_map : np.ndarray
        geometry : Geometry2DPlane
        filename : str
            The filename to save the plot to.
        """
        import matplotlib.pyplot as plt

        ny, nx = flux_map.shape
        plt.figure(figsize=(8, 7))
        plt.imshow(
            flux_map,
            origin="lower",
            extent=[geometry.x_min, geometry.x_max, geometry.y_min, geometry.y_max],
            cmap="inferno",
            aspect="equal"
        )
        plt.colorbar(label="Flux (Track-Length)")
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.title("2D Neutron Flux Map (Track-Length Estimator)")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Saved: {filename}")
        plt.close()


# ============================================================================
# Material model
# ============================================================================

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
       \sigma_{\text{total}} = \sigma_{\text{scatter}} +
                               \sigma_{\text{absorption}} +
                               \sigma_{\text{fission}}
    """

    name: str
    sigma_total: list[float]
    sigma_scatter: list[float]
    sigma_absorption: list[float]
    sigma_fission: list[float]
    nu: list[float]
    fission_spectrum: list[float]

    def __post_init__(self) -> None:
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

    # Lightweight accessors to keep interface explicit
    def get_sigma_total(self, energy_group: int) -> float:
        return self.sigma_total[energy_group]

    def get_sigma_scatter(self, energy_group: int) -> float:
        return self.sigma_scatter[energy_group]

    def get_sigma_absorption(self, energy_group: int) -> float:
        return self.sigma_absorption[energy_group]

    def get_sigma_fission(self, energy_group: int) -> float:
        return self.sigma_fission[energy_group]

    def get_nu(self, energy_group: int) -> float:
        return self.nu[energy_group]

    # Convenience constructors for demos / defaults
    @staticmethod
    def create_u235() -> Material:
        """Create U-235 fuel material with toy but plausible cross sections."""
        return Material(
            name="U235",
            sigma_total=[8.0, 4.0, 5.0],      # cm^-1
            sigma_scatter=[2.0, 1.5, 2.5],    # cm^-1
            sigma_absorption=[3.0, 1.0, 0.5], # cm^-1
            sigma_fission=[3.0, 1.5, 2.0],    # cm^-1
            nu=[2.5, 2.5, 2.5],               # neutrons per fission
            fission_spectrum=[0.1, 0.2, 0.7], # mostly fast
        )

    @staticmethod
    def create_water() -> Material:
        """Create H2O moderator material."""
        return Material(
            name="H2O",
            sigma_total=[3.0, 2.0, 1.5],
            sigma_scatter=[2.8, 1.9, 1.4],
            sigma_absorption=[0.2, 0.1, 0.1],
            sigma_fission=[0.0, 0.0, 0.0],
            nu=[0.0, 0.0, 0.0],
            fission_spectrum=[0.0, 0.0, 0.0],
        )

    @staticmethod
    def create_graphite() -> Material:
        """Create graphite moderator material."""
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
        """Create void (vacuum) material."""
        return Material(
            name="Void",
            sigma_total=[0.0, 0.0, 0.0],
            sigma_scatter=[0.0, 0.0, 0.0],
            sigma_absorption=[0.0, 0.0, 0.0],
            sigma_fission=[0.0, 0.0, 0.0],
            nu=[0.0, 0.0, 0.0],
            fission_spectrum=[0.0, 0.0, 0.0],
        )


# ============================================================================
# Neutron state
# ============================================================================

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
    energy_group: int = 2  # start as fast
    weight: float = 1.0
    alive: bool = True



# ============================================================================
# Geometry Classes
# ============================================================================

class Geometry1DSlab:
    r"""
    One-dimensional slab geometry with multiple material regions.

    The slab extends from x_min to x_max with regions defined by boundaries.
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
            raise ValueError(
                "Number of materials must equal number of regions (len(boundaries) - 1)"
            )
        if self.boundaries[0] != x_min or self.boundaries[-1] != x_max:
            raise ValueError("boundaries must start with x_min and end with x_max")

    def get_material(self, x: float) -> Optional[Material]:
        """Get the material at position x, or None if outside geometry."""
        if x < self.x_min or x > self.x_max:
            return None

        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] <= x < self.boundaries[i + 1]:
                return self.materials[i]

        # Handle right boundary exactly
        if x == self.x_max:
            return self.materials[-1]

        return None

    def is_inside(self, x: float) -> bool:
        """Check if position x is inside the geometry."""
        return self.x_min <= x <= self.x_max

    def distance_to_boundary(self, x: float, mu: float) -> float:
        """Calculate distance to next slab boundary in direction mu."""
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
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        regions: list[tuple[float, float, float, float, Material]],
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
        """Get material at position (x, y), or None if outside geometry."""
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
        Calculate distance to geometry boundary in direction angle (radians).
        """
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        distances: list[float] = []

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

        positive = [d for d in distances if d > 1e-10]
        return min(positive) if positive else float("inf")


# ============================================================================
# Tally System
# ============================================================================

@dataclass
class NeutronTallies:
    """
    Accumulator for neutron transport statistics.

    Attributes
    ----------
    flux : np.ndarray
        Track-length flux per spatial bin.
    """

    flux: Optional[np.ndarray] = None
    absorption_count: int = 0
    leakage_count: int = 0
    fission_count: int = 0
    scatter_count: int = 0
    total_tracks: int = 0
    fission_neutrons_produced: int = 0
    energy_spectrum: Optional[list[int]] = None  # Count per energy group

    def __post_init__(self) -> None:
        if self.flux is None:
            self.flux = np.zeros(100)
        if self.energy_spectrum is None:
            self.energy_spectrum = [0, 0, 0]

    @staticmethod
    def create(flux_bins: int = 100) -> NeutronTallies:
        """Create a new tally object with specified number of flux bins."""
        return NeutronTallies(
            flux=np.zeros(flux_bins),
            energy_spectrum=[0, 0, 0],
        )

    def record_flux(self, path_length: float, bin_index: int) -> None:
        """Record flux contribution in a spatial bin."""
        if 0 <= bin_index < len(self.flux):
            self.flux[bin_index] += path_length

    def record_absorption(self) -> None:
        self.absorption_count += 1

    def record_leakage(self) -> None:
        self.leakage_count += 1

    def record_fission(self, nu: float) -> None:
        self.fission_count += 1
        self.fission_neutrons_produced += int(nu)

    def record_scatter(self) -> None:
        self.scatter_count += 1

    def record_track(self) -> None:
        self.total_tracks += 1

    def get_k_eff_estimate(self) -> float:
        """
        Estimate k-effective from tallies via:
            k_eff ≈ (neutrons produced) / (neutrons lost)
        """
        if self.total_tracks == 0:
            return 0.0
        denom = self.absorption_count + self.leakage_count
        if denom == 0:
            return 0.0
        return float(self.fission_neutrons_produced) / float(denom)


# ============================================================================
# Physics Sampling Functions
# ============================================================================

def _sample_distance(sigma_total: float, rng: Generator) -> float:
    r"""
    Sample free-flight distance from exponential distribution.

    P(s) = Σ_t exp(-Σ_t s)
    """
    if sigma_total <= 0.0:
        return float("inf")
    return -np.log(rng.random()) / sigma_total


def _sample_scatter_angle(rng: Generator, dimension: Literal["1d", "2d"]) -> float:
    """
    Sample isotropic scattering angle.

    Returns
    -------
    float
        Direction cosine (1D) or angle in radians (2D).
    """
    if dimension == "1d":
        return 2.0 * rng.random() - 1.0  # uniform in [-1, 1]
    else:
        return 2.0 * np.pi * rng.random()  # angle in [0, 2π]


def _sample_fission_energy(fission_spectrum: list[float], rng: Generator) -> int:
    """Sample energy group for a fission neutron from a discrete spectrum."""
    cumulative = np.cumsum(fission_spectrum)
    xi = rng.random()
    for i, cum_prob in enumerate(cumulative):
        if xi <= cum_prob:
            return i
    return len(fission_spectrum) - 1


def _sample_interaction_type(
    material: Material,
    energy_group: int,
    rng: Generator,
) -> Literal["scatter", "absorption", "fission"]:
    """
    Sample the type of neutron interaction.

    Returns
    -------
    str
        "scatter", "absorption", or "fission".
    """
    sigma_s = material.get_sigma_scatter(energy_group)
    sigma_a = material.get_sigma_absorption(energy_group)
    sigma_f = material.get_sigma_fission(energy_group) # noqa: F841
    sigma_t = material.get_sigma_total(energy_group)

    if sigma_t <= 0.0:
        return "absorption"

    xi = rng.random()

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

    Returns
    -------
    (reason, final_state)
        reason ∈ {"absorption", "fission", "leakage", "max_collisions"}.
    """
    tallies.record_track()
    n_collisions = 0

    while neutron.alive and n_collisions < max_collisions:
        material = geometry.get_material(neutron.x)

        if material is None:
            tallies.record_leakage()
            neutron.alive = False
            return "leakage", neutron

        sigma_t = material.get_sigma_total(neutron.energy_group)
        distance = _sample_distance(sigma_t, rng)

        dist_boundary = geometry.distance_to_boundary(neutron.x, neutron.mu)

        # Collision before boundary
        if distance < dist_boundary:
            neutron.x += neutron.mu * distance

            bin_idx = _flux_bin_index(
                neutron.x, geometry.x_min, geometry.x_max, len(tallies.flux)
            )
            tallies.record_flux(distance, bin_idx)

            interaction = _sample_interaction_type(material, neutron.energy_group, rng)

            if interaction == "scatter":
                tallies.record_scatter()
                neutron.mu = _sample_scatter_angle(rng, "1d")
                # simple downscatter heuristic
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

        # Boundary before collision
        else:
            neutron.x += neutron.mu * dist_boundary

            bin_idx = _flux_bin_index(
                neutron.x, geometry.x_min, geometry.x_max, len(tallies.flux)
            )
            tallies.record_flux(dist_boundary, bin_idx)

            if geometry.boundary_condition == "vacuum":
                tallies.record_leakage()
                neutron.alive = False
                return "leakage", neutron
            elif geometry.boundary_condition == "reflective":
                neutron.mu = -neutron.mu
                neutron.x += neutron.mu * 1e-8  # nudge off boundary

    neutron.alive = False
    return "max_collisions", neutron


def _track_neutron_2d(
    neutron: NeutronState,
    geometry: Geometry2DPlane,
    tallies: NeutronTallies,
    rng: Generator,
    max_collisions: int = 2000,
):
    """
    Track a neutron in 2D and return every straight-line segment.

    Returns
    -------
    (reason, final_state, segments)
    where segments = list[(x0, y0, x1, y1)]
    """

    segments = []
    tallies.record_track()
    n_collisions = 0

    while neutron.alive and n_collisions < max_collisions:

        material = geometry.get_material(neutron.x, neutron.y)
        if material is None:
            tallies.record_leakage()
            neutron.alive = False
            return "leakage", neutron, segments

        x0, y0 = neutron.x, neutron.y

        sigma_t = material.get_sigma_total(neutron.energy_group)
        distance = _sample_distance(sigma_t, rng)
        boundary_dist = geometry.distance_to_boundary(neutron.x, neutron.y, neutron.mu)

        if distance < boundary_dist:
            # Move to collision point
            dx = distance * np.cos(neutron.mu)
            dy = distance * np.sin(neutron.mu)
            neutron.x += dx
            neutron.y += dy

            # Record the segment
            segments.append((x0, y0, neutron.x, neutron.y))

            # Interaction
            interaction = _sample_interaction_type(material, neutron.energy_group, rng)

            if interaction == "scatter":
                tallies.record_scatter()
                neutron.mu = _sample_scatter_angle(rng, "2d")
                if neutron.energy_group > 0 and rng.random() < 0.3:
                    neutron.energy_group -= 1

            elif interaction == "absorption":
                tallies.record_absorption()
                neutron.alive = False
                return "absorption", neutron, segments

            elif interaction == "fission":
                nu = material.get_nu(neutron.energy_group)
                tallies.record_fission(nu)
                neutron.alive = False
                return "fission", neutron, segments

            n_collisions += 1

        else:
            # Move to boundary
            dx = boundary_dist * np.cos(neutron.mu)
            dy = boundary_dist * np.sin(neutron.mu)
            neutron.x += dx
            neutron.y += dy

            segments.append((x0, y0, neutron.x, neutron.y))

            if geometry.boundary_condition == "vacuum":
                tallies.record_leakage()
                neutron.alive = False
                return "leakage", neutron, segments

            else:
                # Reflect angle
                neutron.mu = (neutron.mu + np.pi) % (2*np.pi)
                # tiny nudge
                neutron.x += 1e-8 * np.cos(neutron.mu)
                neutron.y += 1e-8 * np.sin(neutron.mu)

    neutron.alive = False
    return "max_collisions", neutron, segments



# ============================================================================
# Main Simulation Classes
# ============================================================================

class NeutronTransportSimulation(MonteCarloSimulation):
    r"""
    Monte Carlo neutron transport simulation for 1D and 2D geometries.

    Simulates individual neutron histories through specified geometry with
    energy-dependent cross sections. Computes flux distributions, absorption
    rates, and leakage probabilities.
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

    def _default_geometry(self) -> Geometry1DSlab:
        """Fallback geometry used when none is provided."""
        u235 = Material.create_u235()
        return Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235])

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
        """
        Simulate a single neutron history and return a scalar observable.

        Returns
        -------
        float
            - "flux": total flux (sum over all bins)
            - "leakage_prob": 1.0 if leaked, 0.0 otherwise
            - "absorption_prob": 1.0 if absorbed, 0.0 otherwise
        """
        rng = self._rng(_rng, self.rng)

        geom = geometry or self.geometry or self._default_geometry()
        src_pos = source_position or self.source_position
        src_energy = source_energy_group if source_energy_group is not None else self.source_energy_group
        n_bins = flux_bins or self.flux_bins

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

        tallies = NeutronTallies.create(n_bins)

        if geom_type == "1d":
            reason, _ = _track_neutron_1d(neutron, geom, tallies, rng)
        else:
            reason, _, _ = _track_neutron_2d(neutron, geom, tallies, rng)

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
        parallel: bool = False,  # reserved for future parallelization
    ) -> np.ndarray:
        """
        Compute spatial flux distribution over multiple histories.

        Returns
        -------
        np.ndarray
            Flux distribution normalized per history.
        """
        rng = self.rng
        geom = geometry or self.geometry or self._default_geometry()
        src_pos = source_position or self.source_position
        src_energy = source_energy_group if source_energy_group is not None else self.source_energy_group

        flux_total = np.zeros(flux_bins)

        for _ in range(n_histories):
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
    
    def compute_flux_map_2d_tracklength(
        self,
        nx: int = 60,
        ny: int = 60,
        n_histories=30000,
        geometry=None,
        source_position=None,
        source_energy_group=None,
    ):
        """
        Full 2D flux map using MCNP-style track-length estimator.

        Returns
        -------
        flux_map : np.ndarray (ny, nx)
        """

        geom = geometry or self.geometry
        if not isinstance(geom, Geometry2DPlane):
            raise ValueError("compute_flux_map_2d_tracklength requires a 2D geometry")

        src_pos = source_position or self.source_position
        src_energy = source_energy_group or self.source_energy_group
        rng = self.rng

        flux_map = np.zeros((ny, nx))

        for _ in range(n_histories):
            neutron = NeutronState(
                x=src_pos[0],
                y=src_pos[1],
                mu=_sample_scatter_angle(rng, "2d"),
                energy_group=src_energy,
            )

            # temporary tallies
            tallies = NeutronTallies()

            reason, final, segments = _track_neutron_2d(
                neutron, geom, tallies, rng
            )

            _bin_segments_2d(segments, geom, nx, ny, flux_map)

        return flux_map / n_histories

    




class KEFFSimulation(MonteCarloSimulation):
    r"""
    k-effective (criticality) calculation using power iteration.

    Computes the effective neutron multiplication factor k_eff using
    successive fission generations. Each generation tracks neutrons from
    fission sources to their next fission or termination.
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

    def _default_geometry(self) -> Geometry1DSlab:
        u235 = Material.create_u235()
        return Geometry1DSlab(0.0, 10.0, [0.0, 10.0], [u235])

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
        """
        Run power iteration to estimate k-effective for a single MC sample.
        """
        rng = self._rng(_rng, self.rng)

        geom = geometry or self.geometry or self._default_geometry()
        src_positions = initial_source_positions or self.initial_source_positions
        n_gen = n_generations if n_generations is not None else self.n_generations
        n_per_gen = (
            n_neutrons_per_generation
            if n_neutrons_per_generation is not None
            else self.n_neutrons_per_generation
        )

        geom_type: Literal["1d", "2d"] = (
            "1d" if isinstance(geom, Geometry1DSlab) else "2d"
        )

        # Initialize fission bank with source positions
        fission_bank: list[tuple[float, float, int]] = []
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

        k_eff_values: list[float] = []

        # Power iteration
        for gen in range(n_gen):
            next_fission_bank: list[tuple[float, float, int]] = []
            tallies = NeutronTallies.create()

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
                    reason, final, _segments = _track_neutron_2d(neutron, geom, tallies, rng)

                if reason == "fission":
                    material = (
                        geom.get_material(final.x, final.y)
                        if geom_type == "2d"
                        else geom.get_material(final.x)
                    )
                    if material:
                        nu = material.get_nu(final.energy_group)
                        n_secondaries = int(nu) + (
                            1 if rng.random() < (nu - int(nu)) else 0
                        )

                        for _ in range(n_secondaries):
                            new_energy = _sample_fission_energy(
                                material.fission_spectrum, rng
                            )
                            next_fission_bank.append(
                                (final.x, final.y, new_energy)
                            )

            k_gen = (
                len(next_fission_bank) / len(fission_bank)
                if len(fission_bank) > 0
                else 0.0
            )

            if gen >= skip_generations:
                k_eff_values.append(k_gen)

            if len(next_fission_bank) == 0:
                # System died out
                break

            # Normalize population to target
            if len(next_fission_bank) > n_per_gen:
                idx = rng.choice(len(next_fission_bank), n_per_gen, replace=False)
                fission_bank = [next_fission_bank[i] for i in idx]
            elif len(next_fission_bank) < n_per_gen:
                while len(next_fission_bank) < n_per_gen:
                    next_fission_bank.append(
                        next_fission_bank[rng.integers(0, len(next_fission_bank))]
                    )
                fission_bank = next_fission_bank[:n_per_gen]
            else:
                fission_bank = next_fission_bank

        return float(np.mean(k_eff_values)) if k_eff_values else 0.0

