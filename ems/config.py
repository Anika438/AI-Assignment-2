
"""Configuration for the ambulance redeployment simulation."""

from dataclasses import dataclass, field


@dataclass
class CityConfig:
    """Parameters describing the simulated city and ambulance service."""

    # City geometry
    city_size_km: float = 12.0
    grid_size: int = 8

    # Candidate ambulance standby locations: (x, y), in kilometres
    bases: list[tuple[float, float]] = field(default_factory=lambda: [
        (1.5, 1.5),
        (6.0, 1.5),
        (10.5, 1.5),
        (3.0, 4.5),
        (7.5, 4.5),
        (10.5, 5.0),
        (1.5, 7.5),
        (5.0, 7.0),
        (9.0, 8.0),
        (3.0, 10.5),
        (6.5, 10.5),
        (10.5, 10.5),
    ])

    # Hospital coordinates
    hospitals: list[tuple[float, float]] = field(default_factory=lambda: [
        (4.0, 4.0),
        (9.0, 6.5),
        (3.5, 9.5),
    ])

    # Ambulance fleet
    num_ambulances: int = 10
    home_bases: list[int] = field(
        default_factory=lambda: [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    )

    # Travel assumptions
    emergency_speed_kmh: float = 36.0
    reposition_speed_kmh: float = 28.0
    road_distance_factor: float = 1.25
    travel_noise_sigma: float = 0.15
    dispatch_delay_min: float = 1.0

    # Time spent handling incidents
    scene_time_mean_min: float = 16.0
    scene_time_sigma: float = 0.35
    transport_probability: float = 0.75
    hospital_handover_mean_min: float = 14.0
    hospital_handover_sigma: float = 0.40

    # Emergency demand
    expected_calls_per_day: float = 150.0
    background_demand_weight: float = 0.15

    # Relative demand by hour of day
    hourly_profile: list[float] = field(default_factory=lambda: [
        0.75, 0.65, 0.55, 0.50, 0.48, 0.52,
        0.70, 0.95, 1.15, 1.25, 1.30, 1.30,
        1.28, 1.25, 1.22, 1.20, 1.22, 1.25,
        1.25, 1.20, 1.12, 1.05, 0.95, 0.85,
    ])

    # (x, y, spread in kilometres, hotspot name)
    hotspots: list[tuple[float, float, float, str]] = field(
        default_factory=lambda: [
            (8.5, 5.5, 1.6, "CBD"),
            (2.5, 9.0, 2.0, "Residential-NW"),
            (9.5, 10.0, 1.8, "Residential-NE"),
            (2.5, 2.0, 1.4, "Nightlife-SW"),
            (6.0, 1.5, 1.6, "Industrial-S"),
        ]
    )

    # Response-time target and decision schedule
    response_target_min: float = 8.0
    redeployment_interval_min: float = 30.0
    episode_duration_hours: float = 24.0


def hotspot_weights(hour: float) -> list[float]:
    """Return relative activity levels for hotspots at a given hour."""

    import math

    hour = hour % 24.0

    def daily_peak(center: float, width: float) -> float:
        distance = abs(hour - center)
        distance = min(distance, 24.0 - distance)
        return math.exp(-0.5 * (distance / width) ** 2)

    business_district = 0.15 + 1.6 * daily_peak(13.0, 3.5)
    residential_nw = (
        0.35
        + 0.9 * daily_peak(7.5, 2.0)
        + 1.0 * daily_peak(20.0, 2.5)
    )
    residential_ne = (
        0.35
        + 0.8 * daily_peak(8.0, 2.0)
        + 0.9 * daily_peak(19.5, 2.5)
    )
    nightlife = 0.10 + 1.8 * daily_peak(1.0, 2.5)
    industrial = 0.10 + 1.0 * daily_peak(11.0, 3.0)

    return [
        business_district,
        residential_nw,
        residential_ne,
        nightlife,
        industrial,
    ]
