"""Unit tests for baseline redeployment policies."""

import unittest

from ems.ambulance import AmbulanceStatus
from ems.city import City, Location
from ems.config import CityConfig
from ems.policies import (
    FixedHomeBasePolicy,
    GreedyCoveragePolicy,
    NearestBasePolicy,
    NoRedeploymentPolicy,
)
from ems.simulator import AmbulanceSimulator


class TestBaselinePolicies(unittest.TestCase):
    """Verify each baseline policy initiates intended relocation behaviors."""

    def setUp(self):
        self.config = CityConfig()
        self.city = City(self.config)

    def test_no_redeployment_policy(self):
        """NoRedeploymentPolicy must execute 0 relocations."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        policy = NoRedeploymentPolicy()
        sim.run_day(decision_callback=policy.decide)
        self.assertEqual(sim.relocation_count, 0)
        self.assertEqual(sim.relocation_minutes, 0.0)

    def test_fixed_home_base_policy(self):
        """FixedHomeBasePolicy relocates displaced ambulances back to their home bases."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        # Move ambulance 0 to a hospital
        amb0 = sim.ambulances[0]
        hospital = sim.city.hospitals[0]
        amb0.location = hospital
        amb0.status = AmbulanceStatus.IDLE

        policy = FixedHomeBasePolicy(sim.city)
        policy.decide(sim)

        # Ambulance 0 should have started repositioning to its home base
        self.assertEqual(sim.relocation_count, 1)
        self.assertEqual(amb0.status, AmbulanceStatus.REPOSITIONING)

    def test_nearest_base_policy(self):
        """NearestBasePolicy relocates ambulances stranded at hospitals to closest base."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        amb0 = sim.ambulances[0]
        hospital = sim.city.hospitals[0]  # (4.0, 4.0)
        amb0.location = hospital
        amb0.status = AmbulanceStatus.IDLE

        expected_base = sim.city.nearest_base(hospital)
        policy = NearestBasePolicy(sim.city)
        policy.decide(sim)

        self.assertEqual(sim.relocation_count, 1)
        self.assertEqual(amb0.status, AmbulanceStatus.REPOSITIONING)
        # Verify event destination matches the nearest base
        event = [e for e in sim._events if e.event_type == "finish_repositioning"][0]
        self.assertEqual(event.destination, sim.city.bases[expected_base])

    def test_greedy_coverage_policy(self):
        """GreedyCoveragePolicy relocates surplus unit when a demand deficit exists."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        # Put an ambulance at hospital
        amb0 = sim.ambulances[0]
        amb0.location = sim.city.hospitals[0]
        amb0.status = AmbulanceStatus.IDLE

        policy = GreedyCoveragePolicy(sim.city, min_deficit_threshold=0.01)
        policy.decide(sim)

        # Policy should find an uncovered high-demand base and relocate amb0 there
        self.assertGreaterEqual(sim.relocation_count, 1)
        self.assertEqual(amb0.status, AmbulanceStatus.REPOSITIONING)


if __name__ == "__main__":
    unittest.main()

