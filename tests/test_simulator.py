"""Unit tests verifying all Priority 1 simulator invariants and correctness."""

import math
import unittest

from ems.ambulance import Ambulance, AmbulanceStatus
from ems.city import City, Location
from ems.config import CityConfig
from ems.demand import EmergencyCall
from ems.simulator import AmbulanceSimulator


class TestSimulatorInvariants(unittest.TestCase):
    """Test discrete-event simulator against the 10 operational invariants."""

    def setUp(self):
        self.config = CityConfig()

    def test_invariant_1_event_ordering_and_clock_advancement(self):
        """Invariant 1: Service completions at time t must be processed before calls at time t,

        and clock advances monotonically.
        """
        sim = AmbulanceSimulator(config=self.config, seed=42)

        # Mark all ambulances busy
        for amb in sim.ambulances:
            amb.status = AmbulanceStatus.AT_HOSPITAL
            amb.assigned_call_id = 100
            amb.available_at_min = 30.0

        # One ambulance finishes handover at exactly t = 30.0
        amb0 = sim.ambulances[0]
        sim._schedule_event(
            30.0,
            "finish_handover",
            priority=0,
            ambulance_id=amb0.ambulance_id,
        )

        call = EmergencyCall(
            call_id=1,
            arrival_time_min=30.0,
            location=(amb0.location.x, amb0.location.y),
        )

        # Run day with custom call at t = 30.0 without resetting ambulance statuses
        outcomes = sim.run_day(calls=[call], reset=False)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(sim.dropped_calls, 0)
        self.assertEqual(outcomes[0].ambulance_id, amb0.ambulance_id)

    def test_invariant_2_no_dispatch_while_busy_or_relocating(self):
        """Invariant 2: An ambulance cannot be dispatched while busy or relocating."""
        sim = AmbulanceSimulator(config=self.config, seed=100)

        # Make 9 ambulances busy
        for amb in sim.ambulances[1:]:
            amb.status = AmbulanceStatus.RESPONDING
            amb.assigned_call_id = 999

        # Make the last ambulance relocating
        amb0 = sim.ambulances[0]
        amb0.status = AmbulanceStatus.REPOSITIONING
        amb0.available_at_min = 100.0

        call = EmergencyCall(call_id=1, arrival_time_min=10.0, location=(5.0, 5.0))
        success = sim._dispatch(call)
        self.assertFalse(success)
        self.assertEqual(len(sim.outcomes), 0)

    def test_invariant_3_ambulance_availability_and_location_consistency(self):
        """Invariant 3: An ambulance's availability and location are consistent at every event."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        amb = sim.ambulances[0]
        home_loc = amb.location

        self.assertTrue(amb.is_available(0.0))
        self.assertEqual(amb.status, AmbulanceStatus.IDLE)
        self.assertIsNone(amb.assigned_call_id)

        # Assign call
        amb.assign_call(1)
        self.assertFalse(amb.is_available(0.0))
        self.assertEqual(amb.status, AmbulanceStatus.RESPONDING)

        # Arrive scene
        scene_loc = Location(6.0, 6.0)
        amb.location = scene_loc
        amb.status = AmbulanceStatus.AT_SCENE
        self.assertFalse(amb.is_available(10.0))

        # Finish call
        amb.finish_call(scene_loc, current_time_min=25.0)
        self.assertTrue(amb.is_available(25.0))
        self.assertFalse(amb.is_available(20.0))
        self.assertEqual(amb.location, scene_loc)

    def test_invariant_4_dropped_calls_counted_when_unserved(self):
        """Invariant 4: Calls are counted as dropped only when the dispatch rule cannot serve them."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        # All ambulances busy
        for amb in sim.ambulances:
            amb.status = AmbulanceStatus.AT_SCENE
            amb.assigned_call_id = 999

        call = EmergencyCall(call_id=1, arrival_time_min=5.0, location=(5.0, 5.0))
        outcomes = sim.run_day(calls=[call], reset=False)
        self.assertEqual(len(outcomes), 0)
        self.assertEqual(sim.dropped_calls, 1)

    def test_invariant_5_response_time_measured_from_call_arrival_to_scene(self):
        """Invariant 5: Response time is measured from call arrival to ambulance arrival at incident."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        # Location away from any base so travel time > 0
        call = EmergencyCall(call_id=42, arrival_time_min=15.0, location=(4.0, 4.0))
        sim.current_time_min = 15.0

        success = sim._dispatch(call)
        self.assertTrue(success)
        outcome = sim.outcomes[-1]
        self.assertEqual(outcome.call_id, 42)
        self.assertEqual(outcome.arrival_time_min, 15.0)
        # Verify response time includes dispatch delay + positive travel time
        self.assertGreater(outcome.response_time_min, sim.config.dispatch_delay_min)
        # Verify arrive_scene event time matches call arrival + response time
        arrive_event = [e for e in sim._events if e.event_type == "arrive_scene"][0]
        self.assertAlmostEqual(arrive_event.time_min, 15.0 + outcome.response_time_min, places=6)

    def test_invariant_6_documented_drop_policy(self):
        """Invariant 6: Explicit documented loss system: immediate drop when all units are busy."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        for amb in sim.ambulances:
            amb.status = AmbulanceStatus.AT_HOSPITAL
            amb.assigned_call_id = 50

        call1 = EmergencyCall(call_id=1, arrival_time_min=10.0, location=(1.0, 1.0))
        call2 = EmergencyCall(call_id=2, arrival_time_min=11.0, location=(2.0, 2.0))
        sim.run_day(calls=[call1, call2], reset=False)
        self.assertEqual(sim.dropped_calls, 2)
        self.assertEqual(len(sim.outcomes), 0)

    def test_invariant_7_service_cycle_and_return_to_service(self):
        """Invariant 7: Transport, hospital handover, and return-to-service times are accounted for."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        # Force transport probability to 1.0
        sim.config.transport_probability = 1.0

        call = EmergencyCall(call_id=1, arrival_time_min=0.0, location=(4.0, 4.0))
        outcomes = sim.run_day(calls=[call])
        self.assertEqual(len(outcomes), 1)

        # Find the ambulance that served the call
        amb = sim.ambulances[outcomes[0].ambulance_id]
        # At the end of the day, ambulance must have completed hospital handover and returned to service (IDLE)
        self.assertEqual(amb.status, AmbulanceStatus.IDLE)
        self.assertIsNone(amb.assigned_call_id)
        # It must be located at a hospital
        self.assertTrue(any(amb.location.is_close_to(hosp) for hosp in sim.city.hospitals))

    def test_invariant_8_repositioning_invariants(self):
        """Invariant 8: Repositioning cannot teleport ambulances or incorrectly mark them available."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        amb = sim.ambulances[0]
        start_loc = amb.location
        target_base = 5

        success = sim.relocate_ambulance(amb.ambulance_id, target_base)
        self.assertTrue(success)
        self.assertEqual(amb.status, AmbulanceStatus.REPOSITIONING)
        self.assertFalse(amb.is_available(sim.current_time_min))
        # Location during transit must not be the destination
        self.assertEqual(amb.location, start_loc)
        self.assertGreater(amb.available_at_min, sim.current_time_min)

        # Cannot relocate an ambulance that is already relocating
        success_again = sim.relocate_ambulance(amb.ambulance_id, target_base)
        self.assertFalse(success_again)

    def test_invariant_9_reproducibility(self):
        """Invariant 9: Running the same configuration and seed is reproducible."""
        sim1 = AmbulanceSimulator(config=self.config, seed=77)
        out1 = sim1.run_day()

        sim2 = AmbulanceSimulator(config=self.config, seed=77)
        out2 = sim2.run_day()

        self.assertEqual(len(out1), len(out2))
        self.assertEqual(sim1.dropped_calls, sim2.dropped_calls)
        self.assertEqual(sim1.relocation_count, sim2.relocation_count)
        self.assertAlmostEqual(sim1.relocation_minutes, sim2.relocation_minutes, places=5)
        for o1, o2 in zip(out1, out2):
            self.assertEqual(o1.call_id, o2.call_id)
            self.assertAlmostEqual(o1.response_time_min, o2.response_time_min, places=5)

    def test_invariant_10_simulator_reset(self):
        """Invariant 10: The simulator resets all episode state correctly."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        out1 = sim.run_day()
        drops1 = sim.dropped_calls

        sim.reset(seed=42)
        self.assertEqual(sim.current_time_min, 0.0)
        self.assertEqual(sim.dropped_calls, 0)
        self.assertEqual(sim.relocation_count, 0)
        self.assertEqual(len(sim.outcomes), 0)
        for amb in sim.ambulances:
            self.assertEqual(amb.status, AmbulanceStatus.IDLE)
            self.assertEqual(amb.location, sim.city.bases[amb.home_base_id])

        out2 = sim.run_day(seed=42)
        self.assertEqual(len(out1), len(out2))
        self.assertEqual(drops1, sim.dropped_calls)


if __name__ == "__main__":
    unittest.main()
