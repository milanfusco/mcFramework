from unittest.mock import Mock

import numpy as np

from mcframework.core import MonteCarloSimulation, SimulationResult


class TestMocking:
    """Test components in isolation with mocks"""

    def test_simulation_with_mocked_rng(self):
        """Test simulation with mocked random number generator"""

        class MockRNG:
            def normal(self, *args, **kwargs):
                return 5.0

        class TestSim(MonteCarloSimulation):
            def single_simulation(self, **kwargs):
                return float(self.rng.normal())

        sim = TestSim()
        sim.rng = MockRNG()

        result = sim.single_simulation()
        assert result == 5.0

    def test_framework_with_mocked_simulation(self, framework):
        """Test framework with mocked simulation"""
        mock_sim = Mock(spec=MonteCarloSimulation)
        mock_sim.name = "MockSim"

        mock_result = SimulationResult(
            results=np.array([1.0, 2.0, 3.0]),
            n_simulations=3,
            execution_time=1.0,
            mean=2.0,
            std=1.0,
            percentiles={50: 2.0}
        )
        mock_sim.run.return_value = mock_result

        framework.register_simulation(mock_sim)
        result = framework.run_simulation("MockSim", 100)

        mock_sim.run.assert_called_once()
        assert result.mean == 2.0

    def test_progress_callback_mock(self, simple_simulation):
        """Test progress callback receives correct values"""
        callback = Mock()

        simple_simulation.run(
            100,
            parallel=False,
            progress_callback=callback,
            compute_stats=False
        )

        # Should be called with (completed, total)
        assert callback.call_count > 0
        last_call = callback.call_args_list[-1]
        assert last_call[0] == (100, 100)
