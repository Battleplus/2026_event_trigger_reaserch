"""Pure helper checks; these do not import or initialize Isaac Sim."""
from run_capture import segment_step_time, suffix_horizon, stable_drop


def test_horizon_and_time_boundary():
    assert suffix_horizon(1200) == 1800
    assert suffix_horizon(5400) == 1
    assert suffix_horizon(5401) == 0


def test_segment_step_time_preserves_resume_decision_boundary():
    assert segment_step_time(1200, 0, skip_prepublic_first_step=True) == 1200
    assert segment_step_time(1200, 1, skip_prepublic_first_step=True) == 1201
    assert segment_step_time(1200, 0) == 1201
def test_suffix_draws_are_shared_within_suffix_and_can_differ_across_suffixes():
    # A suffix seed is one scalar shared by both branches; changing suffix seed
    # is allowed to change future time-addressed dropout outcomes.
    assert stable_drop(730201, "uav-0-0", 1210, 0.5) == stable_drop(730201, "uav-0-0", 1210, 0.5)
    assert any(stable_drop(730201, "uav-0-0", t, 0.5) != stable_drop(730202, "uav-0-0", t, 0.5)
               for t in range(1200, 1220))
