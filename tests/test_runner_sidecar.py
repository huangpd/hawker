import threading
import time
from hawker_agent.agent.runner import _OBSERVER_THREADS, wait_for_observer_sidecars


def test_wait_for_observer_sidecars():
    # Make sure list is empty initially
    _OBSERVER_THREADS.clear()

    # Create a dummy thread that sleeps for 0.1 seconds
    def worker():
        time.sleep(0.1)

    t = threading.Thread(target=worker, daemon=True)
    _OBSERVER_THREADS.append(t)
    t.start()

    assert len(_OBSERVER_THREADS) == 1
    assert t.is_alive()

    start_time = time.time()
    wait_for_observer_sidecars()
    duration = time.time() - start_time

    # Thread should have finished and joined
    assert not t.is_alive()
    # It should have taken at least 0.1 seconds
    assert duration >= 0.1
    # The list should be cleared
    assert len(_OBSERVER_THREADS) == 0

    # Calling it again when empty should be safe
    wait_for_observer_sidecars()
    assert len(_OBSERVER_THREADS) == 0
