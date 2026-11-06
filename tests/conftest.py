import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "realdata: needs cached real IBL windows under data/phase5/windows; skipped (reason 'missing dataset') when absent")
