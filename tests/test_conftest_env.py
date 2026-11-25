# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The conftest rule that turns off the instruction the simulator does not support."""

from conftest import apply_simulator_env

NAME = "TT_METAL_DISABLE_SFPLOADMACRO"


def test_simulator_sets_the_variable():
    env = {"TT_METAL_SIMULATOR": "/x/libttsim.so"}
    apply_simulator_env(env)
    assert env[NAME] == "1"


def test_no_simulator_leaves_it_unset():
    for env in ({}, {"TT_METAL_SIMULATOR": ""}):
        apply_simulator_env(env)
        assert NAME not in env


def test_an_existing_value_is_kept():
    env = {"TT_METAL_SIMULATOR": "/x/libttsim.so", NAME: "0"}
    apply_simulator_env(env)
    assert env[NAME] == "0"
