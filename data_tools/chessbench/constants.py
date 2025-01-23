# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Constants, interfaces, and types."""

import abc
from collections.abc import Mapping
from typing import Any, NamedTuple

from apache_beam import coders

import jaxtyping as jtp



# Integer sequences of token ids.
Sequences = jtp.UInt32[jtp.Array, 'B T']

# The predictions are log-probabilities (natural logarithm) for the passed
# sequences. It can either be marginal log-probabilities (i.e. log P(s) for all
# sequences s in the batch), or full conditionals (i.e. log P(token | s_<t) for
# all sequence s, time t and token in the alphabet).
Marginals = jtp.Float32[jtp.Array, '*B']
Conditionals = jtp.Float32[jtp.Array, '*B T F']
Predictions = Marginals | Conditionals

# True means the loss will be masked there, i.e. we ignore it.
LossMask = jtp.Bool[jtp.Array, 'B T']

CODERS = {
    'fen': coders.StrUtf8Coder(),
    'move': coders.StrUtf8Coder(),
    'count': coders.BigIntegerCoder(),
    'win_prob': coders.FloatCoder(),
}
CODERS['state_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['win_prob'],
))
CODERS['action_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
    CODERS['win_prob'],
))
CODERS['behavioral_cloning'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
))


class BehavioralCloningData(NamedTuple):
  fen: str
  move: str


class StateValueData(NamedTuple):
  fen: str
  win_prob: float


class ActionValueData(NamedTuple):
  fen: str
  move: str
  win_prob: float