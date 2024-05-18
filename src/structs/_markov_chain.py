from pprint import pformat
from typing import Literal

import numpy as np
from graphviz import Digraph

from src.utils import Float, Matrix


class MarkovChain:
    """A markov chain is a stochastic model describing a sequence of possible events
    in which the probability of each event depends only on the state attained in the previous event.
    """

    def __init__(
        self,
        states: Matrix[Literal["N"], Float],
        transitions: Matrix[Literal["N N"], Float],
        initial_state: int,
    ) -> None:
        """
        Args:
            states: The initial probabilities of being in each state.
            transitions: The probabilities of transitioning from one state to another.
            initial_state: The initial state of the chain.
        """
        self.states = states
        self.transitions = transitions
        self.initial_state = initial_state

    def __str__(self) -> str:
        np.set_printoptions(precision=2, linewidth=200)
        return "MarkovChain: " + pformat(
            {
                "states": self.states,
                "transitions": self.transitions,
                "initial_state": self.initial_state,
            },
            indent=2,
            compact=False,
            sort_dicts=False,
        )

    def nsteps(self, n: int) -> Matrix[Literal["N"], Float]:
        """
        Computes the probabilities of being in every state after n steps.

        Parameters:
            n: number of steps to compute the probabilities for.

        Returns:
            The probabilities of being in every state after n steps."""
        return self.states @ self.transitions**n

    def to_image(self, filename: str) -> None:
        """
        Creates a graph image of the Markov Chain.

        Parameters:
            filename: The name of the file to save the image to.
        """
        graph = Digraph("Markov Chain", filename=filename, format="png")
        graph.attr(rankdir="LR", size="8,5")

        for i, state in enumerate(self.states):
            graph.node(str(i), label=f"{state:.2f}")

        for i, row in enumerate(self.transitions):
            for j, transistion in enumerate(row):
                graph.edge(str(i), str(j), label=f"{transistion:.2f}")

        graph.render(f"data/results/{filename}")

    def to_image_with_known_var(
        self, filename: str, known_var_markov_chain: "MarkovChain"
    ):
        """
        Creates a graph image of the Markov Chain, with a linked known variable Markov Chain.

        Parameters:
            filename: The name of the file to save the image to.
        """
        graph = Digraph("Markov Chain", filename=filename, format="png")
        graph.attr(rankdir="LR", size="8,5")

        for i, state in enumerate(self.states):
            graph.node(str(i), label=f"H{i}-{state:.2f}")

        for i, row in enumerate(self.transitions):
            for j, transistion in enumerate(row):
                graph.edge(str(i), str(j), label=f"h-{transistion:.2f}")

        n_hidden_states = len(self.states)
        for i, state in enumerate(known_var_markov_chain.states):
            graph.node(str(i + n_hidden_states), label=f"K{i}-{state:.2f}")

        for i, row in enumerate(known_var_markov_chain.transitions):
            for j, transistion in enumerate(row):
                graph.edge(
                    str(i), str(j + n_hidden_states), label=f"k-{transistion:.2f}"
                )

        graph.render(f"data/results/{filename}")
