from pprint import pformat
from typing import Literal

import numpy as np
from graphviz import Digraph

from src.structs import HiddenState, KnownVariables
from src.utils import Float, Matrix

MIN_PROBABILITY = 0.01


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
        np.set_printoptions(formatter={"float": "{: 0.2f}".format})
        return pformat(
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

    def random_walk(self, prev_state: int) -> int:
        """
        Computes the next state in the random walk.

        Parameters:
            prev_state: The current state of the chain.

        Returns:
            The next state in the random walk.
        """
        return np.random.choice(range(len(self.states)), p=self.transitions[prev_state])

    def to_image(self, filename: str) -> None:
        """
        Creates a graph image of the Markov Chain.

        Parameters:
            filename: The name of the file to save the image to.
        """
        graph = Digraph("Markov Chain", filename=filename, format="png")
        graph.attr(rankdir="LR", size="8,5")

        if self.states.size == 1:
            state_prob = self.states.item()
            graph.node("0", label=f"{state_prob:.2f}")
            graph.node("1", label=f"{1 - state_prob:.2f}")
        else:
            for i, state in enumerate(self.states):
                graph.node(str(i), label=f"{state:.2f}")

        for i, row in enumerate(self.transitions):
            for j, transition in enumerate(row):
                graph.edge(str(i), str(j), label=f"{transition:.2f}")

        graph.render(f"data/results/{filename}")

    def to_image_with_known_var(
        self, filename: str, known_var_markov_chain: "MarkovChain"
    ):
        """
        Creates a graph image of the Markov Chain, with a linked known variable Markov Chain.

        Parameters:
            filename: The name of the file to save the image to.
            known_var_markov_chain: The known variable Markov Chain to link to.
        """
        graph = Digraph("Markov Chain", filename=filename, format="png")
        graph.attr(rankdir="LR", size="50")

        with graph.subgraph(name="hidden_graph") as c:  # type: ignore
            c.attr(color="invis")
            if self.states.size == 1:
                state_prob = self.states.item()
                c.node("0", label=f"{state_prob:.2f}")
                c.node("1", label=f"{1 - state_prob:.2f}")
            else:
                for i, state in enumerate(self.states):
                    c.node(str(i), label=f"{HiddenState(i).name} - {state:.2f}")

            for i, row in enumerate(self.transitions):
                for j, transition in enumerate(row):
                    if transition < MIN_PROBABILITY:
                        c.edge(str(i), str(j), color="darkgrey", style="dotted")
                    else:
                        c.edge(str(i), str(j), label=f"h-{transition:.2f}")

        n_hidden_states = self.states.size if self.states.size > 1 else 2

        with graph.subgraph(name="known_var") as c:  # type: ignore
            c.attr(color="invis")
            for i, state in enumerate(known_var_markov_chain.states):
                c.node(
                    str(i + n_hidden_states),
                    label=f"{KnownVariables(i).name} - {state:.2f}",
                )

            for i, row in enumerate(known_var_markov_chain.transitions):
                for j, transition in enumerate(row):
                    if transition < MIN_PROBABILITY:
                        c.edge(
                            str(i),
                            str(j + n_hidden_states),
                            color="tomato",
                            style="dotted",
                        )
                    else:
                        c.edge(
                            str(i),
                            str(j + n_hidden_states),
                            label=f"k-{transition:.2f}",
                            color="red",
                        )

        graph.render(f"data/results/{filename}")
