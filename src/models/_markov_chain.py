from typing import Literal

from graphviz import Digraph

from src.utils import Float, Matrix


class MarkovChain:
    """A markov chain is a stochastic model describing a sequence of possible events
    in which the probability of each event depends only on the state attained in the previous event.
    """

    def __init__(
        self,
        states: Matrix[Literal["N"], Float],
        transistions: Matrix[Literal["N N"], Float],
        initial_state: int,
    ) -> None:
        """
        Args:
            states: The initial probabilities of being in each state.
            transistions: The probabilities of transitioning from one state to another.
            initial_state: The initial state of the chain.
        """
        self.states = states
        self.transistions = transistions
        self.initial_state = initial_state

    def nsteps(self, n: int) -> Matrix[Literal["N"], Float]:
        """
        Computes the probabilities of being in every state after n steps.

        Parameters:
            n: number of steps to compute the probabilities for.

        Returns:
            The probabilities of being in every state after n steps."""
        return self.states @ self.transistions**n

    def to_image(self, filename: str) -> None:
        """
        Creates a graph image of the Markov Chain.

        Parameters:
            filename: The name of the file to save the image to.
        """
        graph = Digraph("Markov Chain", filename=filename, format="png")
        graph.attr(rankdir="LR", size="8,5")

        for i, state in enumerate(self.states):
            graph.node(str(i), label=f"{state}")

        for i, row in enumerate(self.transistions):
            for j, transistion in enumerate(row):
                graph.edge(str(i), str(j), label=f"{transistion}")

        graph.render(f"data/results/{filename}")
