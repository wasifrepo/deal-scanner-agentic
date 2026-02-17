from agents.agent import Agent
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.preprocessor import Preprocessor


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW
    FRONTIER_WEIGHT = 0.9
    SPECIALIST_WEIGHT = 0.1

    def __init__(self, collection):
        """
        Create an ensemble from the two online estimators:
        1) Frontier model + RAG
        2) Specialist fine-tuned model on Modal
        """
        self.log("Initializing Ensemble Agent")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.preprocessor = Preprocessor()
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        """
        Run this ensemble model
        Ask each model to estimate the product price
        Then return the weighted blend.
        :param description: the description of a product
        :return: an estimate of its price
        """
        self.log("Running Ensemble Agent - preprocessing text")
        rewrite = self.preprocessor.preprocess(description)
        self.log(f"Pre-processed text using {self.preprocessor.model_name}")
        specialist = self.specialist.price(rewrite)
        frontier = self.frontier.price(rewrite)
        combined = (
            frontier * self.FRONTIER_WEIGHT
            + specialist * self.SPECIALIST_WEIGHT
        )
        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")
        return combined
