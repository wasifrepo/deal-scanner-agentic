from agents.agent import Agent


class SpecialistAgent(Agent):
    """
    An Agent that runs our fine-tuned LLM that's running remotely on Modal
    """

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        """
        Set up this Agent by creating an instance of the modal class
        """
        self.log("Specialist Agent is initializing - connecting to modal")
        try:
            import modal
        except ImportError as exc:
            raise RuntimeError(
                "Modal SDK is required for SpecialistAgent. Install dependencies with "
                "`uv sync --extra modal`."
            ) from exc
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()

    def price(self, description: str) -> float:
        """
        Make a remote call to return the estimate of the price of this item
        """
        self.log("Specialist Agent is calling remote fine-tuned model")
        result = self.pricer.price.remote(description)
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result
