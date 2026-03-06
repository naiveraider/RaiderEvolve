from abc import ABC, abstractmethod
import copy
import random


class BaseProblem(ABC):
    """Base for evolutionary problems. Supports multiple candidate generation strategies."""

    @abstractmethod
    def initialize_population(self, pop_size):
        pass

    @abstractmethod
    def mutate(self, candidate):
        pass

    def mutate_perturb(self, candidate):
        """Randomly perturb parameters (e.g. pivot selection). Default uses mutate()."""
        return self.mutate(candidate)

    def get_templates(self):
        """Return list of predefined template names/code fragments for template-based mutation."""
        return []

    def mutate_from_template(self, candidate):
        """Randomly replace (parts of) candidate from a template. Override in subclass."""
        templates = self.get_templates()
        if not templates:
            return self.mutate_perturb(candidate)
        name = random.choice(templates)
        child = copy.deepcopy(candidate)
        self._apply_template(child, name)
        return child

    def _apply_template(self, candidate, template_name):
        """Apply a template by name to candidate. Override in subclass."""
        pass

    def mutate_swap(self, candidate):
        """Simple mutation: e.g. swap two lines of code or two parameter choices. Override in subclass."""
        return self.mutate_perturb(candidate)

    def improve_with_llm(self, candidate):
        """Optional: prompt-based LLM improvement. Override in subclass; default returns copy."""
        return copy.deepcopy(candidate)

    @abstractmethod
    def evaluate(self, candidate):
        pass

    @abstractmethod
    def serialize_solution(self, candidate):
        pass