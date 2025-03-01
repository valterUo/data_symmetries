import random
import math
from typing import List, Tuple, Dict

class EfficientSymmetricGroup:
    """
    A class for efficiently sampling from symmetric groups and generating
    orbital-structured datasets without enumerating the full group.
    """

    def __init__(self, n: int):
        """
        Initialize a symmetric group S_n.

        Args:
            n: The degree of the symmetric group
        """
        self.n = n
        self.identity = tuple(range(1, n + 1))

    def random_permutation(self) -> Tuple[int, ...]:
        """
        Generate a random permutation from S_n without enumerating the full group.

        Returns:
            A random permutation in one-line notation
        """
        elements = list(range(1, self.n + 1))
        random.shuffle(elements)
        return tuple(elements)

    def random_permutation_with_cycle_structure(self, cycle_lengths: List[int]) -> Tuple[int, ...]:
        """
        Generate a random permutation with a specified cycle structure.

        Args:
            cycle_lengths: List of cycle lengths that should sum to n

        Returns:
            A random permutation with the specified cycle structure
        """
        if sum(cycle_lengths) != self.n:
            raise ValueError(f"Cycle lengths must sum to {self.n}")

        # Start with identity permutation
        perm = list(range(1, self.n + 1))

        # Track which elements have been used
        unused = set(range(1, self.n + 1))

        # For each cycle length
        for length in cycle_lengths:
            if length <= 1 or not unused:
                continue  # Skip trivial cycles or if no elements left

            # Select random elements for this cycle
            cycle_elements = random.sample(list(unused), length)

            # Remove selected elements from unused set
            for elem in cycle_elements:
                unused.remove(elem)

            # Create cycle: each element maps to the next, last to first
            for i in range(length - 1):
                perm[cycle_elements[i] - 1] = cycle_elements[i + 1]
            perm[cycle_elements[-1] - 1] = cycle_elements[0]

        return tuple(perm)

    def to_cycle_notation(self, perm: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        Convert a permutation in one-line notation to cycle notation.

        Args:
            perm: A permutation in one-line notation

        Returns:
            The permutation in cycle notation as a list of cycles (orbitals)
        """
        result = []
        visited = set()

        for i in range(1, self.n + 1):
            if i not in visited:
                cycle = []
                current = i

                while current not in visited:
                    visited.add(current)
                    cycle.append(current)
                    # Find where current maps to in the permutation
                    current = perm[current - 1]

                if len(cycle) > 1:  # Don't include 1-cycles (fixed points)
                    result.append(tuple(cycle))

        return result if result else [(1,)]  # Return identity if no non-trivial cycles

    def generate_angular_dataset(self, cycles: List[Tuple[int, ...]], min_val=-math.pi, max_val=math.pi) -> Dict[int, float]:
        """
        Generate a dataset with angular values between min_val and max_val for each cycle.

        Args:
            cycles: List of cycles/orbitals
            min_val: Minimum value for the angular range (default: -π)
            max_val: Maximum value for the angular range (default: π)

        Returns:
            Dictionary mapping positions to angular values
        """
        result = {}
        for cycle in cycles:
            # Generate one random angle for this cycle
            angle = random.uniform(min_val, max_val)
            # Assign this same angle to all positions in the cycle
            for pos in cycle:
                result[pos] = angle

        # Handle fixed points (elements not in any cycle)
        all_elements = set(range(1, self.n + 1))
        elements_in_cycles = set(pos for cycle in cycles for pos in cycle)
        fixed_points = all_elements - elements_in_cycles

        # Assign random angles to fixed points
        for pos in fixed_points:
            result[pos] = random.uniform(min_val, max_val)

        return result

    def generate_structured_angular_dataset(self, num_samples: int = 1, min_val=-math.pi, max_val=math.pi) -> List[Dict[int, float]]:
        """
        Generate multiple structured datasets where angular values respect orbital structure.

        Args:
            num_samples: Number of dataset samples to generate
            min_val: Minimum value for the angular range (default: -π)
            max_val: Maximum value for the angular range (default: π)

        Returns:
            List of dictionaries, each mapping positions to angular values
        """
        datasets = []
        for _ in range(num_samples):
            # Generate a random permutation
            perm = self.random_permutation()
            # Convert to cycle notation
            cycles = self.to_cycle_notation(perm)
            # Generate angular values that respect cycle structure
            dataset = self.generate_angular_dataset(cycles, min_val, max_val)
            datasets.append((perm, cycles, dataset))
        return datasets

    def generate_dataset_with_custom_cycle_structure(self, cycle_structure: List[int], min_val=-math.pi, max_val=math.pi) -> Dict[int, float]:
        """
        Generate a dataset with angular values based on a specific cycle structure.

        Args:
            cycle_structure: List of cycle lengths (must sum to n)
            min_val: Minimum value for the angular range (default: -π)
            max_val: Maximum value for the angular range (default: π)

        Returns:
            Dictionary mapping positions to angular values
        """
        # Generate a permutation with the desired cycle structure
        perm = self.random_permutation_with_cycle_structure(cycle_structure)
        # Convert to cycle notation
        cycles = self.to_cycle_notation(perm)
        # Generate angular values that respect cycle structure
        dataset = self.generate_angular_dataset(cycles, min_val, max_val)

        return perm, cycles, dataset

    def generate_k_cycles_dataset(self, k: int, num_samples: int = 1, min_val=-math.pi, max_val=math.pi) -> List[Dict[int, float]]:
        """
        Generate datasets where permutations contain exactly k non-trivial cycles.

        Args:
            k: Number of cycles to include
            num_samples: Number of dataset samples to generate
            min_val: Minimum value for the angular range
            max_val: Maximum value for the angular range

        Returns:
            List of dictionaries with the generated datasets
        """
        if k > self.n // 2:
            raise ValueError(f"Cannot have {k} non-trivial cycles with only {self.n} elements")

        datasets = []
        for _ in range(num_samples):
            # Try to partition n into k cycles (plus possibly some fixed points)
            remaining = self.n
            cycle_lengths = []

            # Make sure we have at least k cycles of length >= 2
            for i in range(k):
                if remaining < 2:
                    break

                # Generate random cycle length (at least 2)
                if i == k - 1:
                    # Last cycle gets remaining elements (must be at least 2)
                    length = remaining
                else:
                    max_length = min(remaining - (2 * (k - i - 1)), remaining // 2)
                    if max_length < 2:
                        break
                    length = random.randint(2, max_length)

                cycle_lengths.append(length)
                remaining -= length

            # Add fixed points if any elements remain
            cycle_lengths.extend([1] * remaining)

            if len([l for l in cycle_lengths if l >= 2]) == k:
                # Generate dataset with this cycle structure
                perm, cycles, dataset = self.generate_dataset_with_custom_cycle_structure(cycle_lengths, min_val, max_val)
                datasets.append((perm, cycles, dataset))

        return datasets