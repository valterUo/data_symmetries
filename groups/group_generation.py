from sage.all import SymmetricGroup
import json

def subgroup_generators_by_order(n, zero_indexed=False, max_size=None):
    """
    Return a dict mapping subgroup order -> list of subgroups,
    where each subgroup is represented by a list of its generators,
    and each generator is the permutation image list [g(1),...,g(n)]
    (1-based by default; set zero_indexed=True for 0-based).

    Parameters
    ----------
    n : int
        Symmetric group size (S_n).
    zero_indexed : bool
        If True, outputs permutations in 0-based form.
    max_size : int or None
        Maximum number of subgroups to keep per order.
        If None, include all.
    """
    G = SymmetricGroup(n)
    result = {}

    for H in G.subgroups():
        order = int(H.order())
        # If we already reached the cap for this order, skip
        if max_size is not None and len(result.get(order, [])) >= max_size:
            continue

        gens_as_lists = []
        for g in H.gens():
            # Build the image list [g(1), g(2), ..., g(n)]
            perm = [int(g(i)) for i in range(1, n+1)]
            if zero_indexed:
                perm = [p-1 for p in perm]
            gens_as_lists.append(perm)

        result.setdefault(order, []).append(gens_as_lists)

    return result


# Example: S4
for i in range(4, 5):
    
    subs = subgroup_generators_by_order(i, max_size = 30)

    with open(f"subgroup_generators_{i}.json", "w") as f:
        json.dump(subs, f, indent=4)