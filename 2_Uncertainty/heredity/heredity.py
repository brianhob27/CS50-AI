import csv
import itertools
import sys
import copy

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probabilities = {}
    people_here = copy.copy(people)
    while len(people_here):
        people_copy = copy.copy(people_here)
        for p in people_copy:
            if people_copy[p]["mother"] is None and people_copy[p]["father"] is None:
                gene = 0
                trait = False
                if p in two_genes:
                    gene = 2
                elif p in one_gene:
                    gene = 1
                if p in have_trait:
                    trait = True
                probabilities[p] = PROBS["gene"][gene] * PROBS["trait"][gene][trait]
                people_here.pop(p)

            elif people_copy[p]["mother"] in probabilities and people_copy[p]["father"] in probabilities:
                temp_prob = 0
                temp_num = 0
                father = people_copy[p]["father"]
                father_prob_no = 0
                father_prob_yes = 0
                mother = people_copy[p]["mother"]
                mother_prob_no = 0
                mother_prob_yes = 0

                if father in two_genes:
                    father_prob_no = PROBS["mutation"]
                    father_prob_yes = 1 - father_prob_no
                elif father in one_gene:
                    father_prob_no = 0.5
                    father_prob_yes = 1 - father_prob_no
                else:
                    father_prob_no = 1 - PROBS["mutation"]
                    father_prob_yes = 1 - father_prob_no

                if mother in two_genes:
                    mother_prob_no = PROBS["mutation"]
                    mother_prob_yes = 1 - mother_prob_no
                elif mother in one_gene:
                    mother_prob_no = 0.5
                    mother_prob_yes = 1 - mother_prob_no
                else:
                    mother_prob_no = 1 - PROBS["mutation"]
                    mother_prob_yes = 1 - mother_prob_no

                if p in two_genes:
                    temp_prob = father_prob_yes * mother_prob_yes
                    temp_num = 2
                elif p in one_gene:
                    temp_prob = (father_prob_yes * mother_prob_no) + (father_prob_no * mother_prob_yes)
                    temp_num = 1
                else:
                    temp_prob = father_prob_no * mother_prob_no
                    temp_num = 0

                if p in have_trait:
                    probabilities[p] = PROBS["trait"][temp_num][True] * temp_prob
                else:
                    probabilities[p] = PROBS["trait"][temp_num][False] * temp_prob
                people_here.pop(p)
    probability = 1

    for i in probabilities:
        probability = probability * probabilities[i]

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for people in probabilities:
        if people in two_genes:
            probabilities[people]["gene"][2] += p
        elif people in one_gene:
            probabilities[people]["gene"][1] += p
        else:
            probabilities[people]["gene"][0] += p

        if people in have_trait:
            probabilities[people]["trait"][True] += p
        else:
            probabilities[people]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for p in probabilities:
        gene_add = probabilities[p]["gene"][2] + probabilities[p]["gene"][1] + probabilities[p]["gene"][0]
        gene_multi = 1 / gene_add
        for g in probabilities[p]["gene"]:
            probabilities[p]["gene"][g] = probabilities[p]["gene"][g] * gene_multi

        trait_add = probabilities[p]["trait"][True] + probabilities[p]["trait"][False]
        trait_multi = 1 / trait_add
        for t in probabilities[p]["trait"]:
            probabilities[p]["trait"][t] = probabilities[p]["trait"][t] * trait_multi


if __name__ == "__main__":
    main()