import csv
import itertools
import sys

PROBS = {

    # Probabilità a priori sulle copie del gene (senza informazioni sui genitori)
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {
        2: {
            True: 0.65,
            False: 0.35
        },
        1: {
            True: 0.56,
            False: 0.44
        },
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Probabilità di mutazione genetica durante la trasmissione
    "mutation": 0.01
}


def main():

    # Verifica l'utilizzo da riga di comando
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Inizializza il dizionario delle probabilità per ogni individuo
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

    # Itera su tutti i possibili scenari compatibili con i tratti noti
    names = set(people)
    for have_trait in powerset(names):

        # Controlla se la combinazione del tratto rispetta le evidenze note
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Itera su tutte le assegnazioni possibili di geni
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Calcola la probabilità congiunta per questo scenario
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Normalizza le probabilità affinché la somma sia 1
    normalize(probabilities)

    # Mostra i risultati finali
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Carica i dati dal file CSV strutturato con: name, mother, father, trait.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] if row["mother"] else None,
                "father": row["father"] if row["father"] else None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Restituisce tutti i possibili sottoinsiemi dell'insieme s fornito.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Calcola e restituisce la probabilità congiunta dello scenario specificato.
    """
    joint_prob = 1.0

    for person, info in people.items():
        if person in two_genes:
            gene_count = 2
        elif person in one_gene:
            gene_count = 1
        else:
            gene_count = 0

        has_trait = person in have_trait

        # Calcolo probabilità gene
        if info["mother"] is None and info["father"] is None:
            p_gene = PROBS["gene"][gene_count]
        else:
            def pass_prob(parent):
                if parent in two_genes:
                    return 1.0 - PROBS["mutation"]
                elif parent in one_gene:
                    return 0.5
                else:
                    return PROBS["mutation"]

            p_mother = pass_prob(info["mother"])
            p_father = pass_prob(info["father"])

            if gene_count == 2:
                p_gene = p_mother * p_father
            elif gene_count == 1:
                p_gene = (p_mother * (1.0 - p_father)) + ((1.0 - p_mother) * p_father)
            else:
                p_gene = (1.0 - p_mother) * (1.0 - p_father)

        # Calcolo probabilità tratto
        p_trait = PROBS["trait"][gene_count][has_trait]

        joint_prob *= (p_gene * p_trait)

    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Aggiunge la probabilità congiunta p alla distribuzione di ciascuna persona.
    """
    for person in probabilities:
        if person in two_genes:
            gene_count = 2
        elif person in one_gene:
            gene_count = 1
        else:
            gene_count = 0

        has_trait = person in have_trait

        probabilities[person]["gene"][gene_count] += p
        probabilities[person]["trait"][has_trait] += p


def normalize(probabilities):
    """
    Normalizza le distribuzioni di probabilità di geni e tratti per ciascun individuo.
    """
    for person in probabilities:
        gene_sum = sum(probabilities[person]["gene"].values())
        if gene_sum > 0:
            for count in probabilities[person]["gene"]:
                probabilities[person]["gene"][count] /= gene_sum

        trait_sum = sum(probabilities[person]["trait"].values())
        if trait_sum > 0:
            for status in probabilities[person]["trait"]:
                probabilities[person]["trait"][status] /= trait_sum


if __name__ == "__main__":
    main()
