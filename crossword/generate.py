import sys
from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Inizializza il risolutore CSP associando il cruciverba e i domini di partenza.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Costruisce una matrice bidimensionale con i caratteri assegnati.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Stampa la griglia del cruciverba a video.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Genera l'immagine del cruciverba completato.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Risolve il cruciverba applicando consistenza unaria, binaria e backtracking.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Rimuove dai domini tutte le parole che non rispettano la lunghezza della variabile.
        """
        for var in self.crossword.variables:
            for word in set(self.domains[var]):
                if len(word) != var.length:
                    self.domains[var].remove(word)

    def revise(self, x, y):
        """
        Rende l'arco x coerente con y rimuovendo i valori incompatibili nel dominio di x.
        """
        revised = False
        overlap = self.crossword.overlaps.get((x, y))

        if overlap is None:
            return False

        i, j = overlap
        for x_val in set(self.domains[x]):
            if not any(x_val[i] == y_val[j] for y_val in self.domains[y] if x_val != y_val):
                self.domains[x].remove(x_val)
                revised = True

        return revised

    def ac3(self, arcs=None):
        """
        Esegue l'algoritmo AC-3 per garantire la consistenza degli archi nel sistema.
        """
        if arcs is None:
            queue = []
            for v1 in self.crossword.variables:
                for v2 in self.crossword.neighbors(v1):
                    queue.append((v1, v2))
        else:
            queue = list(arcs)

        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for z in self.crossword.neighbors(x):
                    if z != y:
                        queue.append((z, x))

        return True

    def assignment_complete(self, assignment):
        """
        Verifica se tutte le variabili del problema sono state assegnate.
        """
        return len(assignment) == len(self.crossword.variables)

    def consistent(self, assignment):
        """
        Verifica che l'assegnazione soddisfi unicità, lunghezze e sovrapposizioni.
        """
        assigned_words = list(assignment.values())

        if len(assigned_words) != len(set(assigned_words)):
            return False

        for var, word in assignment.items():
            if len(word) != var.length:
                return False

            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    i, j = self.crossword.overlaps[var, neighbor]
                    if word[i] != assignment[neighbor][j]:
                        return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Ordina i valori del dominio secondo l'euristica Least-Constraining Values.
        """
        unassigned_neighbors = [
            n for n in self.crossword.neighbors(var)
            if n not in assignment
        ]

        def count_conflicts(val):
            count = 0
            for neighbor in unassigned_neighbors:
                i, j = self.crossword.overlaps[var, neighbor]
                for n_val in self.domains[neighbor]:
                    if val[i] != n_val[j]:
                        count += 1
            return count

        return sorted(list(self.domains[var]), key=count_conflicts)

    def select_unassigned_variable(self, assignment):
        """
        Seleziona la prossima variabile tramite MRV e degree heuristic come spareggio.
        """
        unassigned = [
            v for v in self.crossword.variables
            if v not in assignment
        ]

        return min(
            unassigned,
            key=lambda v: (len(self.domains[v]), -len(self.crossword.neighbors(v)))
        )

    def backtrack(self, assignment):
        """
        Ricerca a ritroso (Backtracking Search) per trovare un'assegnazione completa.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for val in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = val

            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result

        return None


def main():

    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
