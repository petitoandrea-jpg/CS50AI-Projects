import math
import random
import time


class Nim():

    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Inizializza la partita con una determinata configurazione di pile.
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Restituisce l'insieme di mosse consentite (pila, quantità) per lo stato corrente.
        """
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        """
        Determina l'avversario dato il giocatore corrente.
        """
        return 0 if player == 1 else 1

    def switch_player(self):
        """
        Alterna il turno tra i due partecipanti.
        """
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        Applica la mossa scelta, verifica se la partita è conclusa
        e passa il turno all'avversario.
        """
        pile, count = action

        # Controlli di validità della mossa
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Aggiorna lo stato delle pile
        self.piles[pile] -= count
        self.switch_player()

        # Chi compie l'ultima mossa lasciando tutte le pile vuote perde
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Inizializza l'agente Q-learning con parametri di apprendimento ed esplorazione.
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Aggiorna la tabella Q sulla base della transizione di stato e della ricompensa.
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        """
        Recupera il valore Q per la coppia (stato, azione). Restituisce 0 se assente.
        """
        state_key = tuple(state)
        return self.q.get((state_key, action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Aggiorna il valore Q applicando la formula di Bellman:
        Q(s, a) = Q(s, a) + alpha * (reward + future_rewards - Q(s, a))
        """
        state_key = tuple(state)
        new_estimate = reward + future_rewards
        self.q[(state_key, action)] = old_q + self.alpha * (new_estimate - old_q)

    def best_future_reward(self, state):
        """
        Trova la ricompensa Q massima tra tutte le azioni legali nello stato successivo.
        """
        actions = Nim.available_actions(state)
        if not actions:
            return 0

        best_reward = -math.inf
        for action in actions:
            q_val = self.get_q_value(state, action)
            if q_val > best_reward:
                best_reward = q_val

        return best_reward

    def choose_action(self, state, epsilon=True):
        """
        Seleziona un'azione disponibile impiegando la strategia epsilon-greedy.
        """
        actions = list(Nim.available_actions(state))
        if not actions:
            return None

        # Strategia di esplorazione casuale con probabilità epsilon
        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)

        # Scelta greedy: individua le azioni con il valore Q più alto
        best_reward = -math.inf
        best_actions = []

        for action in actions:
            q_val = self.get_q_value(state, action)
            if q_val > best_reward:
                best_reward = q_val
                best_actions = [action]
            elif q_val == best_reward:
                best_actions.append(action)

        return random.choice(best_actions)


def train(n):
    """
    Addestra l'agente facendolo giocare n partite contro se stesso.
    """
    ai = NimAI()

    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        # Memorizza l'ultima mossa compiuta da ciascun partecipante
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        while True:
            state = game.piles.copy()
            action = ai.choose_action(game.piles)

            last[game.player]["state"] = state
            last[game.player]["action"] = action

            game.move(action)
            new_state = game.piles.copy()

            if game.winner is not None:
                # Chi vince riceve +1, chi perde subisce -1
                ai.update(state, action, new_state, -1)
                ai.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1
                )
                break

            elif last[game.player]["state"] is not None:
                ai.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )

    print("Done training")
    return ai


def play(ai, human_player=None):
    """
    Consente a un essere umano di sfidare l'agente addestrato.
    """
    if human_player is None:
        human_player = random.randint(0, 1)

    game = Nim()

    while True:
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        game.move((pile, count))

        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
