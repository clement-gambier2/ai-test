import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
from copy import deepcopy
from enum import Enum, auto
import time

class PieceType(Enum):
    """Types de pièces disponibles dans le jeu, dans l'ordre décroissant de taille"""
    ANE = auto()    # Taille 4
    CHIEN = auto()  # Taille 3
    CHAT = auto()   # Taille 2
    COQ = auto()    # Taille 1


class Piece:
    """
    Représente une pièce du jeu avec ses caractéristiques
    """
    def __init__(self, piece_type: PieceType, color: bool, movement_type: str):
        self.type = piece_type
        self.is_blue = color  # True pour bleu, False pour rouge
        self.movement = movement_type
        # La taille est déterminée par la position dans l'enum (4 pour Âne à 1 pour Coq)
        self.size = 4 - list(PieceType).index(piece_type)
    
    def __str__(self):
        """Représentation textuelle d'une pièce : Couleur (B/R) + Type (A/D/C/Q)"""
        color = "B" if self.is_blue else "R"
        piece_symbols = {
            PieceType.ANE: "A",
            PieceType.CHIEN: "D",
            PieceType.CHAT: "C",
            PieceType.COQ: "Q"
        }
        return f"{color}{piece_symbols[self.type]}"

class GameState:
    """
    Représente l'état complet du jeu à un moment donné
    """
    def __init__(self):
        # Le plateau est une grille 5x5 de listes (pour les piles de pièces)
        self.board = [[[] for _ in range(5)] for _ in range(5)]
        self.blue_pieces: List[Piece] = []  # Pièces du joueur bleu
        self.red_pieces: List[Piece] = []   # Pièces du joueur rouge
        self.current_player = True  # True pour bleu, False pour rouge
        self.must_move_piece = None  # Pièce qui doit être déplacée (règle de retraite)
        
    def initialize_game(self):
        """Initialise le jeu avec les positions de départ"""
        # Définition des types de mouvement pour chaque pièce
        movements = {
            PieceType.ANE: "orthogonal",   # Mouvements horizontaux et verticaux
            PieceType.CHIEN: "diagonal",    # Mouvements en diagonale
            PieceType.CHAT: "L",           # Mouvements en L comme le cavalier aux échecs
            PieceType.COQ: "queen"         # Combinaison des mouvements orthogonaux et diagonaux
        }
        
        # Création des pièces pour chaque joueur
        for piece_type in PieceType:
            self.blue_pieces.append(Piece(piece_type, True, movements[piece_type]))
            self.red_pieces.append(Piece(piece_type, False, movements[piece_type]))
        
        # Placement initial des pièces sur le plateau
        for i, piece in enumerate(self.blue_pieces):
            self.board[0][i] = [piece]  # Ligne du haut pour bleu
        for i, piece in enumerate(self.red_pieces):
            self.board[4][i] = [piece]  # Ligne du bas pour rouge

    def get_legal_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        """
        Détermine tous les mouvements légaux possibles pour une pièce donnée
        """
        legal_moves = []
        
        # Trouver la position actuelle de la pièce
        current_pos = None
        for i in range(5):
            for j in range(5):
                if self.board[i][j] and piece in self.board[i][j]:
                    current_pos = (i, j)
                    break
            if current_pos:
                break
        
        if not current_pos:
            return []

        # Vérifier si la pièce est couverte par d'autres pièces
        stack = self.board[current_pos[0]][current_pos[1]]
        piece_index = stack.index(piece)
        if piece_index < len(stack) - 1:
            # La pièce est couverte, elle ne peut pas bouger
            return []

        # Définir les directions de mouvement selon le type de pièce
        if piece.movement == "orthogonal":
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        elif piece.movement == "diagonal":
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        elif piece.movement == "L":
            l_moves = []
            for dr in [-2, 2]:
                for dc in [-1, 1]:
                    l_moves.extend([(dr, dc), (dc, dr)])
            directions = l_moves
        else:  # queen
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                         (1, 1), (1, -1), (-1, 1), (-1, -1)]

        # Vérifier les mouvements possibles dans chaque direction
        for direction in directions:
            if piece.movement == "L":
                # Le chat se déplace en L (comme le cavalier aux échecs)
                new_pos = (current_pos[0] + direction[0], 
                          current_pos[1] + direction[1])
                if (0 <= new_pos[0] < 5 and 0 <= new_pos[1] < 5):
                    target = self.board[new_pos[0]][new_pos[1]]
                    # On ne peut s'empiler que sur une pièce plus grande
                    if not target or (target and target[-1].size > piece.size):
                        legal_moves.append(new_pos)
            else:
                # Pour les autres pièces, vérifier chaque case dans la direction
                for step in range(1, 5):
                    new_pos = (current_pos[0] + direction[0] * step,
                             current_pos[1] + direction[1] * step)
                    if not (0 <= new_pos[0] < 5 and 0 <= new_pos[1] < 5):
                        break
                    target = self.board[new_pos[0]][new_pos[1]]
                    if not target:
                        legal_moves.append(new_pos)
                    elif target[-1].size > piece.size:
                        legal_moves.append(new_pos)
                        break
                    else:
                        break

        return legal_moves
    
    def make_move(self, piece: Piece, target: Tuple[int, int]) -> bool:
        """
        Effectue un mouvement si celui-ci est légal
        Retourne True si le mouvement a été effectué, False sinon
        """
        # Trouver la position actuelle de la pièce
        current_pos = None
        for i in range(5):
            for j in range(5):
                if self.board[i][j] and piece in self.board[i][j]:
                    current_pos = (i, j)
                    break
            if current_pos:
                break

        if not current_pos:
            return False

        # Vérifier si le mouvement est légal
        if target not in self.get_legal_moves(piece):
            return False

        # Trouver l'index de la pièce dans sa pile
        stack = self.board[current_pos[0]][current_pos[1]]
        piece_index = stack.index(piece)
        
        # Déplacer la pièce et toutes celles au-dessus
        moving_pieces = stack[piece_index:]
        self.board[current_pos[0]][current_pos[1]] = stack[:piece_index]
        
        # Ajouter les pièces à leur nouvelle position
        if not self.board[target[0]][target[1]]:
            self.board[target[0]][target[1]] = moving_pieces
        else:
            self.board[target[0]][target[1]].extend(moving_pieces)

        # Vérifier la règle de retraite
        retreat_line = 0 if piece.is_blue else 4
        if target[0] == retreat_line:
            opponent_pieces = [p for p in moving_pieces if p.is_blue != piece.is_blue]
            if opponent_pieces:
                self.must_move_piece = max(opponent_pieces, key=lambda p: p.size)

        # Passer au joueur suivant
        self.current_player = not self.current_player
        return True

    def check_victory(self) -> Optional[bool]:
        """
        Vérifie s'il y a un gagnant.
        Retourne True si le joueur bleu gagne, False si le joueur rouge gagne, ou None si la partie continue.
        """
        for i in range(5):
            for j in range(5):
                stack = self.board[i][j]
                if len(stack) >= 4:  # Une pile de 4 pièces ou plus est trouvée
                    print(f"Pile gagnante sur la case ({i}, {j}): {[str(p) for p in stack]}")
                    top_piece = stack[-1]  # La pièce au sommet d’une pile détermine le gagnant
                    return top_piece.is_blue  # True pour bleu, False pour rouge
        return None  # Pas encore de gagnant

# Environnement pour les Tacticiens de Brême
class TacticiansEnv(gym.Env):
    def __init__(self):
        super(TacticiansEnv, self).__init__()
        self.state = GameState()
        self.state.initialize_game()

        # Définir les espaces d'observation et d'action
        self.observation_space = spaces.Box(low=-4, high=4, shape=(5, 5), dtype=np.int32)
        self.action_space = spaces.Discrete(25 * 25)  # Exemple : 625 actions possibles (source, destination)
    
    def reset(self):
        self.state = GameState()
        self.state.initialize_game()
        return self._get_observation()
    
    def step(self, action):
        # Traduire une action (id) en coordonnées (source et destination)
        src_x, src_y, dest_x, dest_y = self._decode_action(action)
        piece = self.state.board[src_x][src_y][-1] if self.state.board[src_x][src_y] else None
        if piece and piece.is_blue == self.state.current_player:
            valid_moves = self.state.get_legal_moves(piece)
            if (dest_x, dest_y) in valid_moves:
                self.state.make_move(piece, (dest_x, dest_y))

        reward = self._calculate_reward()
        done = self.state.check_victory() is not None
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        # Convertir l'état du jeu en une matrice
        obs = np.zeros((5, 5), dtype=np.int32)
        for i in range(5):
            for j in range(5):
                stack = self.state.board[i][j]
                if stack:
                    obs[i, j] = stack[-1].size if stack[-1].is_blue else -stack[-1].size
        return obs

    def _calculate_reward(self):
        winner = self.state.check_victory()
        if winner is not None:
            return 1 if winner == self.state.current_player else -1
        return 0  # Pas de victoire pour le moment

    def _decode_action(self, action):
        src = action // 25
        dest = action % 25
        return src // 5, src % 5, dest // 5, dest % 5

# Réseau neuronal pour DQL
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            # Remove the Flatten layer as the input is already flattened
            #nn.Flatten(),  
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Entraînement DQL
def train_dql(env, num_episodes=10, learning_rate=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    output_dim = env.action_space.n
    model = DQN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for episode in range(num_episodes):
        state = env.reset()
        # Flatten the state before converting to tensor
        state = torch.FloatTensor(state.flatten())  
        done = False
        total_reward = 0

        while not done:
            # Exploration vs Exploitation
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model(state)
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            # Flatten the next_state before converting to tensor
            next_state = torch.FloatTensor(next_state.flatten())  
            total_reward += reward

            # Mise à jour du modèle
            target = reward + gamma * torch.max(model(next_state)).item() * (1 - done)
            current_q = model(state)[action]
            loss = loss_fn(current_q, torch.tensor(target))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        epsilon = max(0.1, epsilon * epsilon_decay)
        print(f"Épisode {episode + 1}, Récompense Totale: {total_reward}")

    torch.save(model.state_dict(), "tacticians_dqn.pth")
    print("Modèle entraîné et sauvegardé.")

# Simulation
if __name__ == "__main__":
    env = TacticiansEnv()
    train_dql(env)
