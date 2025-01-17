import numpy as np
from typing import List, Tuple, Optional
from enum import Enum, auto
import time
from copy import deepcopy

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

    def display(self):
        """
        Affiche l'état du plateau avec les piles représentées horizontalement.
        Format : [BA-BC] signifie BA en bas, BC au-dessus
        """
        print("\n   0      1      2      3      4   ")
        print("  " + "─" * 35)
        
        for i in range(5):
            row = [f"{i} |"]
            
            for j in range(5):
                stack = self.board[i][j]
                if not stack:
                    row.append("  .   ")
                else:
                    # Représente la pile avec des tirets entre les pièces
                    stack_str = "-".join(str(p) for p in stack)
                    # Ajoute des espaces pour aligner
                    row.append(f" {stack_str:<5}")
            
            row.append("|")
            print(" ".join(row))
        
        print("  " + "─" * 35)

def simulate_game():
    """
    Simule une partie complète avec des mouvements aléatoires
    """
    state = GameState()
    state.initialize_game()
    
    while True:
        state.display()
        
        # Déterminer les pièces disponibles pour le joueur actuel
        pieces = state.blue_pieces if state.current_player else state.red_pieces
        player = "Bleu" if state.current_player else "Rouge"
        
        # Vérifier s'il y a une pièce obligatoire à déplacer (règle de retraite)
        if state.must_move_piece:
            piece = state.must_move_piece
            moves = state.get_legal_moves(piece)
            if not moves:
                print(f"Aucun mouvement possible pour la pièce obligatoire {piece}")
                state.must_move_piece = None
                continue
            print(f"Le joueur {player} doit déplacer {piece}")
        else:
            # Sélectionner une pièce au hasard avec des mouvements valides
            valid_pieces = [p for p in pieces if state.get_legal_moves(p)]
            if not valid_pieces:
                print(f"Le joueur {player} n'a pas de mouvement valide !")
                return not state.current_player
            piece = np.random.choice(valid_pieces)
        
        # Sélectionner un mouvement au hasard
        moves = state.get_legal_moves(piece)
        move = moves[np.random.randint(len(moves))]
        
        print(f"Le joueur {player} déplace {piece} vers {move}")
        state.make_move(piece, move)
        
        # Vérifier la victoire
        winner = state.check_victory()
        if winner is not None:
            state.display()
            print(f"Le joueur {'Bleu' if winner else 'Rouge'} a gagné !")
            return winner
        
        #time.sleep(1)  # Pause pour mieux suivre le jeu

if __name__ == "__main__":
    print("Simulation d'une partie des Tacticiens de Brême")
    winner = simulate_game()