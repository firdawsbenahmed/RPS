import numpy as np

import matplotlib.pyplot as plt 


Moves = ["Rock" , "Paper" , "Scissors"] 

## the rules 
R = np.array(
     [[0 , -1 , 1],
     [1 , 0 , -1],
     [-1 , 1 , 0]] 
     )

RN = 2 # nb of rounds 

def run_hedge(T , LS = 0.05) : 
# prob of each move for each player
    P1 = np.array([0.6 , 0.3 , 0.1])
    P2 = np.array([0.2 , 0.5 , 0.3])
    p1_hist , p2_hist = [P1.copy()] , [P2.copy()]

    for _ in range(RN) : 
       rp1 = R @ P2  ## is player 1 scores showing how well that move does against player 2 current strategy
       rp2 = -(P1 @ R)  ## how well each of player 2 moves does against player 1 current strategy  ; when player 1 chosses the rows the player 2 is chosing the columns this is why we inverses the order of the multiplucation 
# Moves that scored better get a higher probability next round.     
       P1_new = P1 * np.exp( LS * rp1)
       P2_new = P2 * np.exp( LS * rp2)
 ## That is why the two players learn from each other bcz it uses eachother proba
 # normalisation
       P1 = P1_new / P1_new.sum()
       P2 = P2_new / P2_new.sum()
       p1_hist.append(P1.copy())
       p2_hist.append(P2.copy())
    return np.array(p1_hist) ,np.array(p2_hist)
# here players learn which moves would have scored well against thier opp current strategy rhen play those more and repeat.