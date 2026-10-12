import numpy as np

import matplotlib.pyplot as plt 


Moves = ["Rock" , "Paper" , "Scissors"] 

## the rules 
R = np.array(
     [[0 , -1 , 1],
     [1 , 0 , -1],
     [-1 , 1 , 0]] 
     )

RN = 2000 # nb of rounds 

def run_hedge(RN , LS = 0.05) : 
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

def run_funct_play(RN) : 
    SP1 = np.array([3.0 ,2.0 , 1.0]) ## a start to learn from
    SP2 = np.array([1.0 , 2.0 , 3.0 ])
    ## for history needeed later '
    hist1 = [ SP1 / SP1.sum()] 

    for _ in range(RN): 
        ## how often each player has played each move so far for each run probs 
        P1_seen = SP1 / SP1.sum()
        P2_seen = SP2 / SP2.sum()

        a1 = np.argmax(R @ P2_seen) # player 1's best reply to what player 2 has done
        a2 = np.argmax (-(P1_seen @ R))  # player 2's best reply to what player 1 has done

        SP1[a1] += 1 
        SP2[a2] += 1 ## so for each player actually plays that best move so its count goes up by 1

        hist1.append(SP1 / SP1.sum())

        return np.array(hist1)

## run 
p_hist, q_hist = run_hedge(RN)
# Running average of player 1's strategy over time
p_avg = np.cumsum(p_hist, axis=0) / np.arange(1, len(p_hist) + 1)[:, None]
fp_hist = run_funct_play(RN)
 
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
panels = [(p_hist, "Hedge: strategy at each round"),
          (p_avg, "Hedge: average strategy so far"),
          (fp_hist, "Fictitious play: how often each move was played")]
for ax, (data, title) in zip(axes, panels):
    for i, name in enumerate(Moves):
        ax.plot(data[:, i], label=name)
    ax.axhline(1 / 3, color="gray", linestyle="--", linewidth=1)  # equilibrium
    ax.set_title(title)
    ax.set_xlabel("Round")
axes[0].set_ylabel("Probability / frequency (player 1)")
axes[0].legend()
plt.tight_layout()
plt.savefig("rps_learning.png", dpi=150)
print("Hedge final strategy (P1):", np.round(p_hist[-1], 3))
print("Hedge average strategy (P1):", np.round(p_avg[-1], 3))
print("Fictitious play frequencies (P1):", np.round(fp_hist[-1], 3))




