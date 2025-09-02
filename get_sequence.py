import pickle
from environments.cube2 import Cube2State, Cube2

results_file = "./results/cube2/results.pkl"

data = pickle.load(open(results_file, "rb"))

print(data.keys())

head_len = 1

for h in range(head_len):
    print(f"=== Solution #{h+1} ===")
    print(data['states'][h].colors) # Scramble된 상태
    print(data['solutions'][h])     # Action Sequence
    print([p.colors for p in data['paths'][h]]) # State Sequence
    print(data['times'][h])         # 걸린 시간


    