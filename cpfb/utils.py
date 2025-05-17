import numpy as np
from moves import move_def
import itertools
fl_mask = 64
bl_mask = 32
dl_mask = 16
dfl_mask = 12
dbl_mask = 3
ce_masks = {
    "U": np.uint64(0b1000000000000000000000000000000000000000000000000000000000000),
    "L": np.uint64(0b0100000000000000000000000000000000000000000000000000000000000),
    "F": np.uint64(0b0010000000000000000000000000000000000000000000000000000000000),
    "R": np.uint64(0b0001000000000000000000000000000000000000000000000000000000000),
    "B": np.uint64(0b0000100000000000000000000000000000000000000000000000000000000),
    "D": np.uint64(0b0000010000000000000000000000000000000000000000000000000000000)
}
e_masks = {
    "ub": np.uint64(0b1100000000000000000000000000000),
    "ur": np.uint64(0b0011000000000000000000000000000),
    "uf": np.uint64(0b0000110000000000000000000000000),
    "ul": np.uint64(0b0000001100000000000000000000000),
    "br": np.uint64(0b0000000011000000000000000000000),
    "fr": np.uint64(0b0000000000110000000000000000000),
    "fl": np.uint64(0b0000000000001100000000000000000),
    "bl": np.uint64(0b0000000000000011000000000000000),
    "db": np.uint64(0b0000000000000000110000000000000),
    "dl": np.uint64(0b0000000000000000001100000000000),
    "df": np.uint64(0b0000000000000000000011000000000),
    "dr": np.uint64(0b0000000000000000000000110000000)
}
fl_vals = {
    "ub": np.uint64(0b0100000000000000000000000000000),
    "ur": np.uint64(0b0001000000000000000000000000000),
    "uf": np.uint64(0b0000010000000000000000000000000),
    "ul": np.uint64(0b0000000100000000000000000000000),
    "br": np.uint64(0b0000000001000000000000000000000),
    "fr": np.uint64(0b0000000000010000000000000000000),
    "fl": np.uint64(0b0000000000000100000000000000000),
    "bl": np.uint64(0b0000000000000001000000000000000),
    "db": np.uint64(0b0000000000000000010000000000000),
    "dl": np.uint64(0b0000000000000000000100000000000),
    "df": np.uint64(0b0000000000000000000001000000000),
    "dr": np.uint64(0b0000000000000000000000010000000)
}
bl_vals = {
    "ub": np.uint64(0b1000000000000000000000000000000),
    "ur": np.uint64(0b0010000000000000000000000000000),
    "uf": np.uint64(0b0000100000000000000000000000000),
    "ul": np.uint64(0b0000001000000000000000000000000),
    "br": np.uint64(0b0000000010000000000000000000000),
    "fr": np.uint64(0b0000000000100000000000000000000),
    "fl": np.uint64(0b0000000000001000000000000000000),
    "bl": np.uint64(0b0000000000000010000000000000000),
    "db": np.uint64(0b0000000000000000100000000000000),
    "dl": np.uint64(0b0000000000000000001000000000000),
    "df": np.uint64(0b0000000000000000000010000000000),
    "dr": np.uint64(0b0000000000000000000000100000000)
}
dl_vals = {
    "ub": np.uint64(0b1100000000000000000000000000000),
    "ur": np.uint64(0b0011000000000000000000000000000),
    "uf": np.uint64(0b0000110000000000000000000000000),
    "ul": np.uint64(0b0000001100000000000000000000000),
    "br": np.uint64(0b0000000011000000000000000000000),
    "fr": np.uint64(0b0000000000110000000000000000000),
    "fl": np.uint64(0b0000000000001100000000000000000),
    "bl": np.uint64(0b0000000000000011000000000000000),
    "db": np.uint64(0b0000000000000000110000000000000),
    "dl": np.uint64(0b0000000000000000001100000000000),
    "df": np.uint64(0b0000000000000000000011000000000),
    "dr": np.uint64(0b0000000000000000000000110000000)
}
zero_corners = ["ubr","ufr","ufl","ubl","dfr","dbr"]
c_masks = {
    "ubr": np.uint64(0b0000001110000000000000000000000000000000000000000000000000000),
    "ufr": np.uint64(0b0000000001110000000000000000000000000000000000000000000000000),
    "ufl": np.uint64(0b0000000000001110000000000000000000000000000000000000000000000),
    "ubl": np.uint64(0b0000000000000001110000000000000000000000000000000000000000000),
    "dbl": np.uint64(0b0000000000000000001110000000000000000000000000000000000000000),
    "dfl": np.uint64(0b0000000000000000000001110000000000000000000000000000000000000),
    "dfr": np.uint64(0b0000000000000000000000001110000000000000000000000000000000000),
    "dbr": np.uint64(0b0000000000000000000000000001110000000000000000000000000000000)
}
dfl_vals = {
    "ubr": np.uint64(0b0000001010000000000000000000000000000000000000000000000000000),
    "ufr": np.uint64(0b0000000001010000000000000000000000000000000000000000000000000),
    "ufl": np.uint64(0b0000000000001010000000000000000000000000000000000000000000000),
    "ubl": np.uint64(0b0000000000000001010000000000000000000000000000000000000000000),
    "dbl": np.uint64(0b0000000000000000001010000000000000000000000000000000000000000),
    "dfl": np.uint64(0b0000000000000000000001010000000000000000000000000000000000000),
    "dfr": np.uint64(0b0000000000000000000000001010000000000000000000000000000000000),
    "dbr": np.uint64(0b0000000000000000000000000001010000000000000000000000000000000)
}
dbl_vals = {
    "ubr": np.uint64(0b0000001000000000000000000000000000000000000000000000000000000),
    "ufr": np.uint64(0b0000000001000000000000000000000000000000000000000000000000000),
    "ufl": np.uint64(0b0000000000001000000000000000000000000000000000000000000000000),
    "ubl": np.uint64(0b0000000000000001000000000000000000000000000000000000000000000),
    "dbl": np.uint64(0b0000000000000000001000000000000000000000000000000000000000000),
    "dfl": np.uint64(0b0000000000000000000001000000000000000000000000000000000000000),
    "dfr": np.uint64(0b0000000000000000000000001000000000000000000000000000000000000),
    "dbr": np.uint64(0b0000000000000000000000000001000000000000000000000000000000000)
}
def turn(move, start):
    result = np.bitwise_and(start, move["main"])
    for shift_info in move["p"]["left"]:
        result = np.bitwise_or(result, np.left_shift(np.bitwise_and(start, shift_info["mask"]), shift_info["shift"]))
    for shift_info in move["p"]["right"]:
        result = np.bitwise_or(result, np.right_shift(np.bitwise_and(start, shift_info["mask"]), shift_info["shift"]))
    if "o_e" in move:
        fl_unchanged = True
        bl_unchanged = True
        dl_unchanged = True
        for edge in move["o_e"]:
            if np.bitwise_and(start, e_masks[edge]) == fl_vals[edge]:
                result = np.bitwise_or(result, np.bitwise_xor(np.bitwise_and(start, fl_mask), fl_mask))
                fl_unchanged = False
            elif np.bitwise_and(start, e_masks[edge]) == bl_vals[edge]:
                result = np.bitwise_or(result, np.bitwise_xor(np.bitwise_and(start, bl_mask), bl_mask))
                bl_unchanged = False
            elif np.bitwise_and(start, e_masks[edge]) == dl_vals[edge]:
                result = np.bitwise_or(result, np.bitwise_xor(np.bitwise_and(start, dl_mask), dl_mask))
                dl_unchanged = False
        if fl_unchanged:
            result = np.bitwise_or(result, np.bitwise_and(start, fl_mask))
        if bl_unchanged:
            result = np.bitwise_or(result, np.bitwise_and(start, bl_mask))
        if dl_unchanged:
            result = np.bitwise_or(result, np.bitwise_and(start, dl_mask))
    if "o_c" in move:
        dfl_unchanged = True
        dbl_unchanged = True
        for o_info in move["o_c"]: 
            if np.bitwise_and(start, c_masks[o_info["c"]]) == dfl_vals[o_info["c"]]:
                result |= np.left_shift(((np.right_shift(np.bitwise_and(start, dfl_mask), 2) + o_info["o"]) % 3), 2)
                dfl_unchanged = False
            elif np.bitwise_and(start, c_masks[o_info["c"]]) == dbl_vals[o_info["c"]]:
                result |= (np.bitwise_and(start, dbl_mask) + o_info["o"]) % 3
                dbl_unchanged = False
        if dfl_unchanged:
            result = np.bitwise_or(result, np.bitwise_and(start, dfl_mask))
        if dbl_unchanged:
            result = np.bitwise_or(result, np.bitwise_and(start, dbl_mask))
    return result

def allowed_move(move1, move2):
    if move1[0] == move2[0] or move1[0].lower() == move2[0]:
        return False
    else:
        return True

def execute_alg(alg, base_state):
    temp_state = turn(move_def[alg[0]], base_state)
    for move in alg[1:]:
        temp_state = turn(move_def[move], temp_state)
    return temp_state

def generate_seq(seq_len, moves):
    if seq_len == 1:
        for move in moves:
            yield [move]
    else:
        for seq in generate_seq(seq_len-1, moves):
            for move in moves:
                if seq[-1][0] != move[0] and seq[-1][0] != move[0].lower():
                    yield seq+[move]

def find_sol_len(scram_state):
    if scram_state in prune_table:
        return prune_table[scram_state]
    else:
        max_search = 3
        curr_depth = 1
        while curr_depth <= max_search:
            for search_seq in generate_seq(curr_depth, zeroSimple):
                cand_state = execute_alg(search_seq, scram_state)
                if cand_state in prune_table:
                    return curr_depth+prune_table[cand_state]
            curr_depth += 1
    return -1

def generate_all_states():
    for ce_pos in ce_masks.values():
        for c_pos in itertools.permutations(zero_corners, 2):
            for e_pos in itertools.permutations(e_masks.keys(), 3):
                for dfl_o in [0b0000, 0b0100, 0b1000]:
                    for dbl_o in [0b00, 0b01, 0b10]:
                        for fl_o in [0b0, 0b1000000]:
                            for bl_o in [0b0, 0b100000]:
                                for dl_o in [0b0, 0b10000]:
                                    yield ce_pos | dfl_vals[c_pos[0]] | dbl_vals[c_pos[1]] | fl_vals[e_pos[0]] | bl_vals[e_pos[1]] | dl_vals[e_pos[2]] | dfl_o | dbl_o | fl_o | bl_o | dl_o

mainSolved = np.uint64(0b0100000000010100111001011101110000000000000110001100000000000)
zeroSolved = np.uint64(0b0100000000000000001001010000000000000000000110001100000000000)
DrSolved = np.uint64(0b0000010000000000000000001011000000000000000000100001111111001)
UlSolved = np.uint64(0b1000000000001011000000000000001000011100000000000000001111001)
rotation_moves = ["x", "x'", "x2", "y", "y'", "y2", "z", "z'", "z2"]
zeroMoves = ["U", "U'", "U2", "u", "u'", "u2", "R", "R'", "R2", "r", "r'", "r2", "M", "M'", "M2", "E", "E'", "E2", "S", "S'", "S2"]
zeroSimple = ["U", "U'", "U2", "u", "u'", "u2", "R", "R'", "R2", "r", "r'", "r2", "S", "S'", "S2"]
noRotations = ["U", "U'", "U2", "D", "D'", "D2", "R", "R'", "R2", "L", "L'", "L2", "F", "F'", "F2", "B", "B'", "B2", "u", "u'", "u2", "d", "d'", "d2", "r", "r'", "r2", "l", "l'", "l2", "f", "f'", "f2", "b", "b'", "b2", "M", "M'", "M2", "E", "E'", "E2", "S", "S'", "S2"]
x2y2_sides = [["i"], ["x2"], ["y2"], ["z2"]]
x2y2_front = [["y"], ["y'"], ["x2", "y"], ["x2", "y'"]]
x2y2_z_sides = [["z"], ["z", "y2"], ["z'"], ["z'", "y2"]]
x2y2_z_front = [["z", "y"], ["z", "y'"], ["z'", "y"], ["z'", "y'"]]
x2y2_x_sides = [["x"], ["x", "y2"], ["x'"], ["x'", "y2"]]
x2y2_x_front = [["x", "y"], ["x", "y'"], ["x'", "y"], ["x'", "y'"]]
prune_table = {int(DrSolved):1, int(UlSolved):1}
curr_layer_algs = [["z"], ["z'"]]
curr_layer = 2
max_layer = 5
while curr_layer <= max_layer:
    print(curr_layer)
    layer_count = 0
    next_layer_algs = []
    for existing_alg in curr_layer_algs:
        for cand_move in zeroSimple:
            if not (existing_alg[-1][0] == cand_move[0] or existing_alg[-1][0].lower() == cand_move[0]):
                cand_state = execute_alg(existing_alg+[cand_move], zeroSolved)
                if cand_state not in prune_table:
                    prune_table[cand_state] = curr_layer
                    layer_count+=1
                    next_layer_algs.append(existing_alg+[cand_move])
                    if layer_count % 1000 == 0:
                        print(layer_count, end="\r")
    print(layer_count)
    print(len(prune_table))
    curr_layer += 1
    curr_layer_algs = next_layer_algs.copy()