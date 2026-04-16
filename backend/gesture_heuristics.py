import math

def distance(x, y):
    """Calculate Euclidean distance between two points"""
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


def hand_scale(pts):
    """Estimate hand size so thresholds can adapt to camera distance."""
    palm_width = distance(pts[5], pts[17])
    palm_height = distance(pts[0], pts[9])
    return max(palm_width, palm_height, 1.0)


def classify_c_or_o(pts):
    """
    Separate C from O using normalized thumb-to-finger gaps.
    O should only win when the hand is tightly closed.
    """
    scale = hand_scale(pts)
    thumb_index = distance(pts[4], pts[8]) / scale
    thumb_middle = distance(pts[4], pts[12]) / scale
    thumb_ring = distance(pts[4], pts[16]) / scale
    avg_gap = (thumb_index + thumb_middle + thumb_ring) / 3.0

    tight_o = thumb_index < 0.34 and thumb_middle < 0.48 and avg_gap < 0.47
    open_c = thumb_index > 0.43 or thumb_middle > 0.62 or avg_gap > 0.56

    if tight_o:
        return "O"
    if open_c:
        return "C"

    return "C" if thumb_middle >= 0.54 else "O"


def looks_like_a(pts):
    """
    Catch compact fist-like shapes that can be mistaken for C after later tweaks.
    """
    scale = hand_scale(pts)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    folded_count = sum(
        1 for tip, pip in zip(finger_tips, finger_pips)
        if pts[tip][1] > pts[pip][1] - 4
    )
    compact_count = sum(
        1 for tip in finger_tips
        if distance(pts[tip], pts[0]) / scale < 1.18
    )
    thumb_outside = (
        pts[4][0] < min(pts[6][0], pts[10][0], pts[14][0], pts[18][0]) or
        pts[4][0] > max(pts[6][0], pts[10][0], pts[14][0], pts[18][0])
    )

    return folded_count >= 3 and compact_count >= 3 and thumb_outside

def evaluate_gesture(ch1, ch2, pts):
    """
    Evaluates the geometric constraints of a hand skeleton.
    
    Args:
        ch1 (int): The primary class predicted by the CNN (0-7)
        ch2 (int): The secondary class predicted by the CNN (0-7)
        pts (list): 21 strict XYZ tuples representing MediaPipe hand landmarks.
        
    Returns:
        str: The final calibrated letter or symbol, else returns the unmodified int.
    """
    pl = [ch1, ch2]

    # --- Phase 1: Group Re-Calibration ---
    # In this phase, we look at the top two group shapes (pl) predicted by the Neural Network.
    # We then apply strict geometric formulas based on the hand landmarks to lock in the true subgroup.

    group_0_rules = [
        [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
         [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
         [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]],
        [[2, 2], [2, 1]]
    ]
    if pl in group_0_rules[0] and (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
        ch1 = 0
    if pl in group_0_rules[1] and (pts[5][0] < pts[4][0]):
        ch1 = 0

    group_2_rules = [
        [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]],
        [[6, 0], [6, 6], [6, 2]]
    ]
    if pl in group_2_rules[0] and (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
        ch1 = 2
    if pl in group_2_rules[1] and distance(pts[8], pts[16]) < 52:
        ch1 = 2

    group_3_rules = [
        [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]],
        [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]],
        [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
    ]
    if pl in group_3_rules[0] and pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
        ch1 = 3
    if pl in group_3_rules[1] and pts[4][0] > pts[0][0]:
        ch1 = 3
    if pl in group_3_rules[2] and pts[2][1] + 15 < pts[16][1]:
        ch1 = 3

    group_4_rules = [
        [[6, 4], [6, 1], [6, 2]],
        [[1, 4], [1, 6], [1, 1]],
        [[3, 6], [3, 4]],
        [[2, 2], [2, 5], [2, 4]]
    ]
    if pl in group_4_rules[0] and distance(pts[4], pts[11]) > 55:
        ch1 = 4
    if pl in group_4_rules[1] and (distance(pts[4], pts[11]) > 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
        ch1 = 4
    if pl in group_4_rules[2] and (pts[4][0] < pts[0][0]):
        ch1 = 4
    if pl in group_4_rules[3] and (pts[1][0] < pts[12][0]):
        ch1 = 4

    group_5_rules = [
        [[3, 6], [3, 5], [3, 4]],
        [[3, 2], [3, 1], [3, 6]],
        [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]],
        [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
    ]
    if pl in group_5_rules[0] and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
        ch1 = 5
    if pl in group_5_rules[1] and pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][1] + 17 > pts[20][1]:
        ch1 = 5
    if pl in group_5_rules[2] and pts[4][0] > pts[0][0]:
        ch1 = 5
    if pl in group_5_rules[3] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
        ch1 = 5

    group_6_rules = [
        [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]],
        [[7, 2]],
        [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]],
        [[4, 6], [4, 2], [4, 1], [4, 4]],
        [[1, 4], [1, 6], [1, 0], [1, 2]]
    ]
    if pl in group_6_rules[0] and pts[5][0] > pts[16][0]:
        ch1 = 6
    if pl in group_6_rules[1] and pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
        ch1 = 6
    if pl in group_6_rules[2] and distance(pts[8], pts[16]) > 50:
        ch1 = 6
    if pl in group_6_rules[3] and distance(pts[4], pts[11]) < 60:
        ch1 = 6
    if pl in group_6_rules[4] and pts[5][0] - pts[4][0] - 15 > 0:
        ch1 = 6

    group_7_rules = [
        [[5, 7], [5, 2], [5, 6]],
        [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]],
        [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
    ]
    if pl in group_7_rules[0] and pts[3][0] < pts[0][0]:
        ch1 = 7
    if pl in group_7_rules[1] and pts[6][1] < pts[8][1]:
        ch1 = 7
    if pl in group_7_rules[2] and pts[18][1] > pts[20][1]:
        ch1 = 7

    group_1_rules = [
        [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0], [6, 3], [6, 4], [7, 5], [7, 2]],
        [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]],
        [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]],
        [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]],
        [[4, 1], [4, 2], [4, 4]],
        [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]],
        [[6, 6], [6, 4], [6, 1], [6, 2]],
        [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]],
        [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]],
        [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]],
        [[5, 0], [5, 5], [0, 1]]
    ]
    
    if pl in group_1_rules[0] and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
        ch1 = 1
    if pl in group_1_rules[1] and (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
        ch1 = 1
    if pl in group_1_rules[2] and (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
        ch1 = 1
    if pl in group_1_rules[3] and ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[4][1] > pts[14][1]):
        ch1 = 1
    if pl in group_1_rules[4] and (distance(pts[4], pts[11]) < 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
        ch1 = 1
    if pl in group_1_rules[5] and ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1] < pts[4][1]):
        ch1 = 1
    if pl in group_1_rules[6] and pts[5][0] - pts[4][0] - 15 < 0:
        ch1 = 1
    if pl in group_1_rules[7] and ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
        ch1 = 1
    if pl in group_1_rules[8] and ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])) and pts[4][1] > pts[14][1]:
        ch1 = 1
    if pl in group_1_rules[9] and not (pts[0][0] + 19 < pts[8][0] and pts[0][0] + 19 < pts[12][0] and pts[0][0] + 19 < pts[16][0] and pts[0][0] + 19 < pts[20][0]) and not (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and distance(pts[4], pts[11]) < 50:
        ch1 = 1
    if pl in group_1_rules[10] and pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
        ch1 = 1

    l_mixed = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
    if pl in l_mixed and (pts[4][0] < pts[5][0] + 15) and ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
        ch1 = 7

    # --- Phase 2: Letter Mapping ---
    # Now that the subgroup is accurate, we differentiate between letters in that subgroup!

    if ch1 == 0:
        ch1 = 'S'
        if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]: ch1 = 'A'
        elif pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]: ch1 = 'T'
        elif pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]: ch1 = 'E'
        elif pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]: ch1 = 'M'
        elif pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]: ch1 = 'N'

    elif ch1 == 2:
        ch1 = classify_c_or_o(pts)
        if ch1 == 'C' and looks_like_a(pts):
            ch1 = 'A'

    elif ch1 == 3:
        ch1 = 'G' if distance(pts[8], pts[12]) > 72 else 'H'

    elif ch1 == 7:
        ch1 = 'Y' if distance(pts[8], pts[4]) > 42 else 'J'

    elif ch1 == 4:
        ch1 = 'L'

    elif ch1 == 6:
        ch1 = 'X'

    elif ch1 == 5:
        if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
            ch1 = 'Z' if pts[8][1] < pts[5][1] else 'Q'
        else:
            ch1 = 'P'

    elif ch1 == 1:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]): ch1 = 'B'
        elif (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]): ch1 = 'D'
        elif (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]): ch1 = 'F'
        elif (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]): ch1 = 'I'
        elif (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]): ch1 = 'W'
        elif (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] < pts[9][1]: ch1 = 'K'
        elif ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]): ch1 = 'U'
        elif ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] > pts[9][1]): ch1 = 'V'
        elif (pts[8][0] > pts[12][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]): ch1 = 'R'

    # --- Phase 3: Punctuation ---
    if isinstance(ch1, str):
        if ch1 in ['1', 'E', 'S', 'X', 'Y', 'B']:
            if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
                ch1 = " "

        if ch1 in ['E', 'Y', 'B']:
            if (pts[4][0] < pts[5][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]): 
                ch1 = "next"

        if ch1 in ['Next', 'B', 'C', 'H', 'F', 'X']:
            if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and (pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and pts[4][1] < pts[16][1] and pts[4][1] < pts[20][1]) and (pts[4][1] < pts[6][1] and pts[4][1] < pts[10][1] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]): 
                ch1 = 'Backspace'

    return ch1
