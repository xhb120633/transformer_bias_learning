"""Observable-only spatial transcripts. Each map is one causal training sequence."""
import re
import string

LETTERS = string.ascii_uppercase[:25]
VOCAB = ['<pad>', '<bos>'] + list(LETTERS) + list('0123456789.-+e') + ['grid', '/', 'cue', 'choice', 'reward', ';']
TOKEN_ID = {s: i for i, s in enumerate(VOCAB)}
CHOICE_IDS = [TOKEN_ID[s] for s in LETTERS]


def label(position):
    row, col = position
    assert 0 <= row < 5 and 0 <= col < 5
    return LETTERS[row * 5 + col]


def number(value):
    # Round-trip precision: do not change the generator's observed rewards.
    return repr(float(value))


def encode_map(episode, mode):
    assert mode in ('full', 'choice_only')
    tokens, mask = ['<bos>', 'grid'], [False, False]
    for row in range(5):
        tokens.extend(LETTERS[row * 5:row * 5 + 5]); tokens.append('/')
    tokens += ['cue', label(episode['initial_position'])]
    if mode == 'full':
        tokens += ['reward', *number(episode['initial_reward'])]
    tokens.append(';')
    mask = [False] * len(tokens)
    assert len(episode['choices']) == len(episode['rewards']) == 20
    for choice, reward in zip(episode['choices'], episode['rewards']):
        tokens.append('choice'); mask.append(False)
        tokens.append(label(choice)); mask.append(True)
        suffix = ['reward', *number(reward)] if mode == 'full' else []
        suffix.append(';')
        tokens.extend(suffix); mask.extend([False] * len(suffix))
    return [TOKEN_ID[t] for t in tokens], mask


def render_map(episode, mode):
    text = ('You are playing a spatial bandit game on a 5 by 5 grid. '
            'Choose a location on each turn to earn points. Locations may be chosen repeatedly. '
            'Nearby locations tend to have similar rewards. This is a new independent map. '
            'The grid, from top to bottom and left to right, is:\n'
            'A B C D E\nF G H I J\nK L M N O\nP Q R S T\nU V W X Y\n')
    if mode == 'choice_only':
        text += 'Reward observations are omitted from this record.\n'
    text += f"Initial revealed location: {label(episode['initial_position'])}."
    if mode == 'full':
        text += f" Reward: {number(episode['initial_reward'])}."
    text += '\n'
    for choice, reward in zip(episode['choices'], episode['rewards']):
        # Space before period prevents BPE merging ">>" with the line break
        # in choice-only records; original Centaur delimiter remains unchanged.
        text += f'Choice: <<{label(choice)}>> .'
        if mode == 'full':
            text += f' Reward: {number(reward)}.'
        text += '\n'
    assert len(re.findall(r'<<([A-Y])>>', text)) == 20
    return text
