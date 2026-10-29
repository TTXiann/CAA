import glob
import os
import re


DEFAULT_DATA_FOLDER = os.environ.get('CD_DATA_ROOT', '/project/datasets')
DEFAULT_SAVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'exps')

# Canonical run name shared by CD and CC: bs{batch_size}_{version}_r{rank}.
_EXP_NAME = re.compile(r'^bs(\d+)_(.+)_r(\d+)$')


def _cd_tag(dataset):
    """CD split tag: levir -> levir-cd, levir+ -> levir-cdplus."""
    name = str(dataset)
    base = name.replace('+', '')
    return f'{base}-cd' + ('plus' if '+' in name else '')


def _cd_tag_legacy(dataset):
    """Pre-suffix tag kept for experiments created before the rename."""
    return str(dataset).replace('+', 'plus')


def model_version(name):
    """Return the version suffix of a backbone/model id: dinov3_small -> small."""
    return str(name).rsplit('_', 1)[-1]


def format_exp_name(batch_size, version, rank):
    """Build the canonical run name bs{batch_size}_{version}_r{rank}."""
    return f'bs{int(batch_size)}_{version}_r{int(rank)}'


def parse_exp_name(exp_name):
    """Return (batch_size, version, rank) for a canonical name, else None."""
    match = _EXP_NAME.match(str(exp_name))
    if match is None:
        return None
    return int(match.group(1)), match.group(2), int(match.group(3))


def experiment_name(exp_name, rank):
    """Normalise a run name; canonical names are kept verbatim."""
    if parse_exp_name(exp_name):
        return exp_name
    return f'{exp_name}_r{int(rank)}'


def _exp_groups(exp_dir):
    """Experiment-group folder aliases, newest naming first."""
    return [exp_dir, f'exps-{exp_dir}']


def cd_exp_dir(save_dir, dataset, exp_dir, rank, exp_name):
    return os.path.join(save_dir, 'cd', _cd_tag(dataset), exp_dir,
                        experiment_name(exp_name, rank))


def resolve_cd_exp_dir(save_dir, dataset, dataset_folder, exp_dir, rank, exp_name):
    name = experiment_name(exp_name, rank)
    candidates = [cd_exp_dir(save_dir, dataset, exp_dir, rank, exp_name)]

    tags = [_cd_tag(dataset)]
    legacy_tag = _cd_tag_legacy(dataset)
    if legacy_tag != tags[0]:
        tags.append(legacy_tag)

    for tag in tags:
        # layout without the rank suffix (used before rank became part of the name)
        candidates.append(os.path.join(save_dir, 'cd', tag, exp_dir, exp_name))

    for group in _exp_groups(exp_dir):
        legacy_root = os.path.join(save_dir, group, f'exps_{dataset_folder.lower()}')
        candidates.append(os.path.join(legacy_root, name))
        candidates.append(os.path.join(legacy_root, exp_name))
        # older runs only encoded the version, e.g. small_r24 inside bs16_small_r24
        candidates.extend(sorted(glob.glob(os.path.join(legacy_root, f'*_{name}'))))

    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        'CD experiment dir not found, tried:\n  ' + '\n  '.join(candidates))


def cc_dataset_tag(data_name):
    name = str(data_name).lower()
    if 'levir' in name:
        return 'levir-cc'
    if 'dubai' in name:
        return 'dubai-cc'
    return name


def cc_exp_dir(save_dir, data_name, exp_dir, rank, exp_name):
    return os.path.join(save_dir, 'cc', cc_dataset_tag(data_name), exp_dir,
                        experiment_name(exp_name, rank))


def resolve_cc_exp_dir(save_dir, data_name, exp_dir, rank, exp_name,
                       n_enc, n_dec, dim, max_len):
    name = experiment_name(exp_name, rank)
    tag = cc_dataset_tag(data_name)

    candidates = [cc_exp_dir(save_dir, data_name, exp_dir, rank, exp_name)]
    # layout without the rank suffix (used before rank became part of the name)
    candidates.append(os.path.join(save_dir, 'cc', tag, exp_dir, exp_name))

    versions = [name]
    parsed = parse_exp_name(exp_name)
    if parsed and parsed[1] != name:
        # released runs are named enc.._dec.._len.._dim.._{version}
        versions.append(parsed[1])

    for group in _exp_groups(exp_dir):
        for version in versions:
            candidates.append(os.path.join(
                save_dir, group, f'enc{n_enc}_dec{n_dec}_len{max_len}_dim{dim}_{version}'))

    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        'CC experiment dir not found, tried:\n  ' + '\n  '.join(candidates))
