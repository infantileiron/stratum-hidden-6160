import copy
import hashlib
import json
import math
import threading

from src.models.entities import StateModel
from src.services.em_fit import evaluate_config, fit_candidate
from src.utils import repository as repo
from src.utils.identity import new_id, now

MAX_FIT_ITERATIONS = 50
MIN_TRAIN_BOREHOLES = 2
_REGISTER_LOCK = threading.Lock()


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def dataset_fingerprint(dataset):
    rows = [{'borehole': r['borehole'], 'depth_m': r['depth_m'], 'value': r['value']}
            for r in dataset['rows']]
    return hashlib.sha256(_canonical(rows).encode()).hexdigest()


def _group_holes(dataset):
    groups = {}
    order = []
    for row in dataset['rows']:
        hole = row['borehole']
        if hole not in groups:
            groups[hole] = []
            order.append(hole)
        groups[hole].append(row)
    for hole in order:
        rows = groups[hole]
        for previous, current in zip(rows, rows[1:]):
            if not math.isclose(current['depth_m'] - previous['depth_m'], 1.0, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f'{hole}孔不具有完整1米等距孔段，拒绝联合拟合')
    return groups, order


def _hole_values(rows):
    return [row['value'] for row in rows]


def _stale_reason(fit):
    refs = fit['references']
    try:
        dataset = repo.get('datasets', refs['dataset_id'])
        model = repo.get('state_models', refs['source_model_id'])
    except Exception:
        return '上游数据集或源状态模型已不可用，旧候选失效。'
    if dataset_fingerprint(dataset) != refs['dataset_fingerprint']:
        return '上游测井数据已更新，旧候选失效。'
    if model.get('revision') != refs['source_revision']:
        return '源状态模型已修订，旧候选失效。'
    return ''


def _public(fit):
    reason = _stale_reason(fit)
    return {**fit, 'stale': bool(reason), 'stale_reason': reason,
            'registration_allowed': fit['status'] == 'completed' and not reason}


def create_fit(data):
    dataset = repo.get('datasets', data.dataset_id)
    source = repo.get('state_models', data.source_model_id)
    if source.get('dataset_id') and source['dataset_id'] != dataset['id']:
        raise ValueError('源状态模型不属于所选测井数据集')
    groups, order = _group_holes(dataset)
    holdout = list(dict.fromkeys(data.holdout_boreholes))
    unknown = [hole for hole in holdout if hole not in groups]
    if unknown:
        raise ValueError('留出孔必须存在于所选数据集中：' + '、'.join(unknown))
    train = [hole for hole in order if hole not in holdout]
    if len(holdout) < 1:
        raise ValueError('至少留出一个整孔用于评估')
    if len(train) < MIN_TRAIN_BOREHOLES:
        raise ValueError(f'联合拟合至少需要{MIN_TRAIN_BOREHOLES}个完整训练孔；请减少留出孔')
    if any(len(groups[hole]) < 3 for hole in train):
        raise ValueError('每个训练孔至少需要3个完整1米点')

    train_sequences = [_hole_values(groups[hole]) for hole in train]
    holdout_pairs = [(hole, _hole_values(groups[hole])) for hole in holdout]

    config = copy.deepcopy(source['config'])
    result = fit_candidate(config, train_sequences, max_iterations=MAX_FIT_ITERATIONS)
    fit = {
        'id': new_id('fit'),
        'dataset_id': data.dataset_id,
        'source_model_id': data.source_model_id,
        'source_model_name': source['name'],
        'references': {
            'dataset_id': dataset['id'],
            'dataset_fingerprint': dataset_fingerprint(dataset),
            'source_model_id': source['id'],
            'source_revision': source['revision'],
        },
        'method': 'whole_borehole_holdout',
        'train_boreholes': train,
        'holdout_boreholes': holdout,
        'status': result['status'],
        'stop_reason': result['stop_reason'],
        'stop_message': result['stop_message'],
        'iterations': result['iterations'],
        'accepted_iterations': result['accepted_iterations'],
        'attempted_iterations': result['attempted_iterations'],
        'max_iterations': MAX_FIT_ITERATIONS,
        'source_config': config,
        'candidate_config': result['candidate_config'],
        'candidate_summary': result['candidate_summary'],
        'evaluations': None,
        'registration': None,
        'created_at': now(),
        'limitations': [
            '只使用训练孔的测井读数联合拟合；留出整孔与模拟参考truth不参与参数估计',
            '每孔独立起算，共享初始分布与转移矩阵，不在孔间串链',
            '原配置中精确为0的初始/转移概率保持为0，不作平滑',
            '留出孔评估只比较候选与源参数，不据留出结果回调参数或挑选迭代',
            '拟合出的标签仍是给定高斯HMM的隐状态，不是真实地层',
        ],
    }
    fit['evaluations'] = {
        'source_holdout': evaluate_config(config, holdout_pairs),
        'candidate_holdout': (evaluate_config(result['candidate_config'], holdout_pairs)
                              if result['candidate_config'] else None),
        'source_train': evaluate_config(config, [(hole, _hole_values(groups[hole])) for hole in train]),
        'candidate_train': (evaluate_config(result['candidate_config'],
                                            [(hole, _hole_values(groups[hole])) for hole in train])
                            if result['candidate_config'] else None),
    }
    return _public(repo.put('model_fits', fit))


def list_fits():
    return [_public(fit) for fit in repo.all_rows('model_fits')]


def get_fit(id):
    return _public(repo.get('model_fits', id))


def register_candidate(id, data):
    with _REGISTER_LOCK:
        return _register_candidate(id, data)

def _register_candidate(id, data):
    fit = repo.get('model_fits', id)
    if fit['status'] != 'completed' or not fit['candidate_config']:
        raise ValueError('该拟合没有可登记的候选参数；请检查停止原因后重新拟合')
    reason = _stale_reason(fit)
    if reason:
        raise ValueError(reason + ' 拒绝登记；已有研究继续使用其锁定快照。')
    if fit['registration']:
        raise ValueError('该候选已登记为新状态模型，请勿重复登记')

    from src.services.model_config import parse_model
    candidate_text = json.dumps(fit['candidate_config'], ensure_ascii=False)
    config = parse_model(candidate_text)
    model = StateModel(
        id=new_id('model'),
        name=data.name,
        config=config,
        revision=1,
        created_at=now(),
    )
    payload = {
        'id': model.id,
        'name': model.name,
        'config': model.config,
        'revision': model.revision,
        'dataset_id': fit['dataset_id'],
        'created_at': model.created_at,
        'provenance': {
            'kind': 'multi_borehole_joint_fit',
            'fit_id': fit['id'],
            'dataset_id': fit['dataset_id'],
            'dataset_fingerprint': fit['references']['dataset_fingerprint'],
            'train_boreholes': fit['train_boreholes'],
            'holdout_boreholes': fit['holdout_boreholes'],
            'source_model_id': fit['source_model_id'],
            'source_revision': fit['references']['source_revision'],
            'accepted_iterations': fit['accepted_iterations'],
            'stop_reason': fit['stop_reason'],
            'candidate_config_sha256': hashlib.sha256(
                _canonical(config).encode()).hexdigest(),
        },
        'note': '由完整训练孔联合拟合得到；留出整孔只作评估；truth不参与拟合。',
    }
    repo.put('state_models', payload)
    fit['registration'] = {
        'model_id': model.id,
        'model_name': model.name,
        'registered_at': now(),
        'candidate_config_sha256': payload['provenance']['candidate_config_sha256'],
    }
    fit['updated_at'] = now()
    repo.put('model_fits', fit)
    return payload
