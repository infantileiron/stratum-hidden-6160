import math
from src.services.hmm_math import NEG_INF, logsumexp, safelog

VARIANCE_MIN = 1e-8
VARIANCE_MAX = 1e8
MEAN_MAX_ABS = 1e6
MIN_EFFECTIVE_COUNT = 1e-8
LIKELIHOOD_DECREASE_TOL = 1e-9


class FitStop(Exception):
    """A stopping condition raised while estimating or decoding a candidate."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason
        self.message = message


def _emissions(values, means, variances):
    return [[-.5 * (math.log(2 * math.pi) + math.log(var) + (value - mean) ** 2 / var)
             for mean, var in zip(means, variances)] for value in values]


def _config_arrays(config):
    states = config['states']
    means = [float(s['mean']) for s in states]
    variances = [float(s['variance']) for s in states]
    initial = [float(p) for p in config['initial']]
    transition = [[float(p) for p in row] for row in config['transition']]
    return initial, transition, means, variances


def zero_masks(config):
    initial_allowed = [p > 0 for p in config['initial']]
    transition_allowed = [[p > 0 for p in row] for row in config['transition']]
    return initial_allowed, transition_allowed


def _check_dimensions(initial, transition, means, variances):
    k = len(means)
    if not 2 <= k <= 4 or len(variances) != k or len(initial) != k or len(transition) != k:
        raise FitStop('invalid_config', '候选模型需2—4个状态及一致概率维度。')
    if any(len(row) != k for row in transition):
        raise FitStop('invalid_config', '候选模型转移矩阵维度不一致。')
    if any(row_sum <= 0 for row_sum in (math.fsum(row) for row in transition)):
        raise FitStop('degenerate_state', '候选模型存在全零转移行，无法估计。')


def _estep_sequence(values, initial, transition, means, variances):
    k = len(initial)
    emissions = _emissions(values, means, variances)
    offsets = [max(row) for row in emissions]
    relative = [[value - offset for value in row] for row, offset in zip(emissions, offsets)]
    log_transition = [[safelog(p) for p in row] for row in transition]

    first = [safelog(initial[j]) + relative[0][j] for j in range(k)]
    scale = logsumexp(first)
    if scale == NEG_INF:
        raise FitStop('no_feasible_path', '源参数下训练孔段无可行初始状态。')
    alpha = [[value - scale for value in first]]
    scales = [scale]
    for t in range(1, len(values)):
        raw = [relative[t][j] + logsumexp(alpha[-1][i] + log_transition[i][j] for i in range(k))
               for j in range(k)]
        scale = logsumexp(raw)
        if scale == NEG_INF:
            raise FitStop('no_feasible_path', '当前候选参数使训练孔段无可行路径（零转移或零初始）。')
        alpha.append([value - scale for value in raw])
        scales.append(scale)

    beta = [[0.0] * k for _ in values]
    for t in range(len(values) - 2, -1, -1):
        beta[t] = [logsumexp(log_transition[i][j] + relative[t + 1][j] + beta[t + 1][j]
                             for j in range(k)) - scales[t + 1] for i in range(k)]

    gammas = []
    for t in range(len(values)):
        raw = [alpha[t][j] + beta[t][j] for j in range(k)]
        normalizer = logsumexp(raw)
        if normalizer == NEG_INF or not math.isfinite(normalizer):
            raise FitStop('numerical_failure', '后验归一化出现非有限值，停止并保留最后有效参数。')
        row = [0.0 if p == NEG_INF else math.exp(p - normalizer) for p in raw]
        total = math.fsum(row)
        if total <= 0 or not math.isfinite(total):
            raise FitStop('numerical_failure', '后验质量非正，停止并保留最后有效参数。')
        gammas.append([p / total for p in row])

    transition_counts = [[0.0] * k for _ in range(k)]
    for t in range(len(values) - 1):
        raw = [[alpha[t][i] + log_transition[i][j] + relative[t + 1][j] + beta[t + 1][j]
                for j in range(k)] for i in range(k)]
        flat = [p for row in raw for p in row]
        normalizer = logsumexp(flat)
        if normalizer == NEG_INF or not math.isfinite(normalizer):
            raise FitStop('numerical_failure', '转移后验归一化出现非有限值，停止并保留最后有效参数。')
        for i in range(k):
            for j in range(k):
                p = raw[i][j]
                if p != NEG_INF:
                    transition_counts[i][j] += math.exp(p - normalizer)

    log_likelihood = math.fsum(offsets) + math.fsum(scales)
    if not math.isfinite(log_likelihood):
        raise FitStop('numerical_failure', '训练对数似然非有限，停止并保留最后有效参数。')
    return log_likelihood, gammas, transition_counts


def _estep(sequences, initial, transition, means, variances):
    """E-step for several complete boreholes, each starting from the same initial distribution."""
    k = len(initial)
    state_counts = [0.0] * k
    initial_counts = [0.0] * k
    transition_counts = [[0.0] * k for _ in range(k)]
    weighted_sum = [0.0] * k
    log_likelihood = 0.0
    stored = []
    for values in sequences:
        sequence_ll, gammas, sequence_transitions = _estep_sequence(
            values, initial, transition, means, variances)
        log_likelihood += sequence_ll
        stored.append((values, gammas))
        for j in range(k):
            initial_counts[j] += gammas[0][j]
            state_counts[j] += math.fsum(gamma[j] for gamma in gammas)
            weighted_sum[j] += math.fsum(gamma[j] * value for value, gamma in zip(values, gammas))
            for i in range(k):
                transition_counts[i][j] += sequence_transitions[i][j]
    stats = {'initial_counts': initial_counts, 'state_counts': state_counts,
             'transition_counts': transition_counts, 'weighted_sum': weighted_sum, 'stored': stored}
    return log_likelihood, stats


def _weighted_variance(stats, index, mean):
    count = stats['state_counts'][index]
    total = math.fsum(gamma[index] * (value - mean) ** 2
                      for values, gammas in stats['stored']
                      for value, gamma in zip(values, gammas))
    return total / count


def _mstep(config, stats, initial_allowed, transition_allowed):
    initial, transition, means, variances = _config_arrays(config)
    k = len(means)

    allowed_initial = [j for j in range(k) if initial_allowed[j]]
    initial_total = math.fsum(stats['initial_counts'][j] for j in allowed_initial)
    if initial_total <= MIN_EFFECTIVE_COUNT:
        raise FitStop('degenerate_state', '允许的初始状态有效样本量趋零，停止并保留最后有效参数。')
    new_initial = [0.0] * k
    for j in allowed_initial:
        new_initial[j] = stats['initial_counts'][j] / initial_total

    new_transition = [[0.0] * k for _ in range(k)]
    for i in range(k):
        allowed = [j for j in range(k) if transition_allowed[i][j]]
        row_total = math.fsum(stats['transition_counts'][i][j] for j in allowed)
        if row_total <= MIN_EFFECTIVE_COUNT:
            raise FitStop('degenerate_state', f'状态{config["states"][i]["id"]}的有效转移样本量趋零，停止并保留最后有效参数。')
        for j in allowed:
            new_transition[i][j] = stats['transition_counts'][i][j] / row_total
    for i in range(k):
        for j in range(k):
            if not transition_allowed[i][j]:
                new_transition[i][j] = 0.0

    new_means = []
    new_variances = []
    for j in range(k):
        count = stats['state_counts'][j]
        if count <= MIN_EFFECTIVE_COUNT:
            raise FitStop('degenerate_state', f'状态{config["states"][j]["id"]}的有效样本量趋零，停止并保留最后有效参数。')
        mean = stats['weighted_sum'][j] / count
        if not math.isfinite(mean) or abs(mean) > MEAN_MAX_ABS:
            raise FitStop('mean_out_of_range', f'状态{config["states"][j]["id"]}的均值超出允许范围，停止并保留最后有效参数。')
        variance = _weighted_variance(stats, j, mean)
        if not math.isfinite(variance):
            raise FitStop('numerical_failure', f'状态{config["states"][j]["id"]}的方差非有限，停止并保留最后有效参数。')
        if variance < VARIANCE_MIN:
            raise FitStop('variance_collapse', f'状态{config["states"][j]["id"]}的方差低于下限，停止并保留最后有效参数。')
        if variance > VARIANCE_MAX:
            raise FitStop('variance_unbounded', f'状态{config["states"][j]["id"]}的方差超出上限，停止并保留最后有效参数。')
        new_means.append(mean)
        new_variances.append(variance)

    candidate = {
        'states': [{'id': state['id'], 'name': state['name'], 'mean': new_means[j],
                    'variance': new_variances[j]} for j, state in enumerate(config['states'])],
        'initial': new_initial,
        'transition': new_transition,
    }
    _validate_candidate(candidate)
    return candidate


def _validate_candidate(config):
    initial, transition, means, variances = _config_arrays(config)
    _check_dimensions(initial, transition, means, variances)
    for row in [initial] + transition:
        if any(not math.isfinite(p) or p < 0 or p > 1 for p in row):
            raise FitStop('invalid_config', '候选概率需有限且位于[0,1]。')
        if abs(math.fsum(row) - 1) > 1e-10:
            raise FitStop('invalid_config', '候选概率行和须为1。')
    for state in config['states']:
        if not isinstance(state['name'], str) or not state['name'].strip():
            raise FitStop('invalid_config', '候选状态名称不能为空。')


def _summary(config, stats, log_likelihood):
    initial, transition, means, variances = _config_arrays(config)
    initial_allowed, transition_allowed = zero_masks(config)
    return {
        'training_log_likelihood': log_likelihood,
        'state_effective_counts': stats['state_counts'],
        'min_variance': min(variances),
        'max_variance': max(variances),
        'preserved_zero_initial': [j for j in range(len(initial)) if not initial_allowed[j]],
        'preserved_zero_transitions': [[j for j in range(len(initial)) if not transition_allowed[i][j]]
                                       for i in range(len(initial))],
    }


def fit_candidate(source_config, sequences, max_iterations=50, tolerance=1e-6):
    """Jointly fit a Gaussian HMM over several complete boreholes.

    ``sequences`` is a list of value lists; each sequence begins independently from the
    shared initial distribution.  Original exact zeros are structural and never smoothed.
    """
    initial, transition, means, variances = _config_arrays(source_config)
    _check_dimensions(initial, transition, means, variances)
    initial_allowed, transition_allowed = zero_masks(source_config)
    iterations = []
    try:
        baseline_ll, stats = _estep(sequences, initial, transition, means, variances)
    except FitStop as exc:
        return {'status': 'failed', 'stop_reason': exc.reason, 'stop_message': exc.message,
                'iterations': [], 'accepted_iterations': 0, 'attempted_iterations': 0,
                'candidate_config': None, 'candidate_summary': None}

    iterations.append({'iteration': 0, 'stage': 'baseline', 'training_log_likelihood': baseline_ll,
                       'improvement': None, 'state_effective_counts': stats['state_counts'],
                       'min_variance': min(variances), 'max_variance': max(variances), 'note': '源参数基线'})

    current_config = source_config
    current_ll = baseline_ll
    current_stats = stats
    accepted = 0
    attempted = 0
    stop_reason = 'max_iterations'
    stop_message = '达到最大迭代次数，保留最后一次有效更新。'

    for iteration in range(1, max_iterations + 1):
        attempted = iteration
        try:
            proposed = _mstep(current_config, current_stats, initial_allowed, transition_allowed)
        except FitStop as exc:
            iterations.append({'iteration': iteration, 'stage': 'invalid', 'training_log_likelihood': None,
                               'improvement': None, 'state_effective_counts': None,
                               'min_variance': None, 'max_variance': None, 'note': exc.message})
            stop_reason, stop_message = exc.reason, exc.message
            break
        proposed_initial, proposed_transition, proposed_means, proposed_variances = _config_arrays(proposed)
        try:
            proposed_ll, proposed_stats = _estep(
                sequences, proposed_initial, proposed_transition, proposed_means, proposed_variances)
        except FitStop as exc:
            iterations.append({'iteration': iteration, 'stage': 'invalid', 'training_log_likelihood': None,
                               'improvement': None, 'state_effective_counts': None,
                               'min_variance': None, 'max_variance': None, 'note': exc.message})
            stop_reason, stop_message = exc.reason, exc.message
            break

        improvement = proposed_ll - current_ll
        scale = max(1.0, abs(current_ll))
        if improvement < -LIKELIHOOD_DECREASE_TOL * scale:
            iterations.append({'iteration': iteration, 'stage': 'rejected',
                               'training_log_likelihood': proposed_ll, 'improvement': improvement,
                               'state_effective_counts': proposed_stats['state_counts'],
                               'min_variance': min(proposed_variances), 'max_variance': max(proposed_variances),
                               'note': '训练似然较上一步下降，丢弃该步并保留最后有效参数。'})
            stop_reason = 'likelihood_decrease'
            stop_message = '训练似然较上一步下降，丢弃该步并保留最后有效参数。'
            break

        current_config, current_ll, current_stats = proposed, proposed_ll, proposed_stats
        accepted = iteration
        iterations.append({'iteration': iteration, 'stage': 'accepted', 'training_log_likelihood': proposed_ll,
                           'improvement': improvement,
                           'state_effective_counts': proposed_stats['state_counts'],
                           'min_variance': min(proposed_variances), 'max_variance': max(proposed_variances),
                           'note': '保留该步参数'})
        if improvement <= tolerance * scale:
            stop_reason = 'converged'
            stop_message = '训练似然改进已低于收敛阈值，候选参数保留。'
            break

    if accepted == 0:
        return {'status': 'failed', 'stop_reason': stop_reason, 'stop_message': stop_message,
                'iterations': iterations, 'accepted_iterations': 0, 'attempted_iterations': attempted,
                'candidate_config': None, 'candidate_summary': None}

    return {'status': 'completed', 'stop_reason': stop_reason, 'stop_message': stop_message,
            'iterations': iterations, 'accepted_iterations': accepted, 'attempted_iterations': attempted,
            'candidate_config': current_config,
            'candidate_summary': _summary(current_config, current_stats, current_ll)}


def evaluate_config(config, hole_values):
    """Evaluate one configured model on complete holes; simulation truth is never read."""
    from src.services.forward_backward import forward_backward
    from src.services.viterbi import viterbi

    states = config['states']
    means = [s['mean'] for s in states]
    variances = [s['variance'] for s in states]
    output = []
    for hole, values in hole_values:
        try:
            forward = forward_backward(values, config['initial'], config['transition'], means, variances)
            path = viterbi(values, config['initial'], config['transition'], means, variances)
            count = len(values)
            path_changes = sum(a != b for a, b in zip(path['path'], path['path'][1:]))
            output.append({
                'borehole': hole,
                'point_count': count,
                'log_likelihood': forward['log_likelihood'],
                'per_point_log_likelihood': forward['log_likelihood'] / count,
                'log_joint_density': path['log_joint_density'],
                'path_changes': path_changes,
                'mean_max_posterior': math.fsum(max(row) for row in forward['posterior']) / count,
                'path_counts': {state['id']: sum(index == j for index in path['path'])
                                for j, state in enumerate(states)},
            })
        except ValueError as exc:
            output.append({'borehole': hole, 'point_count': len(values), 'error': str(exc)})
    return output
