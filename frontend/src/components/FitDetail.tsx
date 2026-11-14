import type {FitEvaluation, FitIteration, ModelFit} from '../types/entities';

const num=(v:number|null|undefined,digits=3)=>v===null||v===undefined?'—':v.toFixed(digits);
const delta=(a:number|null|undefined,b:number|null|undefined)=>(a===null||a===undefined||b===null||b===undefined)?null:b-a;

function EvaluationTable({title,note,source,candidate}:{title:string;note:string;source:FitEvaluation[];candidate:FitEvaluation[]|null}){
  return <section className="paper fit-evaluation">
    <h2>{title}</h2>
    <p className="method-note">{note}</p>
    <div className="table-scroll"><table>
      <thead><tr><th>整孔</th><th>点数</th><th>源参数 每点对数似然</th><th>候选 每点对数似然</th><th>差值（候选−源）</th><th>源路径改态</th><th>候选路径改态</th><th>候选路径状态计数</th></tr></thead>
      <tbody>{source.map((s,i)=>{
        const c=candidate?.[i];
        const improvement=delta(s.per_point_log_likelihood,c?.per_point_log_likelihood);
        return <tr key={s.borehole}>
          <td><b>{s.borehole}</b></td>
          <td>{s.point_count}</td>
          <td>{num(s.per_point_log_likelihood)}</td>
          <td>{num(c?.per_point_log_likelihood)}</td>
          <td className={improvement!==null&&improvement>0?'fit-better':improvement!==null&&improvement<0?'fit-worse':''}>{improvement===null?'—':improvement.toFixed(3)}</td>
          <td>{s.path_changes??'—'}</td>
          <td>{c?.path_changes??'—'}</td>
          <td>{c?.path_counts ? Object.entries(c.path_counts).map(([id,n])=>`${id} ${n}`).join(' / ') : '—'}</td>
        </tr>;
      })}</tbody>
    </table></div>
  </section>;
}

function ParameterComparison({fit}:{fit:ModelFit}){
  const candidate=fit.candidate_config;
  if(!candidate)return null;
  const source=fit.source_config;
  return <section className="paper fit-parameters">
    <h2>候选参数与源参数对照</h2>
    <p className="method-note">状态ID与人工名称沿用源模型，均值/方差按训练孔重估；原始精确为0的初始/转移概率保持为0，不加平滑伪计数。留出结果未参与任何参数更新。</p>
    <div className="fit-parameter-grid">
      <div className="table-scroll"><table>
        <thead><tr><th>状态</th><th>源均值</th><th>候选均值</th><th>源方差</th><th>候选方差</th><th>源初始</th><th>候选初始</th></tr></thead>
        <tbody>{candidate.states.map((s,i)=><tr key={s.id}>
          <td><b>{s.id}</b><small>{s.name}</small></td>
          <td>{source.states[i].mean.toFixed(3)}</td>
          <td>{s.mean.toFixed(3)}</td>
          <td>{source.states[i].variance.toFixed(3)}</td>
          <td>{s.variance.toFixed(3)}</td>
          <td>{source.initial[i].toFixed(4)}</td>
          <td className={source.initial[i]===0?'zero-prob':''}>{candidate.initial[i]===0?'0（保留）':candidate.initial[i].toFixed(4)}</td>
        </tr>)}</tbody>
      </table></div>
      <div className="table-scroll"><table>
        <thead><tr><th>转移 从/到</th>{candidate.states.map(s=><th key={s.id}>{s.id}</th>)}</tr></thead>
        <tbody>{candidate.transition.map((row,i)=><tr key={i}>
          <td>{candidate.states[i].id}</td>
          {row.map((p,j)=><td key={j} className={p===0?'zero-prob':''}>{source.transition[i][j]===0?'0（保留）':p.toFixed(4)}</td>)}
        </tr>)}</tbody>
      </table></div>
    </div>
    {fit.candidate_summary&&<p className="method-note">训练对数似然 {fit.candidate_summary.training_log_likelihood.toFixed(6)}；状态有效样本量 {fit.candidate_summary.state_effective_counts.map(n=>n.toFixed(1)).join(' / ')}；方差范围 {fit.candidate_summary.min_variance.toFixed(4)}—{fit.candidate_summary.max_variance.toFixed(4)}。</p>}
  </section>;
}

function IterationTable({iterations}:{iterations:FitIteration[]}){
  return <section className="paper fit-trace">
    <h2>逐次训练记录</h2>
    <p className="method-note">每次迭代记录训练似然、改进量与最小方差。退化状态、方差塌缩或似然下降时停止并丢弃该步，保留最后有效参数。第0行为源参数基线。</p>
    <div className="table-scroll"><table>
      <thead><tr><th>迭代</th><th>阶段</th><th>训练对数似然</th><th>改进量</th><th>状态有效样本量</th><th>最小方差</th><th>最大方差</th><th>说明</th></tr></thead>
      <tbody>{iterations.map(it=><tr key={it.iteration}>
        <td>{it.iteration}</td>
        <td>{it.stage==='accepted'?'已接受':it.stage==='baseline'?'基线':it.stage==='rejected'?'已拒绝':'无效'}</td>
        <td>{num(it.training_log_likelihood,6)}</td>
        <td>{num(it.improvement,6)}</td>
        <td>{it.state_effective_counts?it.state_effective_counts.map(n=>n.toFixed(1)).join(' / '):'—'}</td>
        <td>{num(it.min_variance,6)}</td>
        <td>{num(it.max_variance,6)}</td>
        <td>{it.note}</td>
      </tr>)}</tbody>
    </table></div>
  </section>;
}

export default function FitDetail({fit}:{fit:ModelFit}){
  return <>
    {fit.stale&&<p className="notice">{fit.stale_reason} 旧候选失效，拒绝登记为新状态模型；已有研究继续使用其锁定快照。</p>}
    {fit.registration&&<p className="method-note">已登记为新状态模型：{fit.registration.model_name}（{fit.registration.model_id}），候选参数SHA256 {fit.registration.candidate_config_sha256.slice(0,18)}…</p>}
    <p className="method-note">训练孔：{fit.train_boreholes.join(' / ')}；留出孔：{fit.holdout_boreholes.join(' / ')}。每孔独立起算，共享初始与转移，不跨孔串链；模拟参考truth不参与拟合与留出评估。</p>
    <EvaluationTable title="留出整孔：源参数 vs 候选参数" note="留出孔从未进入拟合：其读数与模拟参考truth不参与参数估计或迭代选择。差值按每点对数似然，越高（越接近0）越吻合；不把似然指数化成准确率。" source={fit.evaluations.source_holdout} candidate={fit.evaluations.candidate_holdout}/>
    <EvaluationTable title="训练孔：拟合内表现（非独立评估）" note="训练孔参与了参数估计，似然仅用于核对迭代是否收敛，不能当作泛化证据。" source={fit.evaluations.source_train} candidate={fit.evaluations.candidate_train}/>
    <ParameterComparison fit={fit}/>
    <IterationTable iterations={fit.iterations}/>
  </>;
}
