import {useState} from 'react';
import * as fits from '../api/modelFits';
import * as models from '../api/models';
import * as datasets from '../api/datasets';
import {useResource} from '../hooks/useResource';
import {refreshResources} from '../stores/resources';
import PageTitle from '../components/common/PageTitle';
import Notice from '../components/common/Notice';
import EmptyState from '../components/common/EmptyState';
import FitDetail from '../components/FitDetail';
import type {ModelFit, User} from '../types/entities';

const reasonLabels:Record<string,string>={
  converged:'训练似然改进低于阈值，收敛',
  max_iterations:'达到最大迭代次数',
  likelihood_decrease:'训练似然下降，已丢弃该步',
  degenerate_state:'退化状态或有效样本量趋零',
  variance_collapse:'方差低于下限，疑似塌缩',
  variance_unbounded:'方差超出上限',
  mean_out_of_range:'均值超出允许范围',
  no_feasible_path:'零转移导致训练孔无可行路径',
  numerical_failure:'数值非有限',
  invalid_config:'候选配置不合法',
};

export default function Fits({user}:{user:User}){
  const{data,error}=useResource(fits.list,[]),{data:ds}=useResource(datasets.list,[]),{data:ms}=useResource(models.list,[]);
  const[dataset,setDataset]=useState(''),[model,setModel]=useState(''),[holdout,setHoldout]=useState<string[]>([]),[selected,setSelected]=useState(''),[name,setName]=useState(''),[message,setMessage]=useState('');
  const current=ds.find(d=>d.id===(dataset||ds[0]?.id));
  const boreholes=current?.boreholes??[];
  const sourceModel=ms.find(m=>m.id===(model||ms[0]?.id));
  const chosen=selected?data.find((f:ModelFit)=>f.id===selected):data[0];
  const toggleHole=(hole:string)=>setHoldout(h=>h.includes(hole)?h.filter(x=>x!==hole):[...h,hole]);

  return <>
    <PageTitle chapter="03B / JOINT FIT & HOLDOUT" title="用多个完整钻孔联合拟合，整孔留出评估" description="源人工参数与新的测井数据失配时，只使用训练孔的读数联合估计均值/方差/初始/转移；留出整孔不参与拟合与迭代选择。原配置的精确0概率保持为0。逐次记录训练似然，退化、方差塌缩或似然下降时保留最后有效参数。"/>
    <Notice error={error||message}/>
    {user.role!=='viewer'&&<form className="paper form-grid fit-form" onSubmit={async e=>{
      e.preventDefault();
      try{
        const created=await fits.create({dataset_id:current?.id,source_model_id:sourceModel?.id,holdout_boreholes:holdout});
        setMessage('');setSelected(created.id);setName(created.source_model_name+' · 联合拟合候选');refreshResources();
      }catch(e:any){setMessage(e.message)}
    }}>
      <label>训练用测井数据<select aria-label="训练用测井数据" value={current?.id??''} onChange={e=>{setDataset(e.target.value);setHoldout([])}}>{ds.map(d=><option key={d.id} value={d.id}>{d.name}</option>)}</select></label>
      <label>源状态模型<select aria-label="源状态模型" value={sourceModel?.id??''} onChange={e=>setModel(e.target.value)}>{ms.map(m=><option key={m.id} value={m.id}>{m.name} / v{m.revision}</option>)}</select></label>
      <fieldset className="wide fit-holdout"><legend>整孔留出评估（至少1孔，不参与拟合）</legend><div className="fit-hole-list">{boreholes.map(h=><label key={h}><input type="checkbox" aria-label={'留出孔 '+h} checked={holdout.includes(h)} onChange={()=>toggleHole(h)}/> {h}</label>)}</div><small>训练孔：{boreholes.filter(h=>!holdout.includes(h)).join(' / ')||'—'}（至少2孔）；拟合不使用留出孔读数，也不使用truth。</small></fieldset>
      <button className="primary">联合拟合候选参数</button>
    </form>}
    <section className="paper fit-list"><h2>候选拟合记录</h2>
      {!data.length?<EmptyState text="选择数据集与源模型、勾选留出整孔后发起联合拟合。"/>:<div className="table-scroll"><table><thead><tr><th>拟合</th><th>状态</th><th>停止原因</th><th>接受迭代</th><th>训练孔</th><th>留出孔</th><th>登记</th><th></th></tr></thead>
        <tbody>{data.map((f:ModelFit)=><tr key={f.id} className={chosen?.id===f.id?'selected-fit':''}>
          <td><b>{f.id}</b><small>{f.source_model_name}</small></td>
          <td>{f.status==='completed'?'完成':'失败'}</td>
          <td>{reasonLabels[f.stop_reason]||f.stop_reason}</td>
          <td>{f.accepted_iterations} / {f.max_iterations}（尝试{f.attempted_iterations}）</td>
          <td>{f.train_boreholes.join(' / ')}</td>
          <td>{f.holdout_boreholes.join(' / ')}</td>
          <td>{f.registration?<span className="badge approved">已登记</span>:f.stale?<span className="badge rejected">候选失效</span>:f.status==='completed'?<span className="badge">待登记</span>:<span className="badge failed">不可登记</span>}</td>
          <td><button onClick={()=>setSelected(f.id)}>查看比较</button></td>
        </tr>)}</tbody></table></div>}
    </section>
    {chosen&&<>
      <div className="toolbar fit-register">
        <p className="eyebrow">CANDIDATE {chosen.id} / {chosen.status.toUpperCase()} / {reasonLabels[chosen.stop_reason]||chosen.stop_reason}</p>
        {user.role!=='viewer'&&!chosen.registration&&<><label className="fit-name">登记为新状态模型名称<input aria-label="新状态模型名称" value={name||chosen.source_model_name+' · 联合拟合候选'} onChange={e=>setName(e.target.value)}/></label><button className="primary" disabled={!chosen.registration_allowed} onClick={async()=>{
          try{const m=await fits.register(chosen.id,{name:name||chosen.source_model_name+' · 联合拟合候选'});setMessage('');setSelected(chosen.id);refreshResources()}catch(e:any){setMessage(e.message)}
        }}>登记候选参数为新状态模型</button></>}
        {!chosen.registration_allowed&&!chosen.stale&&chosen.status!=='completed'&&<small>拟合未得到有效候选参数，不能登记；请查看逐次训练记录。</small>}
        {chosen.registration&&<small>已登记为 {chosen.registration.model_name}，状态模型页可见；既有研究继续引用各自的锁定快照。</small>}
      </div>
      <FitDetail fit={chosen}/>
    </>}
  </>;
}
