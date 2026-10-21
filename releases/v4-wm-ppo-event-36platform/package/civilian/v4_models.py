"""Dynamic action-conditioned world model for the V4 event PPO study.

Sample contract: ``history`` is the event-model public observation history;
``action`` is a complete environment action (or a candidate wrapper containing
``full_environment_action``). Targets contain ``next_minute`` with entity_ids,
values [x, y, energy, task_progress] and valid masks, plus scalar values/valid
for progress, delay, energy, failure, and communication_risk.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

try:
    from . import event_models as em
except ImportError:
    import event_models as em

V4_MODEL_SCHEMA = "n02-v4-dynamic-action-wm/2"
V4_HEADS = ("progress", "delay", "energy", "failure", "communication_risk")
V4_SEEDS = (8201, 8202, 8203)
ACTION_DIM = 25
ACTION_RELATIONS = 5
NODE_TARGET_DIM = 4


def _action(sample: dict) -> dict:
    value = sample.get("action", sample.get("candidate", {}))
    if isinstance(value, dict) and isinstance(value.get("full_environment_action"), dict):
        value = value["full_environment_action"]
    if not isinstance(value, dict):
        raise ValueError("V4_ACTION_REQUIRED")
    return value


def _dynamic_features(sample: dict, observation: dict, node_ids: list[str], node_types: list[int]):
    """Aggregate a variable set of dispatches/operations onto graph entities."""
    action = _action(sample)
    platform_ids = [str(em._mapping(x).get("id", "")) for x in em._sequence(observation.get("platforms"))]
    task_ids = [str(em._mapping(x).get("id", "")) for x in em._sequence(observation.get("tasks"))]
    offsets = {key: i for i, key in enumerate(node_ids)}
    rows = [[0.0] * ACTION_DIM for _ in node_ids]
    scope = str(action.get("scope", "")).upper()
    scope_code = {"LOCAL": 1, "LOCAL_REPAIR": 1, "GLOBAL": 2, "GLOBAL_REPLAN": 2, "NOOP": 3}.get(scope, 0)
    dispatches, operations = em._sequence(action.get("dispatches")), em._sequence(action.get("queue_operations"))
    op_names = ("transfer", "swap", "insert", "split", "cooperate")
    for i, row in enumerate(rows):
        row[0] = float(scope_code == 1); row[1] = float(scope_code == 2); row[2] = float(scope_code == 3)
        row[3] = float(bool(action.get("noop")))
        row[4] = len(dispatches) / 36.0
        row[5] = len(operations) / 128.0
        row[6] = float(i < len(platform_ids)); row[7] = float(i >= len(platform_ids) and i < len(platform_ids) + len(task_ids))
    for dispatch in dispatches:
        d = em._mapping(dispatch)
        for key in (str(d.get("platform_id", "")), str(d.get("task_id", ""))):
            index = offsets.get(key)
            if index is not None:
                rows[index][8] += 1.0
                rows[index][9] = 1.0
    for operation in operations:
        op = em._mapping(operation); name = str(op.get("op", "")).lower()
        keys = [str(op.get(k, "")) for k in ("task_id", "from_platform", "to_platform", "before_task_id", "other_task_id")]
        for key in set(keys):
            index = offsets.get(key)
            if index is None: continue
            rows[index][10] += 1.0
            if name in op_names: rows[index][11 + op_names.index(name)] += 1.0
            rows[index][16] += em._finite(op.get("share", op.get("workload_share", 0.0)))
            if key == str(op.get("before_task_id", "")) and op.get("before_task_id") is not None:
                rows[index][17] += 1.0
            for feature, field in ((20, "task_id"), (21, "other_task_id"),
                                   (22, "from_platform"), (23, "to_platform"),
                                   (24, "before_task_id")):
                if op.get(field) is not None and key == str(op[field]):
                    rows[index][feature] += 1.0
    for row in rows:
        if not all(math.isfinite(x) for x in row): raise ValueError("V4_ACTION_FEATURE_NONFINITE")
    return rows


def _action_relations(sample: dict, node_ids: list[str]):
    offsets={key:i for i,key in enumerate(node_ids)}
    edges=[[[0.0 for _ in node_ids] for _ in node_ids] for _ in range(ACTION_RELATIONS)]
    action=_action(sample)
    for dispatch in em._sequence(action.get("dispatches")):
        row=em._mapping(dispatch); source=offsets.get(str(row.get("platform_id", ""))); target=offsets.get(str(row.get("task_id", "")))
        if source is not None and target is not None: edges[0][target][source]+=1.0
    for operation in em._sequence(action.get("queue_operations")):
        row=em._mapping(operation); task=offsets.get(str(row.get("task_id", "")))
        other=offsets.get(str(row.get("other_task_id", ""))); source=offsets.get(str(row.get("from_platform", ""))); target=offsets.get(str(row.get("to_platform", "")))
        before=offsets.get(str(row.get("before_task_id", "")))
        if task is not None and source is not None: edges[1][task][source]+=1.0
        if task is not None and target is not None: edges[2][target][task]+=1.0
        if task is not None and other is not None: edges[3][task][other]+=1.0
        if task is not None and before is not None: edges[4][task][before]+=1.0
    return edges


def _graph(sample: dict):
    histories, types, adj, lengths = em._encode_sample_graph(sample)
    observation = em._public_observation(sample["history"][-1])
    platforms = sorted(em._sequence(observation.get("platforms")), key=lambda x: str(em._mapping(x).get("id", "")))
    tasks = sorted(em._sequence(observation.get("tasks")), key=lambda x: str(em._mapping(x).get("id", "")))
    env = sorted(em._sequence(observation.get("environment")), key=lambda x: str(em._mapping(x).get("id", "")))
    links = sorted(em._sequence(observation.get("communication")), key=lambda x: (str(em._mapping(x).get("source", "")), str(em._mapping(x).get("target", ""))))
    ids = [str(em._mapping(x).get("id", "")) for x in platforms + tasks + env] + [f"__communication_{i}" for i in range(len(links))]
    if not ids:
        ids = ["__empty_public_graph"]
    action = _dynamic_features(sample, observation, ids, types[-1])
    action_edges = _action_relations(sample,ids)
    return histories, types, adj, lengths[0], action, action_edges, ids, platforms, tasks


def encode_v4_action_features(observation: dict, action: dict):
    """Encode a complete environment action over event-model public graph nodes."""
    observation=em._public_observation(observation)
    _, node_types, _=em._node_features(observation)
    platforms=sorted(em._sequence(observation.get("platforms")), key=lambda x: str(em._mapping(x).get("id", "")))
    tasks=sorted(em._sequence(observation.get("tasks")), key=lambda x: str(em._mapping(x).get("id", "")))
    env=sorted(em._sequence(observation.get("environment")), key=lambda x: str(em._mapping(x).get("id", "")))
    links=sorted(em._sequence(observation.get("communication")), key=lambda x: (str(em._mapping(x).get("source", "")), str(em._mapping(x).get("target", ""))))
    ids=[str(em._mapping(x).get("id", "")) for x in platforms+tasks+env]+[f"__communication_{i}" for i in range(len(links))]
    if not ids:
        ids=["__empty_public_graph"]; node_types=[em.NODE_TYPES["environment"]]
    if len(ids)!=len(node_types): raise ValueError("V4_ACTION_GRAPH_ALIGNMENT")
    return ids, torch.tensor(_dynamic_features({"action":action},observation,ids,node_types),dtype=torch.float32)


class V4DynamicWorldModel(nn.Module):
    """Two relational graph updates, four-frame GRU, and dynamic action conditioning."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        if hidden_dim != 64: raise ValueError("V4_HIDDEN_DIM_FROZEN_AT_64")
        self.config = {"schema": V4_MODEL_SCHEMA, "hidden_dim": hidden_dim, "graph_layers": 2, "history_length": 4, "action_dim": ACTION_DIM, "node_target_dim": NODE_TARGET_DIM, "heads": list(V4_HEADS)}
        self.register_buffer("node_mean", torch.zeros(em.NODE_FEATURE_DIM)); self.register_buffer("node_scale", torch.ones(em.NODE_FEATURE_DIM))
        self.register_buffer("action_mean", torch.zeros(ACTION_DIM)); self.register_buffer("action_scale", torch.ones(ACTION_DIM))
        self.node_encoder = nn.Sequential(nn.Linear(em.NODE_FEATURE_DIM, 64), nn.Tanh())
        self.type_embedding = nn.Embedding(len(em.NODE_TYPES), 64)
        self.relations = nn.ModuleList(nn.Linear(64, 64, bias=False) for _ in range(em.RELATION_COUNT))
        self.updates = nn.ModuleList(nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.LayerNorm(64)) for _ in range(2))
        self.graph_pool = nn.Sequential(nn.Linear(64 * len(em.NODE_TYPES), 64), nn.Tanh())
        self.temporal = nn.GRU(64, 64, batch_first=True)
        self.action_encoder = nn.Sequential(nn.Linear(ACTION_DIM, 64), nn.Tanh())
        self.action_relations = nn.ModuleList(nn.Linear(64,64,bias=False) for _ in range(ACTION_RELATIONS))
        self.action_update = nn.Sequential(nn.Linear(192,64),nn.Tanh(),nn.LayerNorm(64))
        self.node_head = nn.Sequential(nn.Linear(192, 64), nn.Tanh(), nn.Linear(64, NODE_TARGET_DIM))
        self.global_head = nn.Sequential(nn.Linear(128, 64), nn.Tanh())
        self.heads = nn.ModuleDict({name: nn.Linear(64, 1) for name in V4_HEADS})

    def _graph_seq(self, b):
        x = (b["node_features"] - self.node_mean) / self.node_scale
        mask = b["node_mask"].unsqueeze(-1); batch, steps, nodes, _ = x.shape
        pooled = []
        for t in range(steps):
            h = (self.node_encoder(x[:, t]) + self.type_embedding(b["node_types"][:, t])) * mask[:, t]
            for update in self.updates:
                msg = torch.zeros_like(h)
                for r, transform in enumerate(self.relations): msg = msg + transform(torch.matmul(b["adjacency"][:, t, r], h))
                h = update(torch.cat((h, msg), -1)) * mask[:, t]
            by_type=[]
            for typ in range(len(em.NODE_TYPES)):
                m=((b["node_types"][:,t] == typ) & b["node_mask"][:,t]).to(h.dtype).unsqueeze(-1)
                by_type.append((h*m).sum(1)/m.sum(1).clamp_min(1.0))
            pooled.append(self.graph_pool(torch.cat(by_type, -1)))
        seq=torch.stack(pooled,1)
        lens=b["history_lengths"].cpu().clamp(1, steps)
        from torch.nn.utils.rnn import pack_padded_sequence
        _, hidden=self.temporal(pack_padded_sequence(seq,lens,batch_first=True,enforce_sorted=False))
        return hidden[-1]

    def forward(self, b):
        required={"node_features","node_types","node_mask","adjacency","history_lengths","action_features","action_adjacency"}
        if set(b) != required: raise ValueError("V4_BATCH_FIELDS")
        graph=self._graph_seq(b); current=b["node_features"][:,-1]
        types=b["node_types"][:,-1]; mask=b["node_mask"][:,-1]
        node=(current-self.node_mean)/self.node_scale
        h=self.node_encoder(node)+self.type_embedding(types)
        for update in self.updates:
            aggregate=torch.zeros_like(h)
            for r, transform in enumerate(self.relations):
                aggregate=aggregate+transform(torch.matmul(b["adjacency"][:,-1,r],h))
            h=update(torch.cat((h,aggregate),-1))*mask.unsqueeze(-1)
        a=self.action_encoder((b["action_features"]-self.action_mean)/self.action_scale)
        a=(a+h)*mask.unsqueeze(-1)
        action_message=torch.zeros_like(a)
        for relation,transform in enumerate(self.action_relations):
            action_message=action_message+transform(torch.matmul(b["action_adjacency"][:,relation],a))
        a=self.action_update(torch.cat((h,a,action_message),-1))*mask.unsqueeze(-1)
        conditioned=torch.cat((h,a,graph[:,None,:].expand(-1,h.shape[1],-1)), -1)
        node_pred=self.node_head(conditioned).masked_fill(~mask.unsqueeze(-1),0.0)
        action_pool=(a*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        context=self.global_head(torch.cat((graph,action_pool),-1))
        return {"next_minute":node_pred, **{name: layer(context).squeeze(-1) for name,layer in self.heads.items()}}


def collate_v4_samples(samples: Sequence[dict]):
    if not samples: raise ValueError("V4_SAMPLES_REQUIRED")
    graphs=[_graph(x) for x in samples]; max_nodes=max(len(x[1][-1]) for x in graphs)
    B=len(samples); T=em.HISTORY_LENGTH; N=max_nodes
    nf=torch.zeros(B,T,N,em.NODE_FEATURE_DIM); nt=torch.zeros(B,T,N,dtype=torch.long); nm=torch.zeros(B,T,N,dtype=torch.bool)
    adj=torch.zeros(B,T,em.RELATION_COUNT,N,N); af=torch.zeros(B,N,ACTION_DIM); action_adj=torch.zeros(B,ACTION_RELATIONS,N,N); lengths=torch.zeros(B,dtype=torch.long)
    node_y=torch.zeros(B,N,NODE_TARGET_DIM); node_m=torch.zeros(B,N,NODE_TARGET_DIM,dtype=torch.bool)
    gy={h:torch.zeros(B) for h in V4_HEADS}; gm={h:torch.zeros(B,dtype=torch.bool) for h in V4_HEADS}
    ids=[]
    for bi,(sample,g) in enumerate(zip(samples,graphs)):
        histories,types,adjs,length,actions,action_edges,entity_ids,platforms,tasks=g; ids.append(entity_ids); lengths[bi]=length
        for t in range(T):
            n=len(histories[t]); nf[bi,t,:n]=torch.tensor(histories[t]); nt[bi,t,:n]=torch.tensor(types[t]); nm[bi,t,:n]=True; adj[bi,t,:,:n,:n]=torch.tensor(adjs[t])
        n=len(entity_ids); af[bi,:n]=torch.tensor(actions); action_adj[bi,:,:n,:n]=torch.tensor(action_edges)
        targets=em._mapping(sample.get("targets")); ny=em._mapping(targets.get("next_minute"))
        target_ids=ny.get("entity_ids", []); values=ny.get("values", []); valid=ny.get("valid", [])
        if not (len(target_ids)==len(values)==len(valid)): raise ValueError("V4_NODE_TARGET_ALIGNMENT")
        lookup={key:i for i,key in enumerate(entity_ids)}
        for key,value,valid_row in zip(target_ids,values,valid):
            index=lookup.get(str(key))
            if index is None: continue
            if len(value)!=NODE_TARGET_DIM or len(valid_row)!=NODE_TARGET_DIM: raise ValueError("V4_NODE_TARGET_WIDTH")
            node_y[bi,index]=torch.tensor([em._finite(x,float("nan")) for x in value]); node_m[bi,index]=torch.tensor(valid_row,dtype=torch.bool)
        for head in V4_HEADS:
            entry=em._mapping(targets.get(head)); value=entry.get("value",entry.get("values")); valid=entry.get("valid",True)
            if isinstance(value,(list,tuple)): value=value[0] if value else None
            if value is not None:
                gy[head][bi]=em._finite(value,float("nan")); gm[head][bi]=bool(valid[0] if isinstance(valid,(list,tuple)) else valid)
    for tensor in (nf,af,node_y,*gy.values()):
        if not bool(torch.isfinite(tensor).all()): raise ValueError("V4_NONFINITE_TARGET_OR_FEATURE")
    return {"node_features":nf,"node_types":nt,"node_mask":nm,"adjacency":adj,"history_lengths":lengths,"action_features":af,"action_adjacency":action_adj}, {"next_minute":node_y,"next_minute_valid":node_m,**gy,**{f"{h}_valid":gm[h] for h in V4_HEADS},"entity_ids":ids}


def _fit_norm(model, samples):
    sums = {name: None for name in ("node", "action")}
    squares = {name: None for name in ("node", "action")}
    counts = {name: 0 for name in ("node", "action")}
    for sample in samples:
        features = collate_v4_samples([sample])[0]
        for name, key in (("node", "node_features"), ("action", "action_features")):
            mask = features["node_mask"] if name == "node" else features["node_mask"][:, -1]
            rows = features[key][mask].to(torch.float64)
            counts[name] += len(rows)
            row_sum, row_squares = rows.sum(0), rows.square().sum(0)
            sums[name] = row_sum if sums[name] is None else sums[name] + row_sum
            squares[name] = row_squares if squares[name] is None else squares[name] + row_squares
    if any(count == 0 for count in counts.values()):
        raise ValueError("V4_NORMALIZATION_WITHOUT_PUBLIC_NODES")
    for name in ("node", "action"):
        mean = sums[name] / counts[name]
        scale = (squares[name] / counts[name] - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
        getattr(model, name + "_mean").copy_(mean.to(getattr(model, name + "_mean").dtype))
        getattr(model, name + "_scale").copy_(scale.to(getattr(model, name + "_scale").dtype))
    return {"source":"training_samples_only","node_mean":model.node_mean.cpu().tolist(),"node_scale":model.node_scale.cpu().tolist(),"action_mean":model.action_mean.cpu().tolist(),"action_scale":model.action_scale.cpu().tolist()}


def _identity(samples):
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for sample in samples:
        if not first:
            digest.update(b",")
        first = False
        row = {key: sample.get(key) for key in ("sample_id", "split", "parent_id", "history", "action", "targets")}
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                allow_nan=False).encode())
    digest.update(b"]")
    return digest.hexdigest()


def _validation_loss(model, samples, device):
    model.eval(); losses=[]
    with torch.no_grad():
        for sample in samples:
            b,y=collate_v4_samples([sample]); b={k:v.to(device) for k,v in b.items()}; y={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in y.items()}; pred=model(b)
            terms=[]; valid=y["next_minute_valid"]
            if valid.any(): terms.append(F.smooth_l1_loss(pred["next_minute"][valid],y["next_minute"][valid]))
            for head in V4_HEADS:
                mask=y[f"{head}_valid"]
                if mask.any():
                    terms.append(F.binary_cross_entropy_with_logits(pred[head][mask],y[head][mask].clamp(0,1)) if head in ("failure","communication_risk") else F.smooth_l1_loss(pred[head][mask],y[head][mask]))
            if terms: losses.append(float(torch.stack(terms).mean().cpu()))
    if not losses: raise ValueError("V4_VALIDATION_WITHOUT_LABELS")
    return sum(losses)/len(losses)


def train_v4_ensemble(samples: Sequence[dict], output_dir: str|Path, config: dict|None=None, *, validation_samples: Sequence[dict]):
    config=dict(config or {}); seeds=tuple(config.get("seeds",V4_SEEDS)); epochs=int(config.get("max_epochs",100)); batch=int(config.get("batch_size",8)); patience=int(config.get("patience",10)); lr=float(config.get("learning_rate",.001)); device=torch.device(config.get("device","cuda:0"))
    if seeds!=V4_SEEDS or not samples or any(s.get("split")!="train" for s in samples) or not validation_samples or any(s.get("split")!="validation" for s in validation_samples): raise ValueError("V4_TRAIN_VALIDATION_SPLIT_OR_SEEDS")
    if device.type!="cuda": raise RuntimeError("V4_TRAINING_REQUIRES_CUDA")
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("V4_CUDA_REQUIRED_BUT_UNAVAILABLE")
    identity=_identity(samples); validation_identity=_identity(validation_samples); root=Path(output_dir); root.mkdir(parents=True,exist_ok=True); paths=[]
    for seed in seeds:
        random.seed(seed); torch.manual_seed(seed)
        model=V4DynamicWorldModel().to(device); norm=_fit_norm(model,samples); opt=torch.optim.Adam(model.parameters(),lr=lr)
        generator=torch.Generator().manual_seed(seed+17); best=math.inf; best_state=None; best_optimizer=None; stale=0; logs=[]; updates=0; best_updates=0
        torch.cuda.reset_peak_memory_stats(device); started=time.perf_counter()
        gpu_start=torch.cuda.Event(enable_timing=True); gpu_end=torch.cuda.Event(enable_timing=True); gpu_start.record()
        for epoch in range(min(epochs,100)):
            order=torch.randperm(len(samples),generator=generator).tolist(); losses=[]; model.train()
            for start in range(0,len(order),batch):
                selected=[samples[i] for i in order[start:start+batch]]; b,y=collate_v4_samples(selected); b={k:v.to(device) for k,v in b.items()}; y={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in y.items()}; pred=model(b)
                node_valid=y["next_minute_valid"]; terms=[]
                if node_valid.any(): terms.append(F.smooth_l1_loss(pred["next_minute"][node_valid],y["next_minute"][node_valid]))
                for head in V4_HEADS:
                    valid=y[f"{head}_valid"]
                    if valid.any():
                        if head in ("failure","communication_risk"): terms.append(F.binary_cross_entropy_with_logits(pred[head][valid],y[head][valid].clamp(0,1)))
                        else: terms.append(F.smooth_l1_loss(pred[head][valid],y[head][valid]))
                if not terms: raise ValueError("V4_BATCH_WITHOUT_LABELS")
                loss=torch.stack(terms).mean()
                if not torch.isfinite(loss): raise FloatingPointError("V4_NONFINITE_LOSS")
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); updates+=1; losses.append(float(loss.detach().cpu()))
            mean=sum(losses)/len(losses); val_loss=_validation_loss(model,validation_samples,device); logs.append({"epoch":epoch+1,"train_loss":mean,"validation_loss":val_loss})
            if val_loss < best-1e-8:
                best=val_loss; best_state=copy.deepcopy(model.state_dict()); best_optimizer=copy.deepcopy(opt.state_dict()); best_updates=updates; stale=0
            else: stale+=1
            if stale>=patience: break
        if best_state is None: raise AssertionError("V4_NO_CHECKPOINT")
        model.load_state_dict(best_state); model.eval()
        path=root/f"v4_world_model_seed_{seed}.pt"
        gpu_end.record(); torch.cuda.synchronize(device); wall_elapsed=time.perf_counter()-started; gpu_elapsed=gpu_start.elapsed_time(gpu_end)/1000.0; peak=torch.cuda.max_memory_allocated(device)
        torch.save({"schema":V4_MODEL_SCHEMA,"seed":seed,"config":model.config,"data_identity":identity,"validation_identity":validation_identity,"normalization":norm,"model_state":{k:v.cpu() for k,v in best_state.items()},"optimizer_state":best_optimizer,"torch_rng_state":torch.get_rng_state(),"cuda_rng_state_all":torch.cuda.get_rng_state_all(),"python_rng_state":random.getstate(),"training_history":logs,"optimizer_update_count":updates,"optimizer_update_count_at_best":best_updates,"gpu_seconds":gpu_elapsed,"training_wall_seconds":wall_elapsed,"peak_gpu_memory_bytes":peak},path); paths.append(str(path.resolve()))
    return {"status":"complete","schema":V4_MODEL_SCHEMA,"data_identity":identity,"validation_identity":validation_identity,"checkpoint_paths":paths,"seeds":list(seeds),"epochs_max":epochs,"batch_size":batch,"patience":patience,"learning_rate":lr}


def load_v4_ensemble(paths: Sequence[str|Path], device: str|torch.device="cpu"):
    if len(paths)!=3: raise ValueError("V4_ENSEMBLE_REQUIRES_THREE_CHECKPOINTS")
    models=[]; metadata=[]
    for expected,path in zip(V4_SEEDS,paths):
        payload=torch.load(path,map_location="cpu",weights_only=False)
        if payload.get("schema")!=V4_MODEL_SCHEMA or payload.get("seed")!=expected: raise ValueError("V4_CHECKPOINT_SCHEMA_OR_SEED")
        model=V4DynamicWorldModel(); model.load_state_dict(payload["model_state"],strict=True); model.to(device).eval().requires_grad_(False)
        models.append(model); metadata.append({k:payload[k] for k in ("seed","data_identity","normalization","training_history")})
    return models,metadata


def predict_v4_ensemble(models: Sequence[V4DynamicWorldModel], samples: Sequence[dict], device="cpu"):
    if len(models)!=3: raise ValueError("V4_ENSEMBLE_REQUIRED")
    b,y=collate_v4_samples(samples); b={k:v.to(device) for k,v in b.items()}; outputs=[]
    for model in models:
        model.eval()
        with torch.no_grad(): outputs.append(model.to(device)(b))
    entity_mask=torch.zeros(b["node_mask"].shape[0],b["node_mask"].shape[2],dtype=torch.bool)
    for i,ids in enumerate(y["entity_ids"]): entity_mask[i,:len(ids)]=True
    result={"entity_ids":y["entity_ids"],"entity_mask":entity_mask,"next_minute":torch.stack([x["next_minute"].cpu() for x in outputs]).mean(0)}
    result["next_minute_std"]=torch.stack([x["next_minute"].cpu() for x in outputs]).std(0,unbiased=False)
    result["next_minute_variance"]=result["next_minute_std"].square()
    for head in V4_HEADS:
        members=torch.stack([x[head].cpu() for x in outputs])
        if head in ("failure","communication_risk"): members=members.sigmoid()
        result[head]=members.mean(0)
        result[f"{head}_std"]=members.std(0,unbiased=False)
        result[f"{head}_variance"]=result[f"{head}_std"].square()
    return result


def predict_v4_action_catalog(models: Sequence[V4DynamicWorldModel], history: Sequence[dict], actions: Sequence[dict], device="cpu"):
    """Score each complete public action against a shared observation history.

    Returns predictions in input action order; each candidate receives scalar
    ten-minute heads and its dynamic per-entity one-minute predictions.
    """
    if not history or not actions: raise ValueError("V4_ACTION_CATALOG_REQUIRED")
    samples=[{"history":list(history),"action":action} for action in actions]
    return predict_v4_ensemble(models,samples,device=device)


_NEXT_MINUTE_DIMENSIONS=("x_delta_km","y_delta_km","energy_delta","progress_delta")


def _public_kinematics_baseline(sample: dict, entity_ids: list[str]):
    observation=em._public_observation(sample["history"][-1])
    platform_rows={str(em._mapping(row).get("id")):em._mapping(row) for row in em._sequence(observation.get("platforms"))}
    task_rows={str(em._mapping(row).get("id")):em._mapping(row) for row in em._sequence(observation.get("tasks"))}
    depot_rows={str(em._mapping(row).get("id")):em._mapping(row) for row in em._sequence(observation.get("depots"))}
    queues={str(key):list(value) for key,value in em._mapping(observation.get("queues")).items() if isinstance(value,list)}
    for platform_id in platform_rows: queues.setdefault(platform_id,[])
    action=_action(sample)
    for operation in em._sequence(action.get("queue_operations")):
        row=em._mapping(operation); kind=str(row.get("op","")); task_id=str(row.get("task_id",""))
        if kind in {"split","cooperate"}:
            source=str(row.get("from_platform",""))
            queues[source]=[item for item in queues.get(source,[]) if str(item)!=task_id]
        elif kind=="swap":
            other=str(row.get("other_task_id",""))
            for queue in queues.values():
                for index,item in enumerate(queue):
                    if str(item)==task_id: queue[index]=other
                    elif str(item)==other: queue[index]=task_id
        else:
            for queue in queues.values():
                queue[:]=[item for item in queue if str(item)!=task_id]
            target=str(row.get("to_platform","")); queue=queues.setdefault(target,[])
            anchor=row.get("before_task_id") if kind=="insert" else None
            if anchor is not None and str(anchor) in [str(item) for item in queue]:
                queue.insert(next(i for i,item in enumerate(queue) if str(item)==str(anchor)),task_id)
            else: queue.append(task_id)
    for dispatch in em._sequence(action.get("dispatches")):
        row=em._mapping(dispatch); task_id=str(row.get("task_id","")); platform_id=str(row.get("platform_id",""))
        for queue in queues.values(): queue[:]=[item for item in queue if str(item)!=task_id]
        queues.setdefault(platform_id,[]).insert(0,task_id)

    values={key:[0.0]*NODE_TARGET_DIM for key in entity_ids}
    for platform_id,platform in platform_rows.items():
        if platform_id not in values or platform.get("failed"): continue
        task_id=platform.get("active_task_id")
        if not task_id:
            for queued in queues.get(platform_id,[]):
                candidate=task_rows.get(str(queued))
                if candidate is None or candidate.get("status") in {"completed","failed","expired","cancelled","split_parent"}: continue
                if candidate.get("assignee") not in (None,platform_id): continue
                predecessors=em._sequence(candidate.get("predecessors",em._mapping(observation.get("predecessors")).get(str(queued))))
                if any(task_rows.get(str(predecessor),{}).get("status")!="completed" for predecessor in predecessors): break
                task_id=str(queued); break
        task=task_rows.get(str(task_id)) if task_id is not None else None
        if task is None or str(task.get("id")) not in values: continue
        required="inspect" if task.get("kind")=="inspection" else "deliver"
        if required not in em._sequence(platform.get("capabilities")): continue
        phase=str(task.get("phase","")); target=None
        if task.get("kind")=="inspection":
            if phase in {"observe","completed"}: continue
            target=task.get("position")
        elif phase in {"to_delivery_site","in_transit"}:
            target=task.get("position")
        elif phase in {"to_depot","assigned","pending","released",""}:
            depot=depot_rows.get(str(task.get("depot_id"))); target=depot.get("position") if depot else None
        elif phase in {"loading","unloading"}:
            values[platform_id][2]=-max(0.0,em._finite(platform.get("service_energy_per_min")))
            continue
        if target is None: continue
        start=em._xy(platform.get("position")); destination=em._xy(target)
        dx,dy=destination[0]-start[0],destination[1]-start[1]; distance=math.hypot(dx,dy)
        speed=max(0.0,em._finite(platform.get("speed_km_min"))); rate=max(0.0,em._finite(platform.get("energy_per_km")))
        travel=min(distance,speed)
        if rate>0: travel=min(travel,max(0.0,em._finite(platform.get("energy")))/rate)
        if distance>0 and travel>0:
            fraction=travel/distance
            values[platform_id][0]=dx*fraction; values[platform_id][1]=dy*fraction
            values[platform_id][2]=-travel*rate
    return torch.tensor([values[key] for key in entity_ids],dtype=torch.float32)


def _metric_summary(accumulator: dict):
    count=len(accumulator["absolute"])
    return {"valid_count":count,
            "valid_sample_count":len(accumulator["samples"]),
            "mae":sum(accumulator["absolute"])/count if count else None,
            "rmse":math.sqrt(sum(accumulator["squared"])/count) if count else None,
            "brier":sum(accumulator["brier"])/count if accumulator["brier"] else None}


def evaluate_v4_predictions(models: Sequence[V4DynamicWorldModel], samples: Sequence[dict], *,
                            split: str="validation", device: str|torch.device="cpu", batch_size: int=16):
    """Compare ensemble heads to masked labels and public-only baselines.

    The kinematic baseline uses the current public queue/action and nominal
    public speed/energy rates for one minute. Ten-minute scalar outcomes use a
    zero-change persistence heuristic; no future observations enter either baseline.
    """
    if len(models)!=3 or split not in {"validation","test"} or not samples or batch_size<1:
        raise ValueError("V4_PREDICTION_EVALUATION_INPUT")
    if any(row.get("split")!=split for row in samples): raise ValueError("V4_PREDICTION_EVALUATION_SPLIT")
    methods=("model","persistence","public_kinematics")
    def new_accumulator(): return {method:{"absolute":[],"squared":[],"brier":[],"samples":set()} for method in methods}
    target_specs={name:new_accumulator() for name in (*_NEXT_MINUTE_DIMENSIONS,*V4_HEADS)}
    eval_models=[model.to(device).eval() for model in models]
    for offset in range(0,len(samples),batch_size):
        chunk=[samples[index] for index in range(offset,min(len(samples),offset+batch_size))]
        batch,target=collate_v4_samples(chunk); batch={key:value.to(device) for key,value in batch.items()}
        with torch.no_grad(): outputs=[model(batch) for model in eval_models]
        node_prediction=torch.stack([row["next_minute"].detach().cpu() for row in outputs]).mean(0)
        scalar_prediction={}
        for head in V4_HEADS:
            members=torch.stack([row[head].detach().cpu() for row in outputs])
            scalar_prediction[head]=members.sigmoid().mean(0) if head in ("failure","communication_risk") else members.mean(0)
        for sample_index,sample in enumerate(chunk):
            ids=target["entity_ids"][sample_index]
            public_baseline=_public_kinematics_baseline(sample,ids)
            valid=target["next_minute_valid"][sample_index]
            truth=target["next_minute"][sample_index]
            for dim,name in enumerate(_NEXT_MINUTE_DIMENSIONS):
                mask=valid[:,dim]
                if not bool(mask.any()): continue
                actual=truth[mask,dim].float(); model_values=node_prediction[sample_index,mask,dim]
                persist=torch.zeros_like(actual); motion=public_baseline[mask,dim]
                for label,prediction in (("model",model_values),("persistence",persist),("public_kinematics",motion)):
                    delta=prediction-actual; entry=target_specs[name][label]
                    entry["absolute"].extend(delta.abs().tolist()); entry["squared"].extend(delta.square().tolist())
                    entry["samples"].add(offset+sample_index)
            for head in V4_HEADS:
                mask=target[f"{head}_valid"][sample_index]
                if not bool(mask): continue
                actual=target[head][sample_index].float().reshape(1)
                model_value=scalar_prediction[head][sample_index].reshape(1)
                zero=torch.zeros_like(actual)
                for label,prediction in (("model",model_value),("persistence",zero),("public_kinematics",zero)):
                    difference=(prediction-actual); entry=target_specs[head][label]
                    entry["absolute"].extend(difference.abs().tolist())
                    entry["squared"].extend(difference.square().tolist())
                    if head in ("failure","communication_risk"):
                        entry["brier"].extend(difference.square().tolist())
                    entry["samples"].add(offset+sample_index)
    result={"schema":"n02-v4-world-model-prediction-evaluation/1","status":"complete","split":split,
            "sample_count":len(samples),"prediction_horizon_minutes":10,
            "baseline_contract":{"persistence":"predict zero change for every delta/outcome head",
                "public_kinematics":"one-minute straight-line motion to the current public active/queue/action target, bounded by public speed and energy rate; static task positions and all other components use zero change",
                "ten_minute_scalar_heuristic":"both baselines predict zero change for ten-minute aggregate heads; no future state or event tape is used"},"heads":{}}
    for head,by_method in target_specs.items():
        if head in _NEXT_MINUTE_DIMENSIONS:
            result["heads"].setdefault("next_minute",{"dimensions":{}})["dimensions"][head]={}
        cell=(result["heads"].setdefault("next_minute",{"dimensions":{}})["dimensions"][head]
              if head in _NEXT_MINUTE_DIMENSIONS else result["heads"].setdefault(head,{}))
        for method,accumulator in by_method.items():
            cell[method]=_metric_summary(accumulator)
        if head in ("failure","communication_risk") and cell["model"]["valid_count"]:
            cell["metric_note"]="Brier score is reported for the probability-valued failure and communication-risk targets."
    return result
