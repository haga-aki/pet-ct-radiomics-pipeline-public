"""Guarded smooth SI adaptation: generic, study-path-free implementation.

The shared warp can move internal lobe boundaries within its support.
The guard applies to fitting patch errors, not individual columns or anatomy.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import lsq_linear
import pet_diaphragm_adaptation as core


def anatomy_support(organ, lung, spacing):
    """Support uses CT only, not the PET fitting eligibility mask.

    Columns face lung within 21 mm. Smooth the binary anatomical footprint over
    10 mm, with a 15-mm outward cosine taper; no filling of segmentation holes.
    This field support may span missing observations, not protected organs.
    """
    occupied = organ.any(axis=2)
    top = organ.shape[2] - 1 - np.argmax(organ[:, :, ::-1], axis=2)
    facing = np.zeros_like(occupied)
    for x, y in np.argwhere(occupied):
        k = top[x, y]
        facing[x, y] = lung[x, y, k+1:k+8].sum() >= 2
    sigma = 10 / np.asarray(spacing[:2])
    weights = gaussian_filter(facing.astype(float), sigma, mode='constant')
    heights = gaussian_filter(np.where(facing, (top+.5)*spacing[2], 0.), sigma,
                              mode='constant') / np.maximum(weights, 1e-12)
    if not facing.any():
        return weights, heights
    distance = distance_transform_edt(~facing, sampling=spacing[:2])
    taper = .5*(1+np.cos(np.pi*np.clip(distance/15., 0, 1)))
    # Smooth amplitude, not a hard threshold on observation availability.
    support = np.minimum(weights/.5, 1.) * taper
    return support, heights



def eligible(patches,p):
    return [a for a in patches if a['training'] and abs(a['correction'])<=p.max_shift_mm
            and a['unambiguous_fraction']>=p.minimum_unambiguous_fraction
            and a['crossing_iqr_mm']<=p.maximum_crossing_iqr_mm and a['reliability']>0]


def proposal(patches,support,center,spacing,p):
    accepted=eligible(patches,p);empty=np.zeros_like(support)
    info=dict(activated=False,fitting_patches=len(accepted),model='zero-anchored smooth RBF proposal')
    if not accepted:return (empty,empty.copy(),empty.copy()),info
    target=np.array([a['correction'] for a in accepted]);w=np.array([a['reliability'] for a in accepted])
    baseline=core.weighted_median(target,w);q25=core.weighted_quantile(target,w,.25);q75=core.weighted_quantile(target,w,.75)
    info.update(baseline_mm=baseline,q25_mm=q25,q75_mm=q75)
    if len(accepted)<p.minimum_training_patches or abs(baseline)<=p.organ_bias_gate_mm or not(q25>p.organ_bias_gate_mm or q75< -p.organ_bias_gate_mm):
        return (empty,empty.copy(),empty.copy()),info
    xy=np.array([np.mean([(c[0]*spacing[0],c[1]*spacing[1]) for c in a['columns']],axis=0) for a in accepted])
    step=p.control_spacing_mm
    axes=[np.arange(np.floor(xy[:,i].min()/step)*step,np.ceil(xy[:,i].max()/step)*step+step*.5,step) for i in (0,1)]
    controls=np.array(np.meshgrid(*axes,indexing='ij')).reshape(2,-1).T
    basis=core.radial_basis(xy,controls,p.residual_sigma_mm)
    strength=np.array([np.mean([support[c[0],c[1]] for c in a['columns']]) for a in accepted])
    basis*=strength[:,None];w/=w.sum()
    # Penalize neighbouring control differences; coefficients represent total
    # shifts, so an aligned region can have zero displacement (no global bias).
    distances=np.linalg.norm(controls[:,None]-controls[None,:],axis=2)
    pairs=np.argwhere(np.triu(np.isclose(distances,step),1))
    differences=np.zeros((len(pairs),len(controls)))
    for i,(a,b) in enumerate(pairs):differences[i,a]=1.;differences[i,b]=-1.
    reg=np.r_[np.eye(len(controls))*np.sqrt(.01/len(controls)),
              differences*np.sqrt(.1/max(len(pairs),1))]
    coeff=np.zeros(len(controls))
    for iteration in range(40):
        residual=target-basis@coeff
        robust=np.minimum(1,p.huber_delta_mm/np.maximum(np.abs(residual),1e-12))
        weight=np.sqrt(w*robust)
        solved=lsq_linear(np.r_[basis*weight[:,None],reg],np.r_[target*weight,np.zeros(len(reg))],
                          bounds=(-p.max_shift_mm,p.max_shift_mm),tol=1e-9,max_iter=300)
        if not solved.success:raise RuntimeError(solved.message)
        change=np.max(np.abs(solved.x-coeff));coeff=solved.x
        if change<1e-6:break
    field=empty.copy();indices=np.argwhere(support>0)
    for start in range(0,len(indices),8192):
        a=indices[start:start+8192]
        field[tuple(a.T)]=(core.radial_basis(a*spacing[:2],controls,p.residual_sigma_mm)@coeff)*support[tuple(a.T)]
    info.update(activated=True,controls=len(controls),iterations=iteration+1,
                coefficient_ridge=.01,neighbour_penalty=.1,max_abs_proposal_mm=float(np.max(np.abs(field))))
    return (field,center,support),info


def prepare_columns(patches,labels,protected,shift,center,spacing,p,organ_index):
    selected=eligible(patches,p)
    groups=[];columns=[]
    for patch in selected:
        start=len(columns);columns.extend(patch['columns'])
        groups.append((start,len(columns),patch))
    if not columns:return None
    xy=np.array([(c[0],c[1]) for c in columns])
    return dict(groups=groups,source=labels[xy[:,0],xy[:,1]][:,None,:],
                protected=protected[xy[:,0],xy[:,1]][:,None,:],
                shift=shift[xy[:,0],xy[:,1]][:,None],center=center[xy[:,0],xy[:,1]][:,None],
                top=np.array([c[2] for c in columns]),profiles=np.array([c[3] for c in columns]),
                organ=organ_index,spacing=spacing[2],band=p.band_mm,sigma=p.profile_sigma_voxels)


def evaluate_columns(data,scale):
    if data is None:return []
    a=core.warp_labels(data['source'],data['shift']*scale,data['center'],data['spacing'],data['band'])
    veto=(a!=data['source'])&(a>0)&data['protected'];a[veto]=data['source'][veto]
    mask=a[:,0]==data['organ']
    if not np.all(mask.any(axis=1)):return None
    top=mask.shape[1]-1-np.argmax(mask[:,::-1],axis=1)
    gaps=[]
    for row,k in enumerate(top):
        lung=np.flatnonzero(a[row,0,k+1:k+8]>=3)
        if not len(lung):return None
        gaps.append(lung[0]*data['spacing'])
    deltas=(top-data['top'])*data['spacing'];grid=(np.arange(-9,10)-.5)*data['spacing']
    result=[]
    for start,end,patch in data['groups']:
        translated=np.array([np.interp(grid+d,grid,profile,left=np.nan,right=np.nan)
                             for d,profile in zip(deltas[start:end],data['profiles'][start:end])])
        common=np.all(np.isfinite(translated),axis=0);valid=np.flatnonzero(common)
        if len(valid)<9:return None
        median=np.median(translated[:,common],axis=0)
        filled=np.interp(np.arange(19),valid,median)
        crossing=core.crossings(filled,data['spacing'],data['sigma'])[.5]
        if not np.isfinite(crossing):return None
        error=float(core.interval_error(crossing,0.,np.median(gaps[start:end])))
        before=float(core.interval_error(patch['shift'],0.,patch['gap_mm']))
        result.append(dict(key=[int(v) for v in patch['key']],organ=core.LABELS[data['organ']-1],
                           before_mm=before,after_mm=error,reliability=float(patch['reliability'])))
    return result


def choose_scale(data):
    baseline=[]
    for d in data:
        rows=evaluate_columns(d,0.)
        if rows is None:raise RuntimeError('Invalid original fitting columns')
        baseline.extend(rows)
    if not baseline:return 0.,[],[]
    weights=np.array([r['reliability'] for r in baseline]);weights/=weights.sum()
    before=np.abs([r['before_mm'] for r in baseline]);original_mae=float(weights@before)
    history=[]
    # 1.5 mm is a development safety tolerance (half the study slice interval),
    # not an anatomical accuracy claim. Record all tested scales; no held-out use.
    for scale in np.linspace(1.,0.,21):
        results=[];valid=True
        for d in data:
            rows=evaluate_columns(d,float(scale))
            if rows is None:valid=False;break
            results.extend(rows)
        if not valid:history.append(dict(scale=float(scale),valid=False));continue
        after=np.abs([r['after_mm'] for r in results]);worsening=after-before
        passed=bool(np.max(worsening)<=1.5+1e-6 and weights@after<=original_mae+1e-6)
        history.append(dict(scale=float(scale),valid=True,passed=passed,
            maximum_worsening_mm=float(np.max(worsening)),weighted_mae_mm=float(weights@after)))
        if passed:return float(scale),results,history
    raise RuntimeError('Identity failed safety guard')


def fit_fields(pet, labels, protected, spacing, parameters):
    """Fit and select a scale using eligible fitting patches, never held-out."""
    all_patches, fields, details = {}, [], {}
    for index, organ in enumerate(core.LABELS[:2], 1):
        patches = core.get_patches(pet, labels == index, labels >= 3, spacing, parameters)
        all_patches[organ] = patches
        support, center = anatomy_support(labels == index, labels >= 3, spacing)
        field, info = proposal(patches, support, center, spacing, parameters)
        fields.append(field)
        details[organ] = info
    confidence = fields[0][2] + fields[1][2]
    shift = sum(f[0] * f[2] for f in fields) / np.maximum(confidence, 1e-12)
    center = sum(f[1] * f[2] for f in fields) / np.maximum(confidence, 1e-12)
    data = [prepare_columns(all_patches[organ], labels, protected, shift, center,
                            spacing, parameters, index)
            for index, organ in enumerate(core.LABELS[:2], 1)]
    scale, predicted, history = choose_scale(data)
    for organ in core.LABELS[:2]:
        info = details[organ]
        info["guard_scale"] = scale
        info["proposal_activated"] = info["activated"]
        info["accepted_nonzero"] = bool(np.any(fields[core.LABELS.index(organ)][0] * scale))
    keys = [f"{organ}_{patch['key'][0]}_{patch['key'][1]}"
            for organ in core.LABELS[:2] for patch in eligible(all_patches[organ], parameters)]
    if len(keys) != len(predicted):
        raise AssertionError("Incomplete fitting predictions")
    for key, row in zip(keys, predicted):
        row["patch"] = key
    scaled = [(f[0] * scale, f[1], f[2]) for f in fields]
    guard = dict(scale=scale, scale_history=history, fitting_predictions=predicted,
                 maximum_patch_worsening_mm=1.5,
                 interpretation="Fitting-patch correspondence guard; not column-wise or anatomical accuracy")
    return all_patches, scaled, details, guard


def verify_predictions(patch_rows, guard):
    """Require finite, complete exact agreement with saved-mask evaluation."""
    actual = {row["patch"]: row for row in patch_rows}
    if len(actual) != len(patch_rows):
        raise AssertionError("Duplicate saved patch keys")
    predicted = guard["fitting_predictions"]
    if len({row["patch"] for row in predicted}) != len(predicted):
        raise AssertionError("Duplicate predicted patch keys")
    for expected in predicted:
        got = actual[expected["patch"]]
        values = np.array([got["after_interval_error_mm"], expected["after_mm"]], float)
        if not np.isfinite(values).all() or abs(values[0] - values[1]) > 1e-9:
            raise AssertionError((expected["patch"], values.tolist()))

