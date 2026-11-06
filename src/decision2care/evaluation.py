"""Offline evaluation; reference labels are never supplied to inference."""
import math

def auroc(labels, scores):
    if len(labels) != len(scores):
        raise ValueError('Label and score lengths differ.')
    paired = []
    for label, score in zip(labels,scores):
        if label is None:
            continue
        if label not in (0,1) or isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score):
            raise ValueError('Expected binary labels and finite numeric scores.')
        paired.append((float(score),int(label)))
    paired.sort()
    positives = sum(y for _,y in paired)
    negatives = len(paired)-positives
    if not positives or not negatives:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(paired):
        j = i+1
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        rank_sum += ((i+1+j)/2) * sum(y for _,y in paired[i:j])
        i = j
    return (rank_sum-positives*(positives+1)/2)/(positives*negatives)

def evaluate(task, rows, treatment_reference='indication_rubric'):
    if not rows:
        raise ValueError('No evaluation rows.')
    if task == 'diagnosis':
        from .icd_hierarchy import pathology_match
        correct = [pathology_match(r['predicted_title'],r['reference_pathology'])=='strict' for r in rows]
        return dict(task=task,n=len(rows),strict_diagnosis_accuracy=sum(correct)/len(correct))
    if task == 'prognosis':
        labels=[r['label_dead'] for r in rows]
        return dict(task=task,n=len(rows),known_outcomes=sum(x is not None for x in labels),
                    auroc=auroc(labels,[r['mortality_score'] for r in rows]))
    if task != 'treatment' or treatment_reference not in ('indication_rubric','documented_care'):
        raise ValueError('Unknown task or treatment reference.')
    if any(not isinstance(r['reference_positive'],bool) or not isinstance(r['selected'],bool) for r in rows):
        raise ValueError('Treatment flags must be booleans.')
    tp=sum(r['reference_positive'] and r['selected'] for r in rows)
    fp=sum(not r['reference_positive'] and r['selected'] for r in rows)
    fn=sum(r['reference_positive'] and not r['selected'] for r in rows)
    return dict(task=task,reference=treatment_reference,n=len(rows),true_positive=tp,false_positive=fp,false_negative=fn,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
