from common import *
from collections import Counter
from PIL import Image, ImageOps

def prepare():
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Missing original source CSV: {SOURCE}. Set TICKPREDICT_SOURCE_CSV; see README.md.')
    allrows = rows(SOURCE)
    selected = [r for r in allrows if r['research_image_eligible']=='True']
    assert len({r['tick_id'] for r in allrows})==len(allrows)
    assert not {'282933','283730'} & {r['tick_id'] for r in selected}
    for key in ['tick_id','sha256','similarity_group']:
        seen={}
        for r in selected:
            value=r[key]; assert value, (key,r['tick_id'])
            assert value not in seen or seen[value]==r['provisional_split'], ('split overlap',key,value)
            seen[value]=r['provisional_split']
    # Never decode test images. Their assignments are retained as metadata only.
    active = [r for r in selected if r['provisional_split'] in ['train','dev']]
    for r in active:
        p=resolve_image_path(r['source_path'])
        if not p.is_file():
            raise FileNotFoundError(f'Missing photo: {p}. Set TICKPREDICT_IMAGE_DIR to the original photo folder.')
        assert digest(p)==r['sha256'], ('changed image',p)
        with Image.open(p) as im: ImageOps.exif_transpose(im).convert('RGB').load()
    fields=['tick_id','source_path','sha256','similarity_group','provisional_split','life_stage','engorgement_level']
    manifest=WORK/'data/penn_train_dev.csv'
    manifest.parent.mkdir(parents=True,exist_ok=True)
    (WORK/'notes').mkdir(parents=True,exist_ok=True)
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in active)
    report={'source_sha256':digest(SOURCE),'manifest_sha256':digest(manifest),
        'counts':dict(Counter(r['provisional_split'] for r in selected)),
        'excluded_records':len(allrows)-len(selected),'images_decoded':len(active),'test_images_decoded':0,
        'class_counts':{s:{c:dict(Counter(r[c] for r in active if r['provisional_split']==s)) for c in ['life_stage','engorgement_level']} for s in ['train','dev']},
        'caveats':['Provisional research split preserved; person IDs unavailable.', 'Engorgement labels are source labels; expert rubric not independently confirmed.', 'International data retained separately pending review and harmonization.']}
    write_json(WORK/'notes/preflight.json',report)
    print(json.dumps(report,indent=2));return report
if __name__=='__main__': prepare()
