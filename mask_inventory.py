"""Expected mask inventory from the installed TotalSegmentator task map."""


def expected_masks(config, modality, class_map=None):
    if class_map is None:
        from totalsegmentator.map_to_binary import class_map
    segmentation = config.get('segmentation', {})
    tasks = segmentation.get('tasks', {})
    task = tasks.get(modality, 'total')
    if modality in ('PET', 'PT', 'SPECT') or task == 'use_ct_mask':
        task = tasks.get('CT', 'total')
    if task not in class_map:
        raise ValueError(f'No expected label map for task {task}')
    names = list(class_map[task].values())
    subset = segmentation.get('roi_subset')
    if subset:
        unknown = set(subset) - set(names)
        if unknown:
            raise ValueError(f'Unknown roi_subset labels: {sorted(unknown)}')
        names = [name for name in names if name in subset]
    if not names:
        raise ValueError('Expected mask inventory is empty')
    return names
