"""Adapt standard ShareGPT supervision to a private EasyR1 runtime JSONL."""
import argparse
import json
import math
import re
from pathlib import Path


def convert_record(record, index):
    if set(record) != {'conversations', 'images'}:
        raise ValueError('Expected only conversations and images in the public record.')
    messages, images = record['conversations'], record['images']
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError('This adapter requires one human/assistant turn.')
    for message, role in zip(messages, ['human', 'gpt']):
        if (not isinstance(message, dict) or set(message) != {'from', 'value'}
                or message['from'] != role or not isinstance(message['value'], str)
                or not message['value'].strip()):
            raise ValueError('Expected nonempty human/gpt messages with from/value fields.')
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str) or not images[0]:
        raise ValueError('This ROI adapter requires exactly one image path per record.')
    prompt, target = messages[0]['value'], messages[1]['value'].strip()
    if prompt.count('<image>') != len(images):
        raise ValueError('Image placeholder count differs from image count.')
    full = re.fullmatch(
        r'<LOC>\s*(.+?)\s*</LOC>\s*<VIS>\s*(.+?)\s*</VIS>\s*'
        r'<KNO>\s*(.+?)\s*</KNO>\s*<CON>\s*(.+?)\s*</CON>', target, flags=re.S)
    has_tag = re.search(r'</?[A-Za-z][^>]*>', target) is not None
    conclusion = re.fullmatch(r'<CON>\s*(.+?)\s*</CON>', target, flags=re.S)
    boxes = []
    if full:
        if any(not value.strip() or re.search(r'</?[A-Za-z][^>]*>', value) for value in full.groups()):
            raise ValueError('Stage content is empty, nested or repeated.')
        location = json.loads(full.group(1))
        if not isinstance(location, dict) or location.get('bbox_format', 'xyxy_pixel') != 'xyxy_pixel':
            raise ValueError('Localization must use bbox_xyxy_pixel coordinates.')
        boxes = location.get('bbox_xyxy_pixel')
        if not isinstance(boxes, list) or not boxes:
            raise ValueError('Full-LVKC needs at least one source box.')
        for box in boxes:
            if (not isinstance(box, list) or len(box) != 4
                    or not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in box)
                    or box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError('Invalid nonnegative XYXY box.')
        answer = full.group(4).strip()
    elif conclusion and not re.search(r'</?[A-Za-z][^>]*>', conclusion.group(1)):
        answer = conclusion.group(1).strip()
    elif not has_tag:
        answer = target
    else:
        raise ValueError('Incomplete/partial stages: no reliable terminal answer can be derived.')
    if not answer:
        raise ValueError('Empty terminal answer.')
    is_full = full is not None
    ground_truth = {
        'answer': answer, 'task_type': 'full_LVKC' if is_full else 'C',
        'full_LVKC': is_full, 'reference_boxes': boxes,
        'stage_availability': {'L': is_full, 'V': is_full, 'K': is_full, 'C': True},
    }
    return {'sample_id': f'row-{index:07d}', 'prompt': prompt, 'images': list(images),
            'reward_ground_truth': json.dumps(ground_truth, ensure_ascii=False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--image-root', type=Path, help='Defaults to the input JSON directory.')
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Input must be a nonempty JSON array.')
    image_root = (args.image_root or args.input.resolve().parent).resolve()
    converted = []
    for index, record in enumerate(rows, 1):
        try:
            item = convert_record(record, index)
            image_path = (image_root / item['images'][0]).resolve()
            if not image_path.is_file():
                raise ValueError(f'Image file does not exist: {image_path}')
            item['images'] = [str(image_path)]
            converted.append(item)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f'Row {index}: {exc}') from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        for item in converted:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'Prepared {len(converted)} runtime records; source JSON unchanged.')


if __name__ == '__main__':
    main()
