"""Shared Penn preprocessing with the frozen, training-only foreground proposal.

The otsu/propose algorithm is copied unchanged from crop_candidate.py.
Frozen candidate SHA256: f3e8079dc7f23d992e5380a8f221711d6bf7013dcca4da5732e7cb52da1f37e2
The crop has not been independently validated on development, test, or phones.
prepare_image preserves the original letterbox pipeline when foreground_crop
is absent. No labels, model predictions, or accelerator operations are used.
"""
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

VERSION = "penn_trainonly_foreground_v2"


def otsu(values):
    values = np.clip(values, 0, np.quantile(values, 0.995))
    counts, edges = np.histogram(values, bins=128)
    centres = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(counts)
    sums = np.cumsum(counts * centres)
    total = weight[-1]
    between = (sums[-1] * weight - sums * total) ** 2 / np.maximum(weight * (total - weight), 1)
    return float(centres[np.argmax(between)])


def propose(image):
    """Accept a generous crop only when one central foreground is plausible."""
    w, h = image.size
    small = ImageOps.contain(image, (384, 384), method=Image.Resampling.BILINEAR)
    rgb = np.asarray(small).astype(np.float32) / 255
    sh, sw = rgb.shape[:2]
    border = np.zeros((sh, sw), bool)
    rim = max(4, round(min(sh, sw) * .06))
    border[:rim] = border[-rim:] = True
    border[:, :rim] = border[:, -rim:] = True
    background = np.median(rgb[border], axis=0)
    spread = np.maximum(np.median(np.abs(rgb[border] - background), axis=0) * 1.4826, .04)
    # Border-normalized color contrast plus fine detail captures pale abdomens.
    color = np.sqrt(np.mean(((rgb - background) / spread) ** 2, axis=2))
    detail = np.sqrt(np.mean((rgb - ndimage.gaussian_filter(rgb, (4, 4, 0))) ** 2, axis=2))
    color_threshold = max(2.0, otsu(color))
    detail_threshold = max(.035, otsu(detail))
    mask = (color > color_threshold) | (detail > detail_threshold)
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    components = []
    for label, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None:
            continue
        area = int(np.sum(labels[region] == label))
        if area < max(30, sw * sh * .001):
            continue
        ys, xs = region
        cx = (xs.start + xs.stop) / (2 * sw)
        cy = (ys.start + ys.stop) / (2 * sh)
        distance = ((cx-.5)**2 + (cy-.5)**2)**.5
        score = area * np.exp(-4 * distance**2)
        components.append((float(score), area, label, region, distance))
    full = (0, 0, w, h)
    meta = {'color_threshold': color_threshold, 'detail_threshold': detail_threshold,
            'components': len(components), 'box': list(full), 'area_fraction': 1.,
            'reason': 'no_foreground'}
    if not components:
        return full, meta
    components.sort(reverse=True, key=lambda x: x[0])
    _, area, label, (ys, xs), distance = components[0]
    foreground = labels == label
    # Nearby disconnected legs can belong to the same tick; union their boxes.
    vicinity = ndimage.binary_dilation(foreground, iterations=round(min(sw, sh)*.04))
    for _, other_area, other_label, _, _ in components[1:]:
        if np.any((labels == other_label) & vicinity):
            foreground |= labels == other_label
    yy, xx = np.where(foreground)
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()+1), int(yy.min()), int(yy.max()+1)
    bw, bh = x1-x0, y1-y0
    # Pad relative to both object and frame; never zoom more than 2.2x.
    component_box = [x0, y0, x1, y1]
    pad = max(.35 * max(bw, bh), .10 * max(sw, sh))
    cx, cy = (x0+x1)/2, (y0+y1)/2
    cw, ch = max(bw+2*pad, sw/2.2), max(bh+2*pad, sh/2.2)
    x0, x1 = max(0, cx-cw/2), min(sw, cx+cw/2)
    y0, y1 = max(0, cy-ch/2), min(sh, cy+ch/2)
    bbox = (int(x0*w/sw), int(y0*h/sh), min(w, int(np.ceil(x1*w/sw))), min(h, int(np.ceil(y1*h/sh))))
    fraction = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(w*h)
    competing = len(components)>1 and components[1][0] > components[0][0]*.50
    if bw > .90*sw or bh > .90*sh:
        reason = 'foreground_spans_frame'
    elif area/(sw*sh) < .006:
        reason = 'foreground_too_small'
    elif area/(sw*sh) > .60:
        reason = 'foreground_too_large'
    elif distance > .33:
        reason = 'off_center_uncertain'
    elif competing:
        reason = 'multiple_competing_components'
    elif fraction > .85:
        reason = 'little_crop_benefit'
    else:
        reason = 'crop'
    meta.update({'box': list(bbox if reason == 'crop' else full),
                 'area_fraction': fraction if reason == 'crop' else 1.,
                 'foreground_area_fraction': area/(sw*sh), 'reason': reason,
                 'component_box_small': component_box,
                 'raw_proposal': list(bbox)})
    return tuple(meta['box']), meta




def prepare_image(path, spec):
    """EXIF/RGB -> bottom strip removal -> optional foreground crop -> letterbox.

    ``foreground_crop`` must be absent/None or the exact frozen VERSION. A
    proposal that is uncertain returns the full, bottom-masked image.
    """
    if not isinstance(spec, dict):
        raise ValueError('Image preprocessing must be a dictionary.')
    size = spec.get('image_size')
    bottom = spec.get('bottom_crop_fraction')
    if (type(size) is not int or size < 32
            or not isinstance(bottom, (int, float)) or not 0 <= bottom < 0.5):
        raise ValueError('Unsupported image size or bottom crop fraction.')
    foreground = spec.get('foreground_crop')
    if foreground not in (None, VERSION):
        raise ValueError(f'Unsupported foreground crop version: {foreground!r}')
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    if min(image.size) < 2:
        raise ValueError('Image is too small.')
    image = image.crop((0, 0, image.width, round(image.height * (1 - bottom))))
    if foreground == VERSION:
        box, _ = propose(image)
        image = image.crop(box)
    return ImageOps.pad(image, (size, size), method=Image.Resampling.LANCZOS,
                        color=tuple(spec['pad_color']))
