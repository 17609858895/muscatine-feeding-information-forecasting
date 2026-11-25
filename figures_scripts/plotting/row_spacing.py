"""Pack complete figure rows without changing data axes or scientific artists.

The occupied area includes ticks, labels, legends, colourbars and panel letters.
All shifts are translations in the original physical canvas. Tight vector/raster
exports therefore become shorter while every data panel keeps its original size.
"""
import hashlib
import numpy as np
from matplotlib.transforms import Bbox


def _scientific_state(fig):
    h = hashlib.sha256()
    def add(value):
        a = np.asanyarray(value)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes() if a.dtype.kind != 'O' else repr(value).encode())
    for ax in fig.axes:
        add(ax.get_xlim()); add(ax.get_ylim())
        h.update(str((ax.get_xscale(), ax.get_yscale())).encode())
        for line in ax.lines:
            add(line.get_xdata(orig=False)); add(line.get_ydata(orig=False))
        for artist in ax.collections:
            add(artist.get_offsets())
            if artist.get_array() is not None: add(artist.get_array())
            for path in artist.get_paths(): add(path.vertices)
        for artist in ax.images: add(artist.get_array())
        for patch in ax.patches:
            add(patch.get_path().vertices)
            add(patch.get_patch_transform().get_matrix())
        for text in ax.texts:
            h.update(str((text.get_text(), text.get_position(), text.get_fontsize())).encode())
    return h.hexdigest()


def pack_rows(fig, groups, panel_footprint, alignment, row_gap_pt=4.0,
              ornament_gap_pt=6.0):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    height_pt = fig.get_figheight() * 72
    gap = row_gap_pt / height_pt
    ornament_gap = ornament_gap_pt / height_pt
    original_size = {ax: (ax.get_position().width, ax.get_position().height)
                     for ax in fig.axes}
    science = _scientific_state(fig)
    original_content_height = fig.get_tightbbox(renderer).height * 72
    slots = [spec.get_position(fig).frozen() for spec, _ in groups]
    rows = []
    for i, slot in enumerate(slots):
        for row in rows:
            if abs(row['slot_top'] - slot.y1) < .012:
                row['panels'].append(i)
                break
        else:
            rows.append({'slot_top': slot.y1, 'panels': [i],
                         'texts': [], 'legends': [], 'extras': []})
    rows.sort(key=lambda row: -row['slot_top'])

    def bbox(artist):
        return artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())

    def rowbox(row):
        boxes = [panel_footprint(fig, groups[i][1], renderer) for i in row['panels']]
        boxes += [bbox(t) for t in row['texts'] + row['legends'] if t.get_visible()]
        boxes += [ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
                  for ax in row['extras'] if ax.get_visible()]
        return Bbox.union([b for b in boxes if b is not None])

    axes_bounds = [rowbox(row) for row in rows]
    # Panel letters are figure texts placed by the calibrated panel-letter helper.
    for text in fig.texts:
        center = (bbox(text).y0 + bbox(text).y1) / 2
        row_index = min(range(len(rows)), key=lambda i: abs(center - axes_bounds[i].y1))
        rows[row_index]['texts'].append(text)

    def shift_axes(axes, dy):
        positions = {ax: ax.get_position().frozen() for ax in axes}
        for ax, p in positions.items():
            participates = ax.get_in_layout()
            ax.set_position([p.x0, p.y0 + dy, p.width, p.height], which='both')
            ax.set_in_layout(participates)

    def shift_legend(legend, dy):
        anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        legend.set_bbox_to_anchor((anchor.x0, anchor.y0 + dy,
                                   anchor.width, anchor.height), transform=fig.transFigure)

    ornaments = []
    # Global keys are brought next to the first/last row before rows are packed.
    # Axes legends already participate in their own complete panel footprints.
    for legend in fig.legends:
        bounds = [rowbox(row) for row in rows]
        b = bbox(legend)
        if b.y0 >= bounds[0].y1:
            index = 0
            dy = min(0, bounds[0].y1 + ornament_gap - b.y0)
            old_gap = (b.y0 - bounds[0].y1) * height_pt
            location = 'header key'
        elif b.y1 <= bounds[-1].y0:
            index = len(rows) - 1
            dy = max(0, bounds[-1].y0 - ornament_gap - b.y1)
            old_gap = (bounds[-1].y0 - b.y1) * height_pt
            location = 'footer key'
        else:
            index = min(range(len(rows)), key=lambda i:
                        abs((b.y0 + b.y1) / 2 - (bounds[i].y0 + bounds[i].y1) / 2))
            dy = 0
            old_gap = None
            location = 'within row'
        shift_legend(legend, dy)
        rows[index]['legends'].append(legend)
        ornaments.append({'type': location, 'before_gap_pt': old_gap,
                          'removed_pt': abs(dy) * height_pt})
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()

    owned = {ax for _, axes in groups for ax in axes if ax.get_in_layout()}
    for ax in fig.axes:
        if ax in owned or not ax.get_visible(): continue
        b = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        bounds = [rowbox(row) for row in rows]
        if b.y1 <= bounds[-1].y0:
            index = len(rows) - 1
            dy = max(0, bounds[-1].y0 - ornament_gap - b.y1)
            old_gap = (bounds[-1].y0 - b.y1) * height_pt
        else:
            index = min(range(len(rows)), key=lambda i:
                        abs((b.y0 + b.y1) / 2 - (bounds[i].y0 + bounds[i].y1) / 2))
            dy = 0
            old_gap = None
        shift_axes([ax], dy)
        rows[index]['extras'].append(ax)
        ornaments.append({'type': 'shared colourbar', 'before_gap_pt': old_gap,
                          'removed_pt': abs(dy) * height_pt})
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()

    initial = [rowbox(row) for row in rows]
    shifts = []
    for index in range(1, len(rows)):
        upper = rowbox(rows[index - 1])
        lower = rowbox(rows[index])
        # Already compact rows retain their authored separation.
        dy = max(0, upper.y0 - gap - lower.y1)
        row = rows[index]
        axes = list(dict.fromkeys([ax for i in row['panels'] for ax in groups[i][1]
                                  if ax.get_in_layout()] + row['extras']))
        shift_axes(axes, dy)
        for text in row['texts']:
            x, y = text.get_position(); text.set_position((x, y + dy))
        for legend in row['legends']: shift_legend(legend, dy)
        shifts.append(dy * height_pt)
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    final = [rowbox(row) for row in rows]
    after = [panel_footprint(fig, axes, renderer) for _, axes in groups]
    residuals = []
    for edge in ['x0', 'x1', 'y0', 'y1']:
        for i, slot in enumerate(slots):
            cohort = [j for j, s in enumerate(slots)
                      if abs(getattr(s, edge) - getattr(slot, edge)) < .012
                      and after[j] is not None]
            if len(cohort) > 1 and i == min(cohort):
                values = [getattr(after[j], edge) for j in cohort]
                residuals.append({'edge': edge, 'panels': cohort,
                                  'spread': max(values) - min(values)})
    overlaps = []
    for i, a in enumerate(after):
        for j in range(i + 1, len(after)):
            if a is None or after[j] is None: continue
            intersection = Bbox.intersection(a, after[j])
            if intersection is not None and intersection.width > .001 and intersection.height > .001:
                overlaps.append({'panels': [i, j], 'intersection': list(intersection.extents)})
    physical_change = max(abs(ax.get_position().width - original_size[ax][0])
                          * fig.get_figwidth() * 72 +
                          abs(ax.get_position().height - original_size[ax][1]) * height_pt
                          for ax in fig.axes)
    assert _scientific_state(fig) == science, 'Scientific artists changed during row packing'
    assert physical_change < .01, ('Data axes resized during packing', physical_change)
    alignment['before_row_packing'] = alignment['after']
    alignment['after'] = [list(b.extents) if b is not None else None for b in after]
    alignment['edge_residuals'] = residuals
    alignment['footprint_overlaps'] = overlaps
    alignment['passed'] = all(r['spread'] <= alignment.get('tolerance', .002)
                              for r in residuals) and not overlaps
    assert alignment['passed'], 'Packing broke calibrated outer boundaries'
    return {'method': 'physical translations of complete rows', 'rows': len(rows),
            'target_clearance_pt': row_gap_pt,
            'before_clearance_pt': [(a.y0 - b.y1) * height_pt for a, b in zip(initial, initial[1:])],
            'after_clearance_pt': [(a.y0 - b.y1) * height_pt for a, b in zip(final, final[1:])],
            'row_shifts_pt': shifts, 'ornaments': ornaments,
            'content_height_before_pt': original_content_height,
            'content_height_after_pt': fig.get_tightbbox(renderer).height * 72,
            'max_axis_size_change_pt': physical_change,
            'scientific_artists_preserved': True, 'passed': True}
