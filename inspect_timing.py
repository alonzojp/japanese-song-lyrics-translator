import json, os, sys

cl_path = os.path.join(os.environ.get('TEMP', '/tmp'), 'canonical_lines.json')
aw_path = os.path.join(os.environ.get('TEMP', '/tmp'), 'aligned_words_full.json')

cl = json.loads(open(cl_path, encoding='utf-8').read())
aw = json.loads(open(aw_path, encoding='utf-8').read())

lines = cl['lines']
segs  = aw['segments']

_MIN_WORD_CONFIDENCE      = 0.6
_SEGMENT_OVERRIDE_GAP_S   = 0.6
_SEGMENT_OVERRIDE_MAX_CONF= 0.85
_VOCAL_OVERLAP_S          = 0.5
_VIS_MIN_S                = 1.2
_CONTIGUOUS_THRESH_S      = 0.10
_NEAR_CONTIGUOUS_THRESH_S = 1.5

def classify_gap(g):
    if g > 4.0: return 'STRONG'
    if g > 1.5: return 'SOFT'
    return 'CONT'

def pick_anchor_trace(words, a_start, a_end, overlap_ceiling):
    hard_floor   = a_start + _VIS_MIN_S * 0.5
    hard_ceiling = min(overlap_ceiling, a_end + 0.1)
    def _valid(t): return hard_floor < t <= hard_ceiling
    candidates = []
    last_word_end_raw = None
    if words:
        last = words[-1]
        last_end = last.get('end')
        if last_end is not None:
            last_word_end_raw = float(last_end)
            if _valid(float(last_end)):
                conf = float(last.get('score', 1.0))
                dur  = float(last_end) - float(last.get('start', last_end))
                sc = 5.0 + conf * 1.5
                notes = []
                if conf < _MIN_WORD_CONFIDENCE: sc -= 4.0; notes.append('-4(low_conf)')
                elif conf < _SEGMENT_OVERRIDE_MAX_CONF: sc -= 2.0; notes.append('-2(mod_conf)')
                if dur < 0.05: sc -= 1.5; notes.append('-1.5(short_token)')
                candidates.append((float(last_end), 'last_word', sc, 'conf=%.2f %s' % (conf, '  '.join(notes))))
    if len(words) >= 2:
        sl = words[-2]
        sl_end = sl.get('end')
        if sl_end is not None:
            sl_t = min(float(sl_end) + 0.2, overlap_ceiling)
            if _valid(sl_t):
                conf = float(sl.get('score', 1.0))
                sc = 3.0 + conf * 0.5
                candidates.append((sl_t, 'second_last_word', sc, 'conf=%.2f' % conf))
    seg_end = min(a_end, overlap_ceiling)
    if _valid(seg_end):
        sc = 2.0
        note = ''
        if last_word_end_raw is not None:
            tail = seg_end - last_word_end_raw
            if tail > _SEGMENT_OVERRIDE_GAP_S:
                reward = min(tail * 3.5, 4.0)
                sc += reward
                note = '+%.2f(tail=%.2fs)' % (reward, tail)
        else:
            sc += 2.0; note = '+2.0(no_words)'
        candidates.append((seg_end, 'segment_end', sc, note))
    if not candidates:
        return min(a_end, overlap_ceiling), 'segment_end', 2.0, 'fallback', []
    best = max(candidates, key=lambda c: c[2])
    return best[0], best[1], best[2], best[3], candidates

W = 108
print("=" * W)
print("FORENSIC TRACE: Early Line Advance -- Backend + Frontend Analysis")
print("=" * W)

for i, (line, seg) in enumerate(zip(lines, segs)):
    raw_a_start = float(seg['start'])
    raw_a_end   = float(seg['end'])
    words = seg.get('words', [])

    d_start       = float(line['startTime'])
    d_end         = float(line['endTime'])
    saved_a_start = float(line.get('acousticStart', d_start))
    saved_a_end   = float(line.get('acousticEnd', d_end))

    if i + 1 < len(segs):
        next_raw_start = float(segs[i+1]['start'])
    else:
        next_raw_start = raw_a_end + 30.0

    overlap_ceiling = next_raw_start + _VOCAL_OVERLAP_S

    result = pick_anchor_trace(words, saved_a_start, saved_a_end, overlap_ceiling)
    anchor_t, anchor_src, anchor_sc, anchor_note, all_candidates = result

    gap_acoustic = next_raw_start - raw_a_end
    boundary     = classify_gap(gap_acoustic)
    is_cont      = gap_acoustic <= _CONTIGUOUS_THRESH_S
    is_near_cont = (not is_cont) and gap_acoustic <= _NEAR_CONTIGUOUS_THRESH_S

    ends_before_acoustic = d_end < saved_a_end - 0.05
    early_by = saved_a_end - d_end if ends_before_acoustic else 0.0

    next_d_start = float(lines[i+1]['startTime']) if i+1 < len(lines) else None
    gap_display  = (next_d_start - d_end) if next_d_start is not None else None

    start_was_propagated = abs(saved_a_start - raw_a_start) > 0.01

    text = seg.get('text','')[:25].encode('ascii', errors='replace').decode()

    flag = '  <-- EARLY END' if ends_before_acoustic else ''
    print()
    print("-" * W)
    print("L%02d: \"%s\"  boundary=%s%s" % (i, text, boundary, flag))
    print()
    print("  BACKEND -----------------------------------------------------------------------")
    print("  WhisperX raw:    %.3f -> %.3f  (dur=%.3fs)" % (raw_a_start, raw_a_end, raw_a_end - raw_a_start))
    prop = '  <-- propagated from prev line' if start_was_propagated else ''
    print("  Saved acoustic:  %.3f -> %.3f%s" % (saved_a_start, saved_a_end, prop))
    print("  Display window:  %.3f -> %.3f  (dur=%.3fs)" % (d_start, d_end, d_end - d_start))
    if ends_before_acoustic:
        print("  ** ENDS %.3fs BEFORE acousticEnd=%.3f **" % (early_by, saved_a_end))
    print("  Gap to next:     %.3fs raw  |  ceiling=%.3f" % (gap_acoustic, overlap_ceiling))
    print()
    print("  Anchor pool scores:")
    for cand in all_candidates:
        ct, csrc, csc, cnote = cand[0], cand[1], cand[2], cand[3]
        w = " <-- WINNER" if csrc == anchor_src else ""
        print("    %-22s t=%.3f  score=%+.2f  %s%s" % (csrc, ct, csc, cnote, w))
    print()
    print("  FRONTEND -----------------------------------------------------------------------")
    print("  Active when:     %.3f <= t <= %.3f" % (d_start, d_end))
    print("  Becomes active:  t = %.3f" % d_start)
    print("  Becomes inactive: t > %.3f" % d_end)
    if next_d_start is not None:
        print("  Next line starts: t = %.3f" % next_d_start)
        if gap_display is not None and gap_display > 0.05:
            print("  Display gap:      %.3fs  <-- DARK WINDOW (neither line highlighted)" % gap_display)
        elif gap_display is not None and abs(gap_display) <= 0.05:
            print("  Display gap:      %.3fs  (seamless transition)" % gap_display)
        elif gap_display is not None and gap_display < -0.001:
            ovlp = -gap_display
            print("  OVERLAP: next.startTime=%.3f < this.endTime=%.3f  (overlap=%.3fs)" % (next_d_start, d_end, ovlp))
            print("  --> binary search returns NEXT at t=%.3f  (%.3fs BEFORE this endTime expires)" % (next_d_start, ovlp))
            print("  --> THIS IS A FRONTEND EARLY-ADVANCE: triggered by next.startTime, not endTime")

print()
print("=" * W)
print("SUMMARY: Lines where displayEnd < acousticEnd")
print("=" * W)
early_lines = []
for i, line in enumerate(lines):
    d_end       = float(line['endTime'])
    saved_a_end = float(line.get('acousticEnd', d_end))
    if d_end < saved_a_end - 0.05:
        d_start      = float(line['startTime'])
        next_d_start = float(lines[i+1]['startTime']) if i+1 < len(lines) else None
        gap_display  = (next_d_start - d_end) if next_d_start is not None else None
        text = segs[i].get('text','')[:20].encode('ascii', errors='replace').decode()
        early_lines.append((i, text, d_end, saved_a_end, saved_a_end - d_end, next_d_start, gap_display))
        print("  L%02d '%s': displayEnd=%.3f  acousticEnd=%.3f  early_by=%.3fs  next.startTime=%s" % (
            i, text, d_end, saved_a_end, saved_a_end - d_end, next_d_start))

print()
print("=" * W)
print("OVERLAP CHECK: next.startTime < this.endTime (frontend early advance)")
print("=" * W)
found_overlap = False
for i, line in enumerate(lines[:-1]):
    d_end        = float(line['endTime'])
    next_d_start = float(lines[i+1]['startTime'])
    if next_d_start < d_end - 0.001:
        ovlp = d_end - next_d_start
        text = segs[i].get('text','')[:18].encode('ascii', errors='replace').decode()
        print("  L%02d '%s': endTime=%.3f > L%02d startTime=%.3f  overlap=%.3fs" % (
            i, text, d_end, i+1, next_d_start, ovlp))
        print("  --> frontend switches from L%02d to L%02d at t=%.3f (%.3fs early)" % (i, i+1, next_d_start, ovlp))
        found_overlap = True
if not found_overlap:
    print("  None. All next.startTime >= this.endTime.")

print()
print("=" * W)
print("ROOT CAUSE DETERMINATION")
print("=" * W)
