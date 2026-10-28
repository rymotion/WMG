import '../../../data/labels/recording_label.dart';
import 'dart:math' as math;

import 'package:flutter/gestures.dart' show PointerPanZoomUpdateEvent;
import 'package:flutter/material.dart';
import 'package:wmg_detector/wmg_detector.dart';

import '../models/on_device_detection.dart';
import '../models/spectrogram.dart';
import 'spectrogram_tiles.dart';

/// A span of stream time drawn over the spectrogram, e.g. one saved clip.
typedef SpectrogramRegion = ({double startS, double endS});

/// One detector's window scores, drawn as P(AI) over time under the
/// spectrogram. [windowS] is the window length, so dots sit at window centres.
typedef ScoreSeries = ({
  List<WindowScore> windows,
  double windowS,
  Color color,
  String label,
});

/// The 10 s and 5 s models' windows as chart series, in theme colours.
List<ScoreSeries> scoreSeries(OnDeviceDetection d, ColorScheme colors) => [
  (windows: d.windows5s, windowS: 5, color: colors.tertiary, label: '5 s'),
  (windows: d.windows, windowS: 10, color: colors.primary, label: '10 s'),
];

/// A fixed window P(AI) chart over `[startS, endS]`, drawn like the band under
/// [SpectrogramView] (e.g. one clip's scores on its result card).
class WindowScoreChart extends StatelessWidget {
  const WindowScoreChart({
    super.key,
    required this.scores,
    required this.startS,
    required this.endS,
    this.height = 56,
    this.label,
  });

  final List<ScoreSeries> scores;
  final double startS;
  final double endS;
  final double height;

  /// The recording's label: shades its AI-share band and marks the chart in its colour.
  final AiCategory? label;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: SizedBox(
        height: height,
        width: double.infinity,
        child: ColoredBox(
          color: Colors.black,
          child: CustomPaint(
            painter: _ScorePainter(
              series: scores,
              startS: startS,
              endS: endS,
              regions: const [],
              highlight: null,
              label: label,
            ),
          ),
        ),
      ),
    );
  }
}

/// Lets a parent move a [SpectrogramView], like a [ScrollController].
class SpectrogramController {
  _SpectrogramViewState? _view;

  /// Animates so [seconds] sits near the left edge of the view.
  void seekTo(double seconds) => _view?._seekTo(seconds);

  /// Returns to following the newest frame.
  void followLive() => _view?._followLive();
}

/// Live mel spectrogram: newest frame on the right, low bins at the bottom.
/// With [scores], a window P(AI) chart sits underneath on the same time axis.
/// Drag horizontally to scrub back through the session; dragging to the end
/// resumes following the live edge.
class SpectrogramView extends StatefulWidget {
  const SpectrogramView({
    super.key,
    required this.spectrogram,
    this.controller,
    this.regions = const [],
    this.highlight,
    this.scores = const [],
    this.playheadS,
    this.onTapTime,
    this.endLabel = 'Back to live',
    this.height = 128,
    this.scoreHeight = 56,
    this.width = double.infinity,
    this.label,
  });

  final Spectrogram spectrogram;
  final SpectrogramController? controller;

  /// Boundaries marked with a thin line at each start.
  final List<SpectrogramRegion> regions;

  /// Shaded, e.g. the selected clip.
  final SpectrogramRegion? highlight;

  /// Window P(AI) series; the chart is hidden when empty.
  final List<ScoreSeries> scores;

  /// Playback position, drawn as a line; the view follows it when it leaves the screen.
  final double? playheadS;

  /// Called with the stream time under a tap, e.g. to seek playback there.
  final ValueChanged<double>? onTapTime;

  /// Tooltip of the button that jumps back to the newest audio.
  final String endLabel;
  final double height;
  final double scoreHeight;
  final double width;

  /// The recording's label, drawn on the P(AI) chart (see [WindowScoreChart.label]).
  final AiCategory? label;

  @override
  State<SpectrogramView> createState() => _SpectrogramViewState();
}

class _SpectrogramViewState extends State<SpectrogramView>
    with SingleTickerProviderStateMixin {
  /// Stream time at the view's right edge; null follows the live edge.
  double? _end;
  late final AnimationController _anim = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 350),
  );
  Animation<double>? _seek;

  /// Fraction of the view left of a seek target, so its start stays visible.
  static const _seekLead = 0.1;

  Spectrogram get _spec => widget.spectrogram;

  /// Seconds across the view's width; pinch changes it, double-tap resets it.
  late double _spanS = _defaultSpanS;

  /// The spectrogram's native width (~10 s).
  double get _defaultSpanS => _spec.visibleFrames * _spec.hopS;

  static const _minSpanS = 2.0;
  static const _maxSpanS = 120.0;

  /// Image tiles of the spectrogram, repainted as they become ready.
  late final _tiles = SpectrogramTiles(
    onChanged: () {
      if (mounted) setState(() => _tileRevision++);
    },
  );
  int _tileRevision = 0;

  /// Touch points on the view, for pinch-to-zoom.
  final _pointers = <int, Offset>{};
  ({double distance, double spanS, double fx, double timeS})? _pinch;

  double get _liveEnd => _liveEndOf(widget);

  double get _viewEnd => _seek?.value ?? _end ?? _liveEnd;

  bool get _hasScores => widget.scores.any((s) => s.windows.isNotEmpty);

  /// Newest recorded time: the spectrogram's end, or the last scored window's
  /// when offline there is no spectrogram.
  static double _liveEndOf(SpectrogramView w) {
    var end = w.spectrogram.durationS;
    for (final s in w.scores) {
      if (s.windows.isNotEmpty) {
        end = math.max(end, s.windows.last.startS + s.windowS);
      }
    }
    return end;
  }

  @override
  void initState() {
    super.initState();
    widget.controller?._view = this;
    _tiles.update(widget.spectrogram);
    _anim.addListener(() => setState(() {}));
    _anim.addStatusListener((status) {
      if (status == AnimationStatus.completed) _settle(_seek!.value);
    });
  }

  @override
  void didUpdateWidget(SpectrogramView old) {
    super.didUpdateWidget(old);
    if (old.controller != widget.controller) {
      if (old.controller?._view == this) old.controller!._view = null;
      widget.controller?._view = this;
    }
    _tiles.update(widget.spectrogram);
    // A new session starts from scratch.
    if (_liveEnd < _liveEndOf(old)) _followLive();
    _keepPlayheadInView(old.playheadS);
  }

  /// Pages the view along with playback, unless the user is holding it.
  void _keepPlayheadInView(double? previous) {
    final t = widget.playheadS;
    final held = _pointers.isNotEmpty || _seek != null;
    if (t == null || t == previous || held) return;
    final end = _viewEnd;
    if (t >= end - _spanS && t <= end) return;
    _end = _clampEnd(t + _spanS * (1 - _seekLead));
    if (_end! >= _liveEnd) _end = null;
  }

  @override
  void dispose() {
    if (widget.controller?._view == this) widget.controller!._view = null;
    _anim.dispose();
    _tiles.dispose();
    super.dispose();
  }

  double _clampEnd(double end) =>
      end.clamp(math.min(_spanS, _liveEnd), _liveEnd);

  /// Keeps the right edge within the recording; null at the live edge.
  void _settle(double end) {
    _seek = null;
    final clamped = _clampEnd(end);
    setState(() => _end = clamped >= _liveEnd ? null : clamped);
  }

  void _seekTo(double seconds) {
    _seek = Tween(
      begin: _viewEnd,
      end: _clampEnd(seconds + _spanS * (1 - _seekLead)),
    ).animate(CurvedAnimation(parent: _anim, curve: Curves.easeOutCubic));
    _anim.forward(from: 0);
  }

  void _followLive() {
    _anim.stop();
    _seek = null;
    if (mounted) setState(() => _end = null);
  }

  void _onDrag(DragUpdateDetails d, double width) {
    if (_pinch != null) return; // two fingers zoom instead of scrubbing
    _anim.stop();
    _settle(_viewEnd - d.delta.dx * _spanS / width);
  }

  /// Remembers the time under [focal] so zooming keeps it in place.
  void _startPinch(Offset focal, double width, {double distance = 1}) {
    final fx = (focal.dx / width).clamp(0.0, 1.0);
    _pinch = (
      distance: distance,
      spanS: _spanS,
      fx: fx,
      timeS: _viewEnd - _spanS * (1 - fx),
    );
  }

  /// Zooms so the view shows `pinch.spanS / scale` seconds. Following live
  /// keeps the right edge on the newest audio; otherwise the pinched point
  /// stays under the fingers.
  void _zoomTo(double scale) {
    final pinch = _pinch;
    if (pinch == null || scale <= 0) return;
    _anim.stop();
    _seek = null;
    final following = _end == null;
    setState(() {
      _spanS = (pinch.spanS / scale).clamp(_minSpanS, _maxSpanS);
    });
    if (!following) _settle(pinch.timeS + _spanS * (1 - pinch.fx));
  }

  void _onPointerDown(PointerDownEvent e, double width) {
    _pointers[e.pointer] = e.localPosition;
    if (_pointers.length == 2) {
      final [a, b] = _pointers.values.toList();
      _startPinch((a + b) / 2, width, distance: (a - b).distance);
    }
  }

  void _onPointerMove(PointerMoveEvent e) {
    if (!_pointers.containsKey(e.pointer)) return;
    _pointers[e.pointer] = e.localPosition;
    final pinch = _pinch;
    if (pinch == null || _pointers.length < 2) return;
    final [a, b, ...] = _pointers.values.toList();
    _zoomTo((a - b).distance / pinch.distance);
  }

  void _onPointerEnd(PointerEvent e) {
    _pointers.remove(e.pointer);
    if (_pointers.length < 2) _pinch = null;
  }

  /// Trackpad: pinch zooms, horizontal two-finger swipe scrubs.
  void _onPanZoomUpdate(PointerPanZoomUpdateEvent e, double width) {
    if (e.scale != 1) {
      _zoomTo(e.scale);
    } else if (e.localPanDelta.dx != 0) {
      _anim.stop();
      _settle(_viewEnd + e.localPanDelta.dx * _spanS / width);
    }
  }

  void _resetZoom() {
    final following = _end == null;
    setState(() => _spanS = _defaultSpanS);
    if (!following) _settle(_viewEnd);
  }

  @override
  Widget build(BuildContext context) {
    final showScores = _hasScores;
    final dim = TextStyle(color: Colors.white.withValues(alpha: 0.6));
    if (_spec.isEmpty && !showScores) {
      return _frame(
        widget.height,
        Center(child: Text('Waiting for spectrum…', style: dim)),
      );
    }
    final end = _viewEnd;
    final start = end - _spanS;
    if (!_spec.isEmpty) {
      final perTile = SpectrogramTiles.tileFrames * _spec.hopS;
      _tiles.request((start / perTile).floor(), (end / perTile).floor());
    }
    return _frame(
      widget.height + (showScores ? 1 + widget.scoreHeight : 0),
      Stack(
        children: [
          Positioned.fill(child: _gestureArea(start, end, showScores, dim)),
          // Outside the gesture area so its tap isn't held up by double-tap.
          if (_end != null || _seek != null)
            Positioned(
              right: 4,
              top: 4,
              child: IconButton(
                tooltip: widget.endLabel,
                visualDensity: VisualDensity.compact,
                color: Colors.white,
                icon: const Icon(Icons.fast_forward_rounded),
                onPressed: _followLive,
              ),
            ),
        ],
      ),
    );
  }

  Widget _gestureArea(
    double start,
    double end,
    bool showScores,
    TextStyle dim,
  ) {
    return LayoutBuilder(
      builder: (context, constraints) => Listener(
        onPointerDown: (e) => _onPointerDown(e, constraints.maxWidth),
        onPointerMove: _onPointerMove,
        onPointerUp: _onPointerEnd,
        onPointerCancel: _onPointerEnd,
        onPointerPanZoomStart: (e) =>
            _startPinch(e.localPosition, constraints.maxWidth),
        onPointerPanZoomUpdate: (e) =>
            _onPanZoomUpdate(e, constraints.maxWidth),
        onPointerPanZoomEnd: (_) => _pinch = null,
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onHorizontalDragUpdate: (d) => _onDrag(d, constraints.maxWidth),
          onDoubleTap: _resetZoom,
          onTapUp: widget.onTapTime == null
              ? null
              : (d) => widget.onTapTime!(
                  math.max(
                    0,
                    start +
                        d.localPosition.dx /
                            constraints.maxWidth *
                            (end - start),
                  ),
                ),
          child: Column(
            children: [
              SizedBox(
                height: widget.height,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    if (_spec.isEmpty)
                      Center(
                        child: Text('Spectrogram needs the server', style: dim),
                      )
                    else
                      CustomPaint(
                        painter: _SpectrogramPainter(
                          spec: _spec,
                          tiles: _tiles,
                          tileRevision: _tileRevision,
                          startS: start,
                          endS: end,
                          regions: widget.regions,
                          highlight: widget.highlight,
                          playheadS: widget.playheadS,
                        ),
                      ),
                    Positioned(
                      left: 8,
                      top: 6,
                      child: _Label(
                        '${_mmss(math.max(0, start))} – ${_mmss(end)}',
                      ),
                    ),
                  ],
                ),
              ),
              if (showScores) ...[
                ColoredBox(
                  color: Colors.white.withValues(alpha: 0.2),
                  child: const SizedBox(height: 1, width: double.infinity),
                ),
                SizedBox(
                  height: widget.scoreHeight,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      CustomPaint(
                        painter: _ScorePainter(
                          series: widget.scores,
                          startS: start,
                          endS: end,
                          regions: widget.regions,
                          highlight: widget.highlight,
                          playheadS: widget.playheadS,
                          label: widget.label,
                        ),
                      ),
                      const Positioned(left: 8, top: 2, child: _Label('P(AI)')),
                      Positioned(
                        right: 8,
                        top: 2,
                        child: Row(
                          children: [
                            for (final s in widget.scores)
                              if (s.windows.isNotEmpty)
                                Padding(
                                  padding: const EdgeInsets.only(left: 8),
                                  child: _Label(s.label, color: s.color),
                                ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  // Its own layer, so the page around it (the mic level, timers) never repaints it.
  Widget _frame(double height, Widget child) => RepaintBoundary(
    child: ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: SizedBox(
        height: height,
        width: double.infinity,
        child: ColoredBox(color: Colors.black, child: child),
      ),
    ),
  );

  static String _mmss(double s) =>
      '${s ~/ 60}:${(s % 60).floor().toString().padLeft(2, '0')}';
}

class _Label extends StatelessWidget {
  const _Label(this.text, {this.color});

  final String text;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: TextStyle(
        color: color ?? Colors.white.withValues(alpha: 0.7),
        fontSize: 11,
        fontFeatures: const [FontFeature.tabularFigures()],
      ),
    );
  }
}

/// Maps stream time to x for a view showing `[startS, endS]`.
double Function(double) _timeToX(Size size, double startS, double endS) =>
    (s) => (s - startS) / (endS - startS) * size.width;

/// Selected-clip highlight and clip-start markers, drawn the same in both bands.
void _paintOverlays(
  Canvas canvas,
  Size size,
  double Function(double) xOfTime,
  List<SpectrogramRegion> regions,
  SpectrogramRegion? highlight,
  double? playheadS,
) {
  if (highlight != null) {
    final rect = Rect.fromLTRB(
      xOfTime(highlight.startS),
      0,
      xOfTime(highlight.endS),
      size.height,
    );
    canvas.drawRect(
      rect,
      Paint()..color = Colors.white.withValues(alpha: 0.15),
    );
    canvas.drawRect(
      rect.deflate(1),
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2
        ..color = Colors.white.withValues(alpha: 0.8),
    );
  }

  final marker = Paint()
    ..color = Colors.white.withValues(alpha: 0.35)
    ..strokeWidth = 1;
  for (final r in regions) {
    final x = xOfTime(r.startS);
    if (x >= 0 && x <= size.width) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), marker);
    }
  }

  if (playheadS != null) {
    final x = xOfTime(playheadS);
    if (x >= 0 && x <= size.width) {
      canvas.drawLine(
        Offset(x, 0),
        Offset(x, size.height),
        Paint()
          ..color = Colors.amberAccent
          ..strokeWidth = 2,
      );
    }
  }
}

bool _sameRegions(List<SpectrogramRegion> a, List<SpectrogramRegion> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

class _SpectrogramPainter extends CustomPainter {
  _SpectrogramPainter({
    required this.spec,
    required this.tiles,
    required this.tileRevision,
    required this.startS,
    required this.endS,
    required this.regions,
    required this.highlight,
    required this.playheadS,
  });

  final Spectrogram spec;
  final SpectrogramTiles tiles;

  /// Bumped when a tile image lands, so the painter knows to repaint.
  final int tileRevision;

  /// Stream time at the left and right edges (fractional while scrubbing).
  final double startS;
  final double endS;
  final List<SpectrogramRegion> regions;
  final SpectrogramRegion? highlight;
  final double? playheadS;

  @override
  void paint(Canvas canvas, Size size) {
    final xOfTime = _timeToX(size, startS, endS);
    const frames = SpectrogramTiles.tileFrames;
    final perTile = frames * spec.hopS;
    // Nearest-neighbour keeps cells crisp when zoomed in; zoomed out the GPU downsamples.
    final paint = Paint()..filterQuality = FilterQuality.none;
    canvas.save();
    canvas.clipRect(Offset.zero & size);
    for (
      var k = (startS / perTile).floor();
      k <= (endS / perTile).floor();
      k++
    ) {
      final image = k < 0 ? null : tiles[k];
      if (image == null) continue;
      canvas.drawImageRect(
        image,
        Rect.fromLTWH(0, 0, frames.toDouble(), image.height.toDouble()),
        Rect.fromLTRB(
          xOfTime(k * perTile),
          0,
          xOfTime((k + 1) * perTile),
          size.height,
        ),
        paint,
      );
    }
    canvas.restore();

    _paintOverlays(canvas, size, xOfTime, regions, highlight, playheadS);
  }

  @override
  bool shouldRepaint(_SpectrogramPainter old) =>
      old.spec != spec ||
      old.tileRevision != tileRevision ||
      old.startS != startS ||
      old.endS != endS ||
      old.highlight != highlight ||
      old.playheadS != playheadS ||
      !_sameRegions(old.regions, regions);
}

/// Window P(AI) over time, as in the training script's score plot: a dot per
/// window at its centre, 0..1 on the y axis, dashed line at 0.5.
class _ScorePainter extends CustomPainter {
  _ScorePainter({
    required this.series,
    required this.startS,
    required this.endS,
    required this.regions,
    required this.highlight,
    this.playheadS,
    this.label,
  });

  final List<ScoreSeries> series;

  /// Labelled recordings get their category's AI-share band shaded (e.g. 10–50% for
  /// AI-assisted) with dashed edges, and a strip in the category colour along the top.
  final AiCategory? label;
  final double startS;
  final double endS;
  final List<SpectrogramRegion> regions;
  final SpectrogramRegion? highlight;
  final double? playheadS;

  /// Keeps dots at 0 and 1 from being clipped.
  static const _pad = 6.0;

  @override
  void paint(Canvas canvas, Size size) {
    final xOfTime = _timeToX(size, startS, endS);
    double yOf(double p) => _pad + (1 - p) * (size.height - 2 * _pad);

    final grid = Paint()
      ..color = Colors.white.withValues(alpha: 0.15)
      ..strokeWidth = 1;
    for (final p in const [0.0, 1.0]) {
      canvas.drawLine(Offset(0, yOf(p)), Offset(size.width, yOf(p)), grid);
    }
    final mid = yOf(0.5);
    final dash = Paint()
      ..color = Colors.white.withValues(alpha: 0.4)
      ..strokeWidth = 1;
    for (var x = 0.0; x < size.width; x += 8) {
      canvas.drawLine(Offset(x, mid), Offset(x + 4, mid), dash);
    }

    final l = label;
    if (l != null) {
      final band = Rect.fromLTRB(0, yOf(l.aiShareMax), size.width, yOf(l.aiShareMin));
      canvas.drawRect(band, Paint()..color = l.color.withValues(alpha: 0.18));
      final edge = Paint()
        ..color = l.color.withValues(alpha: 0.7)
        ..strokeWidth = 1;
      for (final y in [band.top, band.bottom]) {
        for (var x = 0.0; x < size.width; x += 6) {
          canvas.drawLine(Offset(x, y), Offset(x + 3, y), edge);
        }
      }
      canvas.drawRect(Rect.fromLTWH(0, 0, size.width, 3), Paint()..color = l.color);
    }

    _paintOverlays(canvas, size, xOfTime, regions, highlight, playheadS);

    for (final s in series) {
      final line = Path();
      final dots = <Offset>[];
      for (final w in s.windows) {
        final centre = w.startS + s.windowS / 2;
        // On screen plus one window either side, so the line runs off the edges.
        if (centre < startS - s.windowS || centre > endS + s.windowS) continue;
        final pt = Offset(xOfTime(centre), yOf(sigmoid(w.logit)));
        dots.isEmpty ? line.moveTo(pt.dx, pt.dy) : line.lineTo(pt.dx, pt.dy);
        dots.add(pt);
      }
      canvas.drawPath(
        line,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5
          ..color = s.color,
      );
      final dot = Paint()..color = s.color;
      for (final pt in dots) {
        canvas.drawCircle(pt, 1.5, dot);
      }
    }
  }

  @override
  bool shouldRepaint(_ScorePainter old) =>
      old.startS != startS ||
      old.endS != endS ||
      old.highlight != highlight ||
      old.label != label ||
      old.playheadS != playheadS ||
      !_sameRegions(old.regions, regions) ||
      old.series.length != series.length ||
      [
        for (var i = 0; i < series.length; i++)
          !identical(old.series[i].windows, series[i].windows) ||
              old.series[i].color != series[i].color,
      ].any((changed) => changed);
}
