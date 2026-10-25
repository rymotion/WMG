import 'dart:math' as math;

import 'package:flutter/gestures.dart' show PointerPanZoomUpdateEvent;
import 'package:flutter/material.dart';
import 'package:wmg_detector/wmg_detector.dart';

import '../models/on_device_detection.dart';
import '../models/spectrogram.dart';

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
  });

  final List<ScoreSeries> scores;
  final double startS;
  final double endS;
  final double height;

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
    this.height = 128,
    this.scoreHeight = 56,
    this.width = double.infinity,
  });

  final Spectrogram spectrogram;
  final SpectrogramController? controller;

  /// Boundaries marked with a thin line at each start.
  final List<SpectrogramRegion> regions;

  /// Shaded, e.g. the selected clip.
  final SpectrogramRegion? highlight;

  /// Window P(AI) series; the chart is hidden when empty.
  final List<ScoreSeries> scores;
  final double height;
  final double scoreHeight;
  final double width;

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
    // A new session starts from scratch.
    if (_liveEnd < _liveEndOf(old)) _followLive();
  }

  @override
  void dispose() {
    if (widget.controller?._view == this) widget.controller!._view = null;
    _anim.dispose();
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
                tooltip: 'Back to live',
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
                          startS: start,
                          endS: end,
                          regions: widget.regions,
                          highlight: widget.highlight,
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

  Widget _frame(double height, Widget child) => ClipRRect(
    borderRadius: BorderRadius.circular(8),
    child: SizedBox(
      height: height,
      width: double.infinity,
      child: ColoredBox(color: Colors.black, child: child),
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
    required this.startS,
    required this.endS,
    required this.regions,
    required this.highlight,
  });

  final Spectrogram spec;

  /// Stream time at the left and right edges (fractional while scrubbing).
  final double startS;
  final double endS;
  final List<SpectrogramRegion> regions;
  final SpectrogramRegion? highlight;

  @override
  void paint(Canvas canvas, Size size) {
    final xOfTime = _timeToX(size, startS, endS);
    final colWidth = size.width / ((endS - startS) / spec.hopS);
    // Zoomed out, frames narrower than a pixel are skipped, so cost stays ~width × bins.
    final stride = math.max(1, (1 / colWidth).ceil());

    final paint = Paint()..isAntiAlias = false;
    final floor = spec.dbFloor;
    // Only the frames on screen; the session can hold tens of thousands.
    final last = (endS / spec.hopS).ceil();
    final first = (startS / spec.hopS).floor();
    // Aligned to the stride so the sampled frames don't shimmer while scrolling.
    for (var f = first - first % stride; f < last; f += stride) {
      final frame = spec.frameAt(f);
      if (frame == null || frame.isEmpty) continue;
      final bins = frame.length;
      final rowHeight = size.height / bins;
      final left = xOfTime(f * spec.hopS);
      for (var b = 0; b < bins; b++) {
        // Absolute dB: floor → 0, ~0 dBFS → 1. Same mapping as the guide's canvas example.
        final v = ((frame[b] - floor) / -floor).clamp(0.0, 1.0);
        paint.color = HSLColor.fromAHSL(
          1,
          (280 - v * 240) % 360,
          0.9,
          v * 0.6,
        ).toColor();
        canvas.drawRect(
          Rect.fromLTWH(
            left,
            size.height - (b + 1) * rowHeight,
            colWidth * stride + 0.5,
            rowHeight + 0.5,
          ),
          paint,
        );
      }
    }

    _paintOverlays(canvas, size, xOfTime, regions, highlight);
  }

  @override
  bool shouldRepaint(_SpectrogramPainter old) =>
      old.spec != spec ||
      old.startS != startS ||
      old.endS != endS ||
      old.highlight != highlight ||
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
  });

  final List<ScoreSeries> series;
  final double startS;
  final double endS;
  final List<SpectrogramRegion> regions;
  final SpectrogramRegion? highlight;

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

    _paintOverlays(canvas, size, xOfTime, regions, highlight);

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
      !_sameRegions(old.regions, regions) ||
      old.series.length != series.length ||
      [
        for (var i = 0; i < series.length; i++)
          !identical(old.series[i].windows, series[i].windows) ||
              old.series[i].color != series[i].color,
      ].any((changed) => changed);
}
