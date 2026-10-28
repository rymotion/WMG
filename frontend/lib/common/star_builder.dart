import 'package:flutter/material.dart';
import 'package:flutter_svg/svg.dart';

class StarCount extends StatelessWidget {
  const StarCount({
    super.key,
    required this.score,
    this.height = 72,
    this.width = 72,
    this.compact = false,
  });

  final double score;
  final double height;
  final double width;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    return _buildScoreComponent(score);
  }

  Widget _buildScoreComponent(double p) {
    List<Widget> stars = _buildStarCount(((p * 100.0) / 20.0).toInt(), p);

    if (p >= 0.9) {
      stars.add(
        SvgPicture.asset(
          'assets/ai-spark-fill.svg',
          colorFilter: ColorFilter.mode(Colors.white, BlendMode.srcIn),
          height: height,
          width: width,
        ),
      );
    }
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        !compact
            ? SvgPicture.asset(
                'assets/music-disclosure.svg',
                height: height,
                width: width,
                colorFilter: ColorFilter.mode(Colors.white, BlendMode.srcIn),
                imageBuilder: (context, child) {
                  return Container(color: Colors.transparent, child: child);
                },
              )
            : const SizedBox.shrink(),
        ...stars,
      ],
    );
  }

  List<Widget> _buildStarCount(int starCount, double p) => List.generate(
    5,
    (index) => SvgPicture.asset(
      index < starCount || p >= 0.9
          ? 'assets/ai-spark-fill.svg'
          : 'assets/ai-spark.svg',
      colorFilter: ColorFilter.mode(Colors.white, BlendMode.srcIn),
      height: height,
      width: width,
    ),
  );
}
