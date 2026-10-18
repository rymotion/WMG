import '../api/live_audio_client.dart';
import '../api/spectro_api_client.dart';
import '../models/audio_upload.dart';

/// Entry point to the backend: live sessions for capture, REST for saved clips.
class IdentificationRepository {
  IdentificationRepository({required SpectroApiClient api, required LiveAudioClient live}) : _api = api, _live = live;

  final SpectroApiClient _api;
  final LiveAudioClient _live;

  Future<LiveAudioSession> openLiveSession() => _live.connect();

  /// All clips saved by one live session, sorted by clip index.
  Future<List<AudioUpload>> sessionClips(String sessionId) => _api.listSessionClips(sessionId);
}
