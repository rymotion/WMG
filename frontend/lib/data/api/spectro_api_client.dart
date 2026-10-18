import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/audio_upload.dart';
import '../models/identification.dart';

class ApiException implements Exception {
  const ApiException(this.message, {this.statusCode});

  final String message;
  final int? statusCode;

  @override
  String toString() => statusCode == null ? message : '$message (HTTP $statusCode)';
}

/// Thin HTTP client for the spectro-api backend.
class SpectroApiClient {
  SpectroApiClient({required this.baseUrl, http.Client? httpClient}) : _http = httpClient ?? http.Client();

  final Uri baseUrl;
  final http.Client _http;

  Uri _uri(String path) => baseUrl.resolve(path);

  /// `POST /audio` with multipart field `file`. Identification starts in the
  /// background; the response has `identification.status == "pending"`.
  Future<AudioUpload> uploadAudio(String filePath, {String? filename}) async {
    final request = http.MultipartRequest('POST', _uri('/audio'))
      ..files.add(await http.MultipartFile.fromPath('file', filePath, filename: filename));
    final response = await http.Response.fromStream(await _http.send(request));
    return AudioUpload.fromJson(_decode(response, expect: 201) as Map<String, dynamic>);
  }

  /// `POST /audio/{id}/identification`
  Future<Identification> getIdentification(String id) async {
    final response = await _http.get(_uri('/audio/$id/identification'));
    return Identification.fromJson(_decode(response) as Map<String, dynamic>);
  }

  /// `POST /audio`
  Future<List<AudioUpload>> listAudio() async {
    final response = await _http.post(_uri('/audio'));
    return AudioUpload.listFromJson(_decode(response) as List<dynamic>);
  }

  /// `GET /audio?session_id=` — every clip saved by one live session, sorted by clip index.
  Future<List<AudioUpload>> listSessionClips(String sessionId) async {
    final response = await _http.get(_uri('/audio').replace(queryParameters: {'session_id': sessionId}));
    return AudioUpload.listFromJson(_decode(response) as List<dynamic>);
  }

  Object? _decode(http.Response response, {int expect = 200}) {
    if (response.statusCode != expect) {
      throw ApiException(_errorMessage(response), statusCode: response.statusCode);
    }
    return jsonDecode(utf8.decode(response.bodyBytes));
  }

  String _errorMessage(http.Response response) {
    try {
      final body = jsonDecode(response.body);
      if (body is Map && body['error'] != null) return body['error'].toString();
      if (body is Map && body['message'] != null) return body['message'].toString();
    } catch (_) {}
    return response.body.isEmpty ? 'Request failed' : response.body;
  }

  void close() => _http.close();
}
