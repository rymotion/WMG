import 'dart:async';

import 'package:equatable/equatable.dart';

class AuthUser extends Equatable {
  const AuthUser({required this.id, this.email});

  final String id;
  final String? email;

  @override
  List<Object?> get props => [id, email];
}

/// The identity provider, behind one seam. Song identification and export need a signed-in
/// user; on-device AI detection never does.
///
/// TODO(auth): implement this for the chosen provider and pick it in `main.dart`.
abstract interface class AuthSession {
  AuthUser? get currentUser;

  /// Emits on sign-in and sign-out.
  Stream<AuthUser?> get changes;

  /// Bearer token for the backend; null when signed out.
  Future<String?> token();

  /// Runs the provider's sign-in UI. Returns the user, or null if they cancelled.
  Future<AuthUser?> signIn();

  Future<void> signOut();

  /// False until a provider is wired in; the app then explains accounts are coming.
  bool get isAvailable;
}

/// Stand-in until the real provider lands: no accounts exist.
class UnavailableAuthSession implements AuthSession {
  const UnavailableAuthSession();

  @override
  AuthUser? get currentUser => null;

  @override
  Stream<AuthUser?> get changes => const Stream.empty();

  @override
  Future<String?> token() async => null;

  @override
  Future<AuthUser?> signIn() async => null;

  @override
  Future<void> signOut() async {}

  @override
  bool get isAvailable => false;
}

/// One local, always-accepted user for development against the fake account backend.
class DevAuthSession implements AuthSession {
  DevAuthSession({AuthUser? signedIn}) : _user = signedIn;

  static const devUser = AuthUser(id: 'dev-user', email: 'dev@localhost');

  AuthUser? _user;
  final _changes = StreamController<AuthUser?>.broadcast();

  @override
  AuthUser? get currentUser => _user;

  @override
  Stream<AuthUser?> get changes => _changes.stream;

  @override
  Future<String?> token() async => _user == null ? null : 'dev-token';

  @override
  Future<AuthUser?> signIn() async {
    _changes.add(_user = devUser);
    return _user;
  }

  @override
  Future<void> signOut() async => _changes.add(_user = null);

  @override
  bool get isAvailable => true;
}
