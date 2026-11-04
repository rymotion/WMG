# Accounts, usage and billing: the backend contract

The handset app (iOS, Android) meters song identification and sells plans. Every limit is
enforced here, on the server; the app only reflects it. Until these routes exist, the app runs
an in-memory copy of these rules (`lib/data/account/fake_account_api.dart`, the default
`--dart-define=ACCOUNT_BACKEND=fake`). Build with `ACCOUNT_BACKEND=http` once they're live.

macOS is the research build and doesn't use any of this.

## Plans

| | Free | Plus | Pro Creator |
|---|---|---|---|
| Song IDs (mic) / month | 10 | unlimited | unlimited |
| Listen window per tap | 10 s | 30 s (Noisy Mode) | 30 s |
| File (video/audio) IDs / month | – | – | 100 |
| History | 7 days, not searchable | forever, searchable | forever, searchable |
| Playlists to Spotify / Apple Music | – | ✓ | ✓ |
| Record now, ID later | – | ✓ | ✓ |
| Multi-song mix scan (≤ 1 h) | – | – | ✓ |
| Pro metadata, CSV / Rekordbox / Serato | – | – | ✓ |
| AI check on device | ✓ no account needed | ✓ | ✓ |

## Prices

One product id per item across the App Store, Google Play and Stripe.

| Product id | Web (Stripe) | In-app (StoreKit / Play) |
|---|---|---|
| `org.activeie.wmg.plus.monthly` | $6.99 | $9.09 (+30%) |
| `org.activeie.wmg.plus.yearly` | $49.99 | $49.99 |
| `org.activeie.wmg.pro.monthly` | $14.99 | $19.49 (+30%) |
| `org.activeie.wmg.pro.yearly` | $119.99 | $119.99 |
| `org.activeie.wmg.credits.mic20` (consumable) | $1.99 | $1.99 |
| `org.activeie.wmg.credits.mic100` (consumable) | $6.99 | $6.99 |
| `org.activeie.wmg.credits.video10` (consumable) | $3.99 | $3.99 |

App Store Connect and Play Console only offer fixed price points: pick the nearest one to the
in-app column. The app always shows the store's own localised price once it has loaded, so
the table in `lib/data/account/catalog.dart` is only a fallback.

## Rules

- **One tap = one ID.** The app reserves a ticket before it opens `/live`. A ticket is spent
  when its session produces a match; a session that matched nothing is refunded.
- **Credits** are used only after the month's allowance runs out, and never expire.
- The month runs from the subscription anniversary (Free: calendar month, UTC).
- **Privacy:** usage is kept to enforce plans and run the service. Listening data is never
  sold or shared.

## Auth

Every route takes `Authorization: Bearer <token>` from the identity provider (to be chosen).
A missing or bad token is `401`. The provider's user id is the account id.

## Routes

### `GET /me` → 200

```json
{
  "user": { "id": "usr_123", "email": "dj@example.com" },
  "tier": "free",                 // free | plus | pro
  "interval": null,               // month | year | null
  "renews_at": null,              // ISO-8601 or null
  "billing_source": null,         // app_store | google_play | stripe | null
  "web_checkout": true,           // may the app link to Stripe Checkout for this user?
  "usage": {
    "period_end": "2026-12-01T00:00:00Z",
    "mic_ids_used": 3,
    "mic_ids_limit": 10,          // null: unlimited
    "upload_ids_used": 0,
    "upload_ids_limit": 0,
    "mic_credits": 0,
    "video_credits": 0
  }
}
```

`web_checkout` is the server's call (by the user's storefront/region). Apple allows a link to
outside purchase for US storefront apps; elsewhere keep it `false`.

### `POST /usage/tickets` → 201

Body `{"kind": "mic"}`. Reserves one identification.

```json
{ "ticket_id": "tkt_abc", "listen_seconds": 10, "usage": { …as in /me… } }
```

`402 {"error": "quota_exhausted", "usage": {…}}` when the allowance and credits are spent.

### `POST /usage/tickets/{id}/release` → 200

Body `{"reason": "no_match" | "failed" | "cancelled"}`. Refunds an unspent ticket; returns
`usage`. Idempotent. Tickets that never reached a lookup expire after 10 minutes on their own.

### `WS /live?…&ticket=tkt_abc`

`/live` must only call ACRCloud for a session carrying a valid, unspent ticket of the caller;
without one it runs with `identify=false` (spectrum and AI scoring only). The ticket is spent
on the session's first match.

For instant results the server should try a lookup early (e.g. after 5 s of audio, then every
5 s) instead of waiting for the 30 s clip, and send `identification` as soon as one matches.
The app stops listening at the first match.

### `POST /usage/network` → 204

```json
{ "entries": [
  { "route": "WS /live", "calls": 2, "failures": 0, "bytes_up": 1280000, "bytes_down": 0 },
  { "route": "GET /audio/:id/identification", "calls": 4, "failures": 1, "bytes_up": 0, "bytes_down": 2310 }
] }
```

Sent every 5 minutes while signed in. Routes are templates: no ids, query strings or content.

### `POST /billing/checkout` → 200

Body `{"product_id": "org.activeie.wmg.plus.yearly"}`. Creates a Stripe Checkout Session
(`mode: subscription` or `payment` for credit packs, `client_reference_id` = user id) and
returns `{"url": "https://checkout.stripe.com/…"}`. The app opens it in the browser and
refreshes `/me` when it comes back to the foreground.

### `POST /billing/portal` → 200

Returns `{"url": …}` for Stripe's customer portal (web subscribers only).

### `POST /billing/store-purchases` → 200

```json
{
  "platform": "app_store",                // app_store | google_play
  "product_id": "org.activeie.wmg.plus.monthly",
  "verification_data": "<StoreKit 2 signed transaction JWS | Play purchase token>",
  "transaction_id": "2000000123456789"
}
```

Verify with the App Store Server API / Google Play Developer API, apply it, return the
account as in `GET /me`. The app finishes the store transaction only after a `200`, so the
store redelivers anything the server didn't confirm. Must be idempotent per transaction.

### Webhooks (server-side only)

- Stripe: `checkout.session.completed`, `customer.subscription.updated|deleted`,
  `invoice.paid` → set tier, interval, renewal, credits.
- App Store Server Notifications v2 and Google Play RTDN → renewals, refunds, cancellations.
